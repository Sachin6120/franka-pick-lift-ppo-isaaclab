# Copyright (c) 2026 Sachin Kumar Pal

"""Deterministic held-out evaluation for ``Franka-PickLift-Cube-v0``.

Loads a trained RSL-RL checkpoint and runs the deterministic (mean-action) policy against a fixed,
non-random 5x5 grid of cube initial XY positions, distinct from the uniform-random distribution
used during training (``mdp.reset_root_state_uniform`` in the task's ``EventCfg``). Every episode
starts by letting the environment's own reset event run unmodified (so training reset behavior is
never touched), then overwriting the object's root pose/velocity directly via
``write_root_pose_to_sim_index`` / ``write_root_velocity_to_sim_index`` -- a real simulation state
write, immediately verified by reading the object's position back -- so the evaluation reset
procedure is the only thing customized here.

Success is judged purely from simulator state (cube world z), never from reward values, using the
project's own existing success definition (``SUCCESS_LIFT_HEIGHT=0.15`` held for
``SUCCESS_HOLD_STEPS=25`` consecutive control steps -- reused directly from the env cfg's own
constants, not re-derived).

Usage:
    scripts/run.sh scripts/evaluate.py --checkpoint <path/to/model_200.pt> --headless
"""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import json
import statistics
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import franka_pick_lift_rl.tasks  # noqa: F401
from franka_pick_lift_rl.tasks.manager_based.franka_pick_lift_cube.franka_pick_lift_cube_env_cfg import (
    MINIMAL_LIFT_HEIGHT,
    REST_Z,
    SUCCESS_HOLD_STEPS,
    SUCCESS_LIFT_HEIGHT,
)

# The near/far distance bucket used only for this evaluation script's reported CLOSE-fraction
# metrics. Not an active task threshold -- the production reward stack has no gripper-timing
# distance gate at all (see mdp/rewards.py module docstring); this is purely a reporting bucket,
# using the measured EE-to-finger approach offset plus cube half-width as a plausible
# grasp-contact range.
NEAR_FAR_REPORTING_BUCKET_M = 0.07

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

TASK_ID = "Franka-PickLift-Cube-v0"

# Fixed, deterministic 5x5 grid spanning the task's validated reachable reset region
# (x in [0.4, 0.6] m, y in [-0.25, 0.25] m -- see EventCfg's reset_object_position pose_range).
# Margin kept from the extremes so every grid point is comfortably inside the trained
# distribution's support, not at its literal boundary. Deliberately a regular grid (not
# re-sampled from the training distribution) so evaluation conditions are reproducible.
EVAL_X = [0.42, 0.46, 0.50, 0.54, 0.58]
EVAL_Y = [-0.22, -0.11, 0.00, 0.11, 0.22]
SPAWN_Z = 0.055  # matches the training task's object init_state z
POSITION_TOLERANCE_M = 0.002  # required agreement between requested and measured post-reset XY

parser = argparse.ArgumentParser(description="Deterministic held-out evaluation for Franka-PickLift-Cube-v0.")
parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path to the .pt checkpoint.")
parser.add_argument(
    "--output_dir", type=str,
    default="/home/sachin/franka_pick_lift_rl/results/held_out_eval",
    help="Directory to write held_out_results.csv and summary.json into.",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args


@hydra_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False  # no observation noise during eval

    positions = [(x, y) for x in EVAL_X for y in EVAL_Y]  # 25 positions, row-major over (i=x, j=y)
    print(f"[EVAL] {len(positions)} held-out grid positions (x, y):")
    for idx, (x, y) in enumerate(positions):
        gi, gj = idx // len(EVAL_Y), idx % len(EVAL_Y)
        print(f"[EVAL]   [{idx:2d}] grid=({gi},{gj}) xy=({x:+.3f}, {y:+.3f})")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_ID, cfg=env_cfg)
        raw_env = env.unwrapped
        obj = raw_env.scene["object"]
        ee_frame = raw_env.scene["ee_frame"]

        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[EVAL] Loading checkpoint: {checkpoint_path}")
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=raw_env.device)

        step_dt = raw_env.step_dt
        num_steps = int(raw_env.max_episode_length)
        device = raw_env.device

        results: list[dict] = []

        for episode_idx, (grid_x, grid_y) in enumerate(positions):
            gi, gj = episode_idx // len(EVAL_Y), episode_idx % len(EVAL_Y)

            env.reset()
            pose = torch.tensor([[grid_x, grid_y, SPAWN_Z, 0.0, 0.0, 0.0, 1.0]], device=device)
            vel = torch.zeros((1, 6), device=device)
            env_ids = torch.tensor([0], device=device)
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            obj.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
            obj.update(dt=0.0)

            measured_xy = obj.data.root_pos_w.torch[0, :2].tolist()
            xy_error = ((measured_xy[0] - grid_x) ** 2 + (measured_xy[1] - grid_y) ** 2) ** 0.5
            if xy_error > POSITION_TOLERANCE_M:
                print(f"[EVAL] WARNING: episode {episode_idx} requested ({grid_x:.4f},{grid_y:.4f}) "
                      f"but measured ({measured_xy[0]:.4f},{measured_xy[1]:.4f}), error={xy_error:.5f} m")

            obs = env.get_observations()

            consecutive_success = 0
            max_consecutive_success = 0
            success = False
            success_complete_step = None
            first_threshold_step = None
            max_cube_z = float("-inf")
            min_ee_dist = float("inf")
            near_close_steps, near_steps = 0, 0
            far_close_steps, far_steps = 0, 0
            reached_bonus = False
            reached_success_height = False

            for step in range(num_steps):
                with torch.no_grad():
                    cube_pos = obj.data.root_pos_w.torch[0, :3]
                    ee_pos = ee_frame.data.target_pos_w.torch[0, 0, :3]
                    dist = torch.linalg.norm(cube_pos - ee_pos).item()
                    action = policy(obs)  # deterministic mean action -- never sampled/random
                    gripper_raw = action[0, 7].item()
                    obs, rew, dones, extras = env.step(action)

                cube_z = obj.data.root_pos_w.torch[0, 2].item()
                max_cube_z = max(max_cube_z, cube_z)
                min_ee_dist = min(min_ee_dist, dist)

                is_near = dist <= NEAR_FAR_REPORTING_BUCKET_M
                is_close = gripper_raw < 0
                if is_near:
                    near_steps += 1
                    near_close_steps += int(is_close)
                else:
                    far_steps += 1
                    far_close_steps += int(is_close)

                if cube_z > MINIMAL_LIFT_HEIGHT:
                    reached_bonus = True
                if cube_z >= SUCCESS_LIFT_HEIGHT:
                    if first_threshold_step is None:
                        first_threshold_step = step
                    reached_success_height = True
                    consecutive_success += 1
                    max_consecutive_success = max(max_consecutive_success, consecutive_success)
                    if consecutive_success >= SUCCESS_HOLD_STEPS and not success:
                        success = True
                        success_complete_step = step
                else:
                    consecutive_success = 0

                if bool(dones[0].item()):
                    break

            final_cube_z = obj.data.root_pos_w.torch[0, 2].item()
            drop_after_lift = reached_bonus and (final_cube_z < REST_Z + 0.02)

            time_to_success_s = success_complete_step * step_dt if success_complete_step is not None else None
            near_close_fraction = (near_close_steps / near_steps) if near_steps > 0 else None
            far_close_fraction = (far_close_steps / far_steps) if far_steps > 0 else None

            row = {
                "episode": episode_idx,
                "grid_i": gi,
                "grid_j": gj,
                "cube_x": grid_x,
                "cube_y": grid_y,
                "measured_x": measured_xy[0],
                "measured_y": measured_xy[1],
                "xy_error_m": xy_error,
                "success": success,
                "time_to_success_s": time_to_success_s,
                "max_cube_z": max_cube_z,
                "max_lift_above_rest": max_cube_z - REST_Z,
                "max_consecutive_success_steps": max_consecutive_success,
                "min_ee_cube_distance": min_ee_dist,
                "near_close_fraction": near_close_fraction,
                "far_close_fraction": far_close_fraction,
                "reached_bonus_0061": reached_bonus,
                "reached_success_0150": reached_success_height,
                "drop_after_lift": drop_after_lift,
            }
            results.append(row)
            print(f"[EVAL] episode {episode_idx:2d} xy=({grid_x:+.3f},{grid_y:+.3f}) "
                  f"success={success} t_success={time_to_success_s} max_z={max_cube_z:.4f} "
                  f"min_ee_dist={min_ee_dist:.4f} drop={drop_after_lift}")

        # ---- write outputs (must happen before the simulation app context exits -- exiting
        # `launch_simulation`'s `with` block terminates the process, so anything after it never
        # runs) ----
        out_dir = Path(args_cli.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "held_out_results.csv"
        columns = [
            "episode", "grid_i", "grid_j", "cube_x", "cube_y", "success", "time_to_success_s",
            "max_cube_z", "max_lift_above_rest", "max_consecutive_success_steps", "min_ee_cube_distance",
            "near_close_fraction", "far_close_fraction", "drop_after_lift",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"[EVAL] wrote {csv_path}")

        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        success_times = [r["time_to_success_s"] for r in successes]

        summary = {
            "checkpoint": args_cli.checkpoint,
            "task": TASK_ID,
            "seed": 0,
            "number_of_positions": len(results),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(results) if results else None,
            "mean_time_to_success_successes_only": statistics.mean(success_times) if success_times else None,
            "median_time_to_success_successes_only": statistics.median(success_times) if success_times else None,
            "min_time_to_success": min(success_times) if success_times else None,
            "max_time_to_success": max(success_times) if success_times else None,
            "mean_max_cube_z": statistics.mean(r["max_cube_z"] for r in results),
            "drop_count": sum(1 for r in results if r["drop_after_lift"]),
        }
        json_path = out_dir / "summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[EVAL] wrote {json_path}")
        print(f"[EVAL] SUMMARY: {json.dumps(summary, indent=2)}")

        env.close()


if __name__ == "__main__":
    main()
