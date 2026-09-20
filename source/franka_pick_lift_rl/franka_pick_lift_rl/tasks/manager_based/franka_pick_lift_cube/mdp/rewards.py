# Copyright (c) 2026 Sachin Kumar Pal

"""Task-specific reward/success terms with no upstream equivalent.

The official ``Isaac-Lift-Cube-Franka-v0`` task shapes lift progress mainly through
``object_goal_distance`` (distance to a randomized, commanded goal pose). This Tier-1 task drops
goal-pose commanding entirely (no domain randomization beyond initial cube geometry, per design),
so it needs its own dense "how close to the target lift height" signal and its own definition of
"success" (sustained lift, not merely reaching a goal pose). Both are defined here.

This module previously also defined ``premature_gripper_close`` and
``near_gripper_open_continuous``, two custom gripper-timing shaping terms intended to fix a
never-closes-the-gripper failure mode. Both were removed (2026-09-20): a causal ablation showed
``premature_gripper_close`` was itself driving an early, self-reinforcing bias toward a fixed
gripper action before the policy had enough near-cube experience to learn otherwise, and a control
run of the unmodified official ``Isaac-Lift-Cube-Franka-v0`` task -- which has no gripper-timing
reward terms at all -- learned strong, state-conditioned gripper behavior (and successful
pick-and-lift) using only its reach/lift/goal-tracking rewards at its intended training scale. The
production task now relies on the same mechanism: gripper timing emerges instrumentally from the
manipulation objective, with no dedicated gripper-timing reward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def lift_height_progress(
    env: ManagerBasedRLEnv,
    rest_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense shaping reward for how close the object is to the target lift height, normalized
    against its own settled resting height rather than absolute world-z.

    Returns a value in ``[0, 1]``: ``0`` at ``rest_height`` (the object's measured resting z with
    the robot doing nothing -- NOT its raw spawn z, which is higher and settles down under gravity
    within a few simulation steps; see diagnostic trace, 2026-09-19), rising linearly to ``1`` at
    ``target_height``. ``rest_height`` must be measured empirically for the object/scene in use --
    it is not derivable from the spawn config alone (contact/penetration settling changes it).

    Earlier version of this function normalized against absolute world-z with an implicit
    rest-height assumption of 0, which handed out ``rest_height / target_height`` (here: 0.14) of
    reward for doing nothing -- fixed by this rest-relative formulation.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height = object.data.root_pos_w.torch[:, 2]
    return torch.clamp((height - rest_height) / (target_height - rest_height), min=0.0, max=1.0)


class lift_success(ManagerTermBase):
    """Tracks the task's success criterion: object height above ``lift_height`` sustained for
    ``hold_steps`` consecutive control steps (a stability window), not merely "episode did not
    terminate". Logs ``Metrics/success_rate`` (fraction of finished episodes that ever reached
    sustained success) and ``Metrics/time_to_success_s`` on reset, matching the evaluation
    protocol's success definition.

    Returns a one-shot ``1.0`` on the step success is first reached, ``0.0`` otherwise, so it can
    be used as a (typically small or zero) reward term as well as a metrics logger.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._consecutive_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._success_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self._step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        log = self._env.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._succeeded[env_ids].float().mean().item()
        succeeded_ids = env_ids[self._succeeded[env_ids]]
        if len(succeeded_ids) > 0:
            log["Metrics/time_to_success_s"] = (
                self._success_step[succeeded_ids].float() * self._env.step_dt
            ).mean().item()
        self._consecutive_steps[env_ids] = 0
        self._succeeded[env_ids] = False
        self._success_step[env_ids] = -1
        self._step_count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_height: float,
        hold_steps: int,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        object: RigidObject = env.scene[object_cfg.name]
        above = object.data.root_pos_w.torch[:, 2] > lift_height
        self._consecutive_steps = torch.where(
            above, self._consecutive_steps + 1, torch.zeros_like(self._consecutive_steps)
        )
        self._step_count += 1
        newly_succeeded = (self._consecutive_steps >= hold_steps) & (~self._succeeded)
        self._success_step = torch.where(newly_succeeded, self._step_count, self._success_step)
        self._succeeded |= newly_succeeded
        return newly_succeeded.float()
