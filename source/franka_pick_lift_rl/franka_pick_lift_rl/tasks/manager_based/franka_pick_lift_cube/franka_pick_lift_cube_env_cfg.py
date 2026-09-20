# Copyright (c) 2026 Sachin Kumar Pal

"""Environment config for ``Franka-PickLift-Cube-v0`` (Tier-1).

Task: a Franka Panda must reach, grasp and lift a cube from a randomized initial XY position on
the table, and hold it above a defined lift height for a short stability window.

Reuse boundary (see project design notes / README):

- Scene composition (table, ground, light, MISSING robot/object/ee_frame slots) is reused
  unmodified from :class:`isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg.ObjectTableSceneCfg`.
- ``object_ee_distance``, ``object_is_lifted``, ``object_position_in_robot_root_frame`` are reused
  from the official lift task's ``mdp`` module (see ``mdp/__init__.pyi``).
- All other MDP terms below (actions, generic observations, generic event/termination/reward
  terms) are the framework's own generic ``isaaclab.envs.mdp`` terms.
- ``lift_height_progress`` and ``lift_success`` (mdp/rewards.py) are new: this task drops the
  official task's goal-pose command/tracking entirely (Tier-1 has no domain randomization beyond
  initial cube geometry), so it needs its own dense shaping signal and its own sustained-lift
  success definition instead of goal-distance tracking.

Upstream ``isaaclab_assets.robots.franka.FRANKA_PANDA_CFG`` is not modified. This project only
overrides the ``usd_path`` on its own copy (see ``FRANKA_PANDA_LEGACY_USD_PATH`` below), because
the asset path resolved via ``ISAAC_NUCLEUS_DIR`` for this Isaac Lab revision (v3.0.0-beta2.patch1)
currently 404s against the live content server; the file was moved server-side to a ``Legacy/``
sub-path (verified by direct HTTP check, 2026-09-19).
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics import PhysxCfg

from . import mdp

##
# Pre-defined configs (reused, not duplicated)
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import ObjectTableSceneCfg  # isort: skip

# See module docstring: works around a remote content-server path mismatch for this Isaac Lab
# revision. Does not modify isaaclab_assets.FRANKA_PANDA_CFG -- only this project's own copy.
FRANKA_PANDA_LEGACY_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
    "IsaacLab/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
)

# Cube lift-height calibration (world-frame z). REST_Z is the cube's *measured* settled resting
# height -- NOT its spawn z (0.055) -- confirmed empirically with the robot doing nothing: it falls
# under gravity from 0.055 to a stable 0.021 within ~4 simulation steps (diagnostic trace,
# 2026-09-19). Earlier revision used absolute world-z with an implicit rest-height assumption of 0
# for lift_progress and MINIMAL_LIFT_HEIGHT=0.04 for lifting_object; both were below the true
# spawn/rest values (0.055 / 0.021) and so handed out reward for doing nothing. Fixed here:
#   - lift_progress is now normalized relative to REST_Z (see lift_height_progress in mdp/rewards.py).
#   - MINIMAL_LIFT_HEIGHT is now REST_Z + a genuine ~4cm lift delta, comfortably above both REST_Z
#     and the 0.055 spawn z, so it can no longer fire without the object actually being lifted.
# SUCCESS_LIFT_HEIGHT (the task's actual success criterion, used by lift_success) is unchanged.
REST_Z = 0.021
LIFT_DELTA_FOR_BONUS = 0.04
MINIMAL_LIFT_HEIGHT = REST_Z + LIFT_DELTA_FOR_BONUS  # ~0.061
SUCCESS_LIFT_HEIGHT = 0.15
SUCCESS_HOLD_STEPS = 25  # at decimation=2, sim.dt=0.01 -> control dt=0.02s -> 0.5s hold window

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Reused verbatim from the official Franka lift task's action wiring (generic framework action
    terms, not lift-specific): 7 arm joint-position actions + 1 gripper open/close action.

    ``gripper_action`` uses the framework's standard
    :class:`~isaaclab.envs.mdp.actions.binary_joint_actions.BinaryJointPositionActionCfg`
    (``raw < 0`` -> CLOSE, ``raw >= 0`` -> OPEN). A control-baseline comparison against the
    unmodified official task -- which uses this same binary action term and successfully learns
    pick-and-lift -- confirmed the binary representation is not a bottleneck for this task, so it
    is used here deliberately rather than a custom actuator mapping.
    """

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    Compact manipulation-relevant policy observation (36 dims in the official task, 32 here):
    joint_pos_rel(9) + joint_vel_rel(9) + object_position_in_robot_root_frame(3) +
    object_position_in_ee_frame(3) + last_action(8) = 32 dims. The official task's
    ``target_object_position`` (goal-pose command, 7 dims) is dropped because this Tier-1 task has
    no commanded goal pose to observe.

    ``object_position_in_ee_frame`` (2026-09-20 A/B experiment): policy-distribution diagnostics
    after the continuous near-open gripper-shaping experiment found near/far gripper-action
    distributions still statistically indistinguishable despite a non-vanishing reward gradient --
    the base/root-relative cube position alone apparently does not give the policy a directly
    usable "is the cube within grasp range of the gripper" feature. This term adds that local,
    end-effector-relative encoding (same ``ee_frame`` target used by ``reaching_object``) without
    removing ``object_position_in_robot_root_frame``, isolating the effect to observation
    representation only (see ``mdp/observations.py``).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_position_ee = ObsTerm(func=mdp.object_position_in_ee_frame)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset distribution.

    Reuses the official lift task's validated reset region verbatim (not invented): object spawns
    at [0.5, 0, 0.055] +/- pose_range, giving an absolute reachable range of x in [0.4, 0.6] m,
    y in [-0.25, 0.25] m on the table -- confirmed reachable by the same robot+table geometry in
    the validated ``Isaac-Lift-Cube-Franka-v0`` smoke test.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms. Each term's purpose:

    - ``reaching_object``: dense tanh-kernel distance-to-cube shaping (reused from the official
      task, but with ``std`` widened from 0.1 to 0.3 -- see below) -- gets the end effector near
      the cube early in training.
    - ``lift_progress``: new dense shaping toward ``SUCCESS_LIFT_HEIGHT``, rest-relative -- replaces
      the official task's goal-distance shaping, which this task doesn't have (no commanded goal
      pose).
    - ``lifting_object``: reused official low-threshold binary bonus for clearing the table, with
      ``minimal_height`` recalibrated (see ``MINIMAL_LIFT_HEIGHT`` above).
    - ``lift_success``: new one-shot bonus + metrics logger for the task's actual success
      criterion (sustained lift, see module docstring); weight kept small so it doesn't dominate
      the dense shaping terms.
    - ``action_rate`` / ``joint_vel``: small regularization penalties (reused, generic) to
      discourage jittery/unsafe motion -- same purpose and same small weights as the official task.

    ``reaching_object`` std: diagnostics (2026-09-19) showed the official task's std=0.1 kernel is
    nearly flat (raw <0.05) across the ~0.2-0.5 m distances actually encountered from typical
    resets under this task's (goal-pose-command-free) reward design, supplying almost no gradient
    to guide the policy toward the cube. Widened to 0.3 for a useful long-range gradient; weight
    left unchanged at 1.0 and no second reach term was added (see project diagnostics).

    This task previously also carried two custom gripper-timing reward terms,
    ``premature_gripper_close`` and ``near_gripper_open_continuous`` (removed 2026-09-20). A
    control-baseline comparison against the unmodified official ``Isaac-Lift-Cube-Franka-v0`` task
    -- which carries no gripper-timing reward terms at all and successfully learns strong,
    state-conditioned gripper behavior purely from its reach/lift/goal-tracking rewards at its
    intended training scale -- found these two terms were themselves an obstacle rather than a fix:
    the larger-weighted term drove an early, self-reinforcing bias toward a fixed gripper action
    before the policy had enough near-cube experience to learn state-conditioned behavior. See
    ``mdp/rewards.py``'s module docstring for further detail.
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.3}, weight=1.0)

    lift_progress = RewTerm(
        func=mdp.lift_height_progress,
        params={"rest_height": REST_Z, "target_height": SUCCESS_LIFT_HEIGHT},
        weight=5.0,
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": MINIMAL_LIFT_HEIGHT}, weight=15.0
    )

    lift_success = RewTerm(
        func=mdp.lift_success,
        params={"lift_height": SUCCESS_LIFT_HEIGHT, "hold_steps": SUCCESS_HOLD_STEPS},
        weight=10.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Episode termination (separate from the ``lift_success`` success criterion above, which is a
    metric/reward term, not a termination -- the episode is allowed to continue after success so
    the policy also demonstrates holding the cube, matching the official task's style of not
    terminating early on goal-reach).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


##
# Environment configuration
##


@configclass
class FrankaPickLiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Tier-1 Franka pick-and-lift cube task."""

    # Scene: reused unmodified (table/ground/light + MISSING robot/object/ee_frame slots)
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # general settings (unchanged from the validated official task)
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )

        # Robot: FRANKA_PANDA_CFG.replace() returns a new instance without mutating the shared
        # upstream singleton; spawn.replace() likewise returns a new UsdFileCfg, so the override
        # below never touches isaaclab_assets.FRANKA_PANDA_CFG or its nested spawn config.
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FRANKA_PANDA_CFG.spawn.replace(usd_path=FRANKA_PANDA_LEGACY_USD_PATH),
        )

        # Cube: reused unmodified from the official task (same asset, same physics tuning; only
        # the Franka asset path needed a project-local override).
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[0, 0, 0, 1]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # End-effector frame: required by the reused `object_ee_distance` reward term.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )
