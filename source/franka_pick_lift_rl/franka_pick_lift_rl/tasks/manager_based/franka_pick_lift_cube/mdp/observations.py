# Copyright (c) 2026 Sachin Kumar Pal

"""Task-specific observation terms with no upstream equivalent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_ee_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The cube's center position [m], expressed in the same end-effector frame used by
    :func:`isaaclab_tasks.manager_based.manipulation.lift.mdp.object_ee_distance` (the
    ``reaching_object`` reward) -- not a second, independently-defined EE frame.

The existing :func:`object_position_in_robot_root_frame` observation gives the cube's position
    relative to the robot base, not relative to the gripper -- so the policy has no observation
    feature that directly and locally encodes "is the cube within the grasp region," only
    base-frame position plus joint angles from which that relationship would have to be inferred
    indirectly. This term adds that missing local, task-relative encoding without removing the
    existing base-frame one.

    Uses the live ``ee_frame`` :class:`~isaaclab.sensors.FrameTransformer` (the "end_effector"
    target already used by ``reaching_object``) and the live cube root position -- not a
    stale/cached USD-stage transform -- via
    :func:`isaaclab.utils.math.subtract_frame_transforms`.

    Returns:
        Cube center position in the end-effector frame, shape ``(num_envs, 3)``, ``[x, y, z]``.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w.torch[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w.torch[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w.torch[..., 0, :]

    object_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, object_pos_w)
    return object_pos_ee
