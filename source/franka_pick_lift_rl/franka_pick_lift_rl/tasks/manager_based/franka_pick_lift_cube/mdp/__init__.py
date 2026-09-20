# Copyright (c) 2026 Sachin Kumar Pal

"""MDP terms for the Franka pick-and-lift cube task.

Framework-generic terms (joint_pos_rel, action_rate_l2, time_out, JointPositionActionCfg, ...)
and the reward/observation functions already validated by the official
``Isaac-Lift-Cube-Franka-v0`` task (object_ee_distance, object_is_lifted,
object_position_in_robot_root_frame) are reused directly rather than duplicated -- see
``__init__.pyi`` for exactly what is re-exported from where. Only ``lift_height_progress`` and
``lift_success`` in ``rewards.py`` are new to this project.
"""

from isaaclab.utils.module import lazy_export

lazy_export()
