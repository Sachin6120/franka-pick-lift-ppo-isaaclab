# Copyright (c) 2026 Sachin Kumar Pal

"""Task implementations for the franka_pick_lift_rl project."""

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
