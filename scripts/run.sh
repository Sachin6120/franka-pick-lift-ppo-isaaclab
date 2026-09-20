#!/usr/bin/env bash
# Copyright (c) 2026 Sachin Kumar Pal
#
# Local runtime launcher. This project's task registers and runs fine from
# /home/sachin/env_isaaclab, but Isaac Sim's own pip package (6.0.1.0) currently lives in a
# separate venv (/home/sachin/env_isaacsim) rather than inside env_isaaclab -- see project README
# "Runtime facts". This wrapper adds env_isaacsim's site-packages to PYTHONPATH for a single
# invocation only; it does not modify either virtual environment or any shell startup file.
#
# Usage:
#   scripts/run.sh scripts/random_agent.py --task Franka-PickLift-Cube-v0 --num_envs 8 --viz none
#   scripts/run.sh scripts/rsl_rl/train.py --task Franka-PickLift-Cube-v0 --num_envs 8 --headless --max_iterations 8

set -euo pipefail

ISAACLAB_ROOT="/home/sachin/IsaacLab"
ISAACSIM_SITE_PACKAGES="/home/sachin/env_isaacsim/lib/python3.12/site-packages"

exec env PYTHONPATH="${ISAACSIM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ISAACLAB_ROOT}/isaaclab.sh" -p "$@"
