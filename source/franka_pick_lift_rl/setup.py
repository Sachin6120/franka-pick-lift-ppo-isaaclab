# Copyright (c) 2026 Sachin Kumar Pal

"""Installation script for the 'franka_pick_lift_rl' python package."""

import os

import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Isaac Lab (isaaclab, isaaclab_assets, isaaclab_rl, isaaclab_tasks) is assumed already installed
# in the active environment -- intentionally not listed here to avoid pip re-resolving/altering it.
INSTALL_REQUIRES = []

setup(
    name="franka_pick_lift_rl",
    packages=["franka_pick_lift_rl"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
