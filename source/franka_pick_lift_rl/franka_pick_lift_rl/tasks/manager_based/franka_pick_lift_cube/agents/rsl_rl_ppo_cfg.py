# Copyright (c) 2026 Sachin Kumar Pal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaPickLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO hyperparameters for the Tier-1 Franka pick-and-lift cube task.

    Copied verbatim from the validated official ``Isaac-Lift-Cube-Franka-v0`` task
    (``isaaclab_tasks`` lift/config/franka/agents/rsl_rl_ppo_cfg.py). Not tuned for this task's
    reduced observation set -- this is the untouched Tier-1 baseline starting point, per design.

    ``std_type="scalar"`` (the framework default) is used deliberately, matching both the
    canonical solved checkpoint (``model_200.pt`` of run ``2026-09-20_10-14-12``) and the official
    baseline's own configuration. A rare, non-deterministic training crash
    (``RuntimeError: normal expects all elements of std >= 0.0``) was observed in some extended
    post-convergence training attempts; the alternative ``std_type="log"`` parameterization was
    tried as a candidate fix but the same crash recurred under it too, so it was not adopted.
    Periodic checkpointing (``save_interval`` below) is the practical mitigation: training success
    is already reached well before the crash tends to occur, and the canonical checkpoint above was
    produced this way.
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_pick_lift_cube"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
