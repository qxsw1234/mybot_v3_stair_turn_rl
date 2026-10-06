"""Phase-5: broad dynamics randomization fine-tuning for Mybot V3.

Continues from the Phase-4 checkpoint with expanded domain randomization
covering the MuJoCo sim2sim dynamics gap: passive joint damping, armature,
frictionloss, wider Kd factor, and wider motor strength.  The Phase-4
training already matched Kd_factor to MuJoCo damping (4-5x); this phase
widens everything so the policy can handle the cross-engine gap on 10-12cm
stairs.
"""


def train_mybot_v3_sim2sim_phase5(
    headless=True,
    num_envs=4096,
    iterations=2000,
    seed=42,
    sim_device="cuda:0",
    preview=False,
    resume_run=None,
    checkpoint=-1,
):

    import isaacgym
    assert isaacgym
    import random
    import numpy as np
    import torch

    from robodog_gym.envs.base.legged_robot_config import Cfg
    from robodog_gym.envs.robodog.mybot_v3_config import config_mybot_v3
    from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv

    from ml_logger import logger

    from robodog_gym_learn.ppo_cse import Runner
    from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    config_mybot_v3(Cfg)

    Cfg.env.num_envs = num_envs
    Cfg.env.num_observation_history = 10
    Cfg.env.dof_history_length = 2
    Cfg.env.dof_history_step_skip = 0
    Cfg.env.action_history_length = 2
    Cfg.env.action_history_step_skip = 0
    Cfg.cfg_ppo.seed = seed
    Cfg.cfg_ppo.runner.resume = resume_run is not None
    Cfg.cfg_ppo.runner.resume_path = resume_run
    Cfg.cfg_ppo.runner.checkpoint = checkpoint
    Cfg.cfg_ppo.runner.resume_curriculum = False
    Cfg.cfg_ppo.runner.save_interval = 50
    Cfg.cfg_ppo.runner.save_video_interval = 0
    Cfg.cfg_ppo.runner.save_curriculum_plot_interval = 50
    Cfg.cfg_ppo.runner.log_freq = 100
    Cfg.cfg_ppo.runner.wandb_logging = False
    Cfg.env.record_video = False
    Cfg.env.export_step_telemetry = False
    Cfg.env.compute_true_feet_height = True

    Cfg.asset.self_collisions = 1

    # control (same as phase4)
    Cfg.control.control_type = "P"
    Cfg.control.stiffness = {'joint': 25.}
    Cfg.control.damping = {'joint': 0.5}
    Cfg.control.action_scale = 0.25
    Cfg.control.hip_scale_reduction = 1.0

    # domain randomization — EXPANDED for sim2sim robustness
    Cfg.domain_rand.rand_interval_s = 4

    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.randomize_lag_timesteps = False

    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.5, 1.5]

    Cfg.domain_rand.randomize_restitution = True
    Cfg.domain_rand.restitution_range = [0.0, 0.2]

    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.10, -0.90]

    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.03, 0.03]

    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.85, 1.15]

    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.03, 0.03]

    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.Kp_factor_range = [0.85, 1.15]

    Cfg.domain_rand.randomize_Kd_factor = True
    # Phase-4 used [4.0, 5.0] to match MuJoCo passive damping.  Widen to
    # cover both low-damping (matched Isaac) and high-damping (MuJoCo) regimes.
    Cfg.domain_rand.Kd_factor_range = [0.8, 5.0]

    Cfg.domain_rand.randomize_rigids_after_start = True
    Cfg.domain_rand.randomize_friction_indep = False

    # Joint dynamics randomization (NEW in phase5)
    Cfg.domain_rand.randomize_joint_damping = True
    Cfg.domain_rand.joint_damping_range = [0.0, 2.5]
    Cfg.domain_rand.randomize_joint_armature = True
    Cfg.domain_rand.joint_armature_range = [0.01, 0.06]
    Cfg.domain_rand.randomize_joint_friction = True
    Cfg.domain_rand.joint_friction_loss_range = [0.0, 0.3]

    Cfg.domain_rand.randomize_gravity = True
    Cfg.domain_rand.gravity_range = [-0.3, 0.3]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99

    Cfg.domain_rand.push_robots = True
    Cfg.domain_rand.max_push_vel_xy = 0.5
    Cfg.domain_rand.push_interval_s = 15.0

    # Privileged observations
    Cfg.normalization.friction_range = [-0.5, 6.0]
    Cfg.normalization.restitution_range = [0, 0.4]
    Cfg.normalization.x_velocity_range = [-2, 2]
    Cfg.normalization.y_velocity_range = [-1, 1]
    Cfg.normalization.z_velocity_range = [-1, 1]
    Cfg.normalization.foot_height_range = [0.02, 0.08]
    Cfg.normalization.height_measurements_z_bias_range = [-0.05, 0.05]

    num_dof = Cfg.env.num_dof
    Cfg.env.privileged_observation_components = [
        ['friction', 1, True],
        ['ground_friction', None, False],
        ['restitution', 1, True],
        ['base_mass', 1, False],
        ['com_displacement ', 3, False],
        ['motor_strength', num_dof, False],
        ['motor_offset', num_dof, False],
        ['body_height', 1, False],
        ['body_velocity', 3, True],
        ['body_angular_velocity', 3, False],
        ['gravity', 3, False],
        ['clock_inputs', 4, False],
        ['desired_contact_states', 4, False],
        ['contact_states', 4, True],
        ['feet_height', 4, False],
        ['height_measurements_bias', 3, False],
        ['height_measurements_z_bias', 1, True],
        ['zero', 1, False],
    ]

    # Height observations
    Cfg.terrain.measure_heights = True
    Cfg.terrain.measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    Cfg.terrain.measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
    Cfg.terrain.height_measurements_per_step_xy_noise_std = 0.02
    Cfg.terrain.height_measurements_per_step_z_noise_std = 0.01
    Cfg.terrain.height_measurements_per_env_xy_noise_std = 0.03
    Cfg.terrain.height_measurements_per_env_z_noise_std = 0.02
    Cfg.terrain.height_measurements_per_env_noise_prob = 0.50
    Cfg.terrain.height_measurements_per_env_resampling_s = 7

    # Terrain — same as phase4 (MuJoCo-matched stairs)
    Cfg.terrain.curriculum = True
    Cfg.terrain.legacy_curriculum = False
    Cfg.terrain.border_size = 25
    Cfg.terrain.sim2sim_straight_stairs = True
    Cfg.terrain.sim2sim_stair_start_offset = 0.55
    Cfg.terrain.sim2sim_stair_tread_depth = 0.30
    Cfg.terrain.sim2sim_stair_width = 2.0
    Cfg.terrain.sim2sim_stair_num_steps = 8
    Cfg.terrain.sim2sim_stair_top_length = 0.90
    Cfg.terrain.sim2sim_stair_step_heights = [
        0.06, 0.07, 0.08, 0.085, 0.09, 0.095,
        0.10, 0.105, 0.11, 0.115, 0.12, 0.125,
    ]
    Cfg.terrain.min_init_terrain_level = 2 if resume_run else 0
    Cfg.terrain.max_init_terrain_level = 6
    Cfg.terrain.curriculum_min_terrain_level = 2 if resume_run else 0
    Cfg.terrain.curriculum_max_terrain_level = 6
    Cfg.terrain.curriculum_resample_within_band = True
    Cfg.terrain.curriculum_min_progress = 1.0
    Cfg.terrain.curriculum_stair_min_progress = 2.80
    Cfg.terrain.curriculum_stair_min_elevation = 0.06
    Cfg.terrain.curriculum_stair_min_elevation_steps = 6.0
    Cfg.terrain.curriculum_stair_max_lateral_displacement = 0.75
    Cfg.terrain.curriculum_stair_fast_command_threshold = 0.25
    Cfg.terrain.curriculum_stair_levelup_lin_vel_threshold = 0.18
    Cfg.terrain.curriculum_stair_leveldown_lin_vel_threshold = 0.30
    Cfg.terrain.curriculum_stair_levelup_ang_vel_threshold = 0.35
    Cfg.terrain.curriculum_stair_leveldown_ang_vel_threshold = 0.70
    Cfg.terrain.terrain_length = 8.
    Cfg.terrain.terrain_width = 8.
    Cfg.terrain.num_rows = 12
    Cfg.terrain.num_cols = 50
    Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.mesh_type = 'trimesh'
    Cfg.terrain.yaw_init_range = 0.03
    Cfg.terrain.difficulty_scale = 1.0
    Cfg.terrain.max_platform_height = 0.23
    Cfg.terrain.max_step_height = 0.23
    Cfg.terrain.terrain_smoothness = 0.005
    Cfg.terrain.x_init_range = 0.02
    Cfg.terrain.y_init_range = 0.02
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False

    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0

    Cfg.asset.armature = 0.03

    # Observation scaling
    Cfg.obs_scales.height_measurements = 5.0
    Cfg.obs_bias.height_measurements = 0.3

    # Episode length
    Cfg.env.episode_length_s = 10

    # Commands
    Cfg.commands.num_lin_vel_bins = 1
    Cfg.commands.num_ang_vel_bins = 1
    Cfg.commands.lin_vel_x = [-0.3, 0.6]
    Cfg.commands.lin_vel_y = [-0.2, 0.2]
    Cfg.commands.ang_vel_yaw = [-0.4, 0.4]

    # Rewards
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.total_reward_function = "leaky_relu_0.25"
    Cfg.rewards.soft_dof_pos_limit = 1.0
    Cfg.rewards.base_height_target = 0.3
    Cfg.rewards.footswing_height = 0.09
    Cfg.rewards.feet_clearance_ji22_target = 0.16
    Cfg.rewards.feet_clearance_ji22 = 0.8
    Cfg.rewards.soft_torque_limit = 0.7

    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)
    runner = Runner(env, Cfg, device=sim_device)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)

    return runner


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    args = parser.parse_args()

    train_mybot_v3_sim2sim_phase5(
        headless=args.headless,
        num_envs=args.num_envs,
        iterations=args.iterations,
        seed=args.seed,
        resume_run=args.resume_run,
        checkpoint=args.checkpoint,
    )
