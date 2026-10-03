"""Phase-4 MuJoCo-matched ascending-stair fine tuning for Mybot V3.

Unlike Isaac Gym's centred pyramid stairs, these environments start on flat
ground 0.55 m before the first riser and reproduce the MuJoCo evaluator's
0.30 m tread, 2.0 m width, eight steps, and 0.90 m top platform.  The row
curriculum spans 8--12.5 cm so an existing 8 cm policy can progressively learn
the actual ascending task used for Sim2Sim acceptance.
"""


def train_mybot_v3_sim2sim_phase4(
    headless=True,
    num_envs=4096,
    iterations=1000,
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
    Cfg.cfg_ppo.seed = seed
    Cfg.cfg_ppo.runner.resume = resume_run is not None
    Cfg.cfg_ppo.runner.resume_path = resume_run
    Cfg.cfg_ppo.runner.checkpoint = checkpoint
    Cfg.cfg_ppo.runner.resume_curriculum = False
    Cfg.cfg_ppo.runner.save_interval = 50
    Cfg.cfg_ppo.runner.save_video_interval = 0
    Cfg.cfg_ppo.runner.save_curriculum_plot_interval = 50
    Cfg.cfg_ppo.runner.wandb_logging = False
    Cfg.env.record_video = False
    Cfg.env.export_step_telemetry = False
    Cfg.env.compute_true_feet_height = True
    # # Cfg.env.num_recording_envs = 1
    # # Cfg.terrain.num_cols = 3
    # # Cfg.terrain.num_rows = 3
    # # Cfg.terrain.center_span = 1
    debug_viz = False
    # Cfg.cfg_ppo.runner.save_video_interval = 250
    # Cfg.env.episode_length_s = 5

    # Cfg.cfg_ppo.runner.wandb_note = "first test"

    # Cfg.cfg_ppo.runner.log_freq = 100

    # curriculum configuration
    # Cfg.commands.num_lin_vel_bins = 1 no used 
    # Cfg.commands.num_ang_vel_bins = 1
    # Cfg.curriculum_thresholds.tracking_ang_vel = 0.95
    # Cfg.curriculum_thresholds.tracking_lin_vel = 0.95
    # Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    # Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    # asset setup
    Cfg.asset.self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter. This has also effect for terminal collisions!

    #-----------------------
    # control
    #-----------------------

    Cfg.control.control_type = "P" # actuator control type
    Cfg.control.stiffness = {'joint': 25.}  # [N*m/rad]
    Cfg.control.damping = {'joint': 0.5}  # [N*m*s/rad]
    Cfg.control.action_scale = 0.25 #0.25
    Cfg.control.hip_scale_reduction = 1.0 # consistent with rsl

    #-----------------------
    # domain randomization
    #-----------------------

    Cfg.domain_rand.rand_interval_s = 4

    # Nominal domain rand
    # Learn the corrected geometry with moderate randomization first.  A
    # later robustness pass can widen these ranges after 10/12 cm succeeds.
    # MuJoCo evaluator applies the action immediately at each 20 ms policy
    # tick.  Remove the extra 0--2 tick delay while fitting its dynamics.
    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.randomize_lag_timesteps = False

    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.8, 1.2]
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.restitution_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_base_mass = True
    # Isaac collapses the 1.001 kg fixed payload/IMU into the base, whereas
    # the deployment MuJoCo XML omits those helper links.  Centre training on
    # the MuJoCo base mass by subtracting that fixed-link mass.
    Cfg.domain_rand.added_mass_range = [-1.10, -0.90]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.02, 0.02]
    Cfg.domain_rand.randomize_ground_friction = False 
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True  # wtw true
    Cfg.domain_rand.motor_strength_range = [0.95, 1.05]
    Cfg.domain_rand.randomize_motor_offset = True # wtw true
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.randomize_Kd_factor = True
    Cfg.domain_rand.randomize_rigids_after_start = True
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.Kp_factor_range = [0.95, 1.05]
    # MuJoCo has 1--2 Nms/rad passive joint damping in addition to the 0.5
    # controller Kd.  Isaac's URDF has zero passive damping, so an effective
    # Kd factor around 4--5 reproduces the target total damping.
    Cfg.domain_rand.Kd_factor_range = [4.0, 5.0]


    # gravity changes and pushes
    Cfg.domain_rand.randomize_gravity = True # wtw true
    Cfg.domain_rand.gravity_range = [-0.2, 0.2]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99

    # Cfg.domain_rand.lag_timesteps = 6
    # Cfg.domain_rand.randomize_lag_timesteps = False # wtw true

    # Cfg.domain_rand.randomize_friction = True
    # Cfg.domain_rand.friction_range = [0.0, 1.5]
    # Cfg.domain_rand.randomize_restitution = False  # wtw true
    # Cfg.domain_rand.restitution_range = [0.0, 0.4]
    # Cfg.domain_rand.randomize_base_mass = True
    # Cfg.domain_rand.added_mass_range = [-2.0, 2.0]
    # Cfg.domain_rand.randomize_com_displacement = False
    # Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    # Cfg.domain_rand.randomize_ground_friction = False 
    # Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    # Cfg.domain_rand.randomize_motor_strength = False  # wtw true
    # Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    # Cfg.domain_rand.randomize_motor_offset = False # wtw true
    # Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    # Cfg.domain_rand.randomize_Kp_factor = False
    # Cfg.domain_rand.randomize_Kd_factor = False
    # Cfg.domain_rand.randomize_rigids_after_start = True
    # Cfg.domain_rand.randomize_friction_indep = False

    # # gravity changes and pushes
    # Cfg.domain_rand.randomize_gravity = False # wtw true
    # Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    # Cfg.domain_rand.gravity_rand_interval_s = 8.0
    # Cfg.domain_rand.gravity_impulse_duration = 0.99

    # pushes as in rsl
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.max_push_vel_xy = 0.5
    Cfg.domain_rand.push_interval_s = 15.0



    #--------------------------
    # Priviledged observations
    #--------------------------

    # Normalization used for estimator
    Cfg.normalization.friction_range = [-0.5,6.0] ##from[0.1, 1.3]
    Cfg.normalization.restitution_range = [0, 0.4]
    Cfg.normalization.x_velocity_range = [-2, 2]
    Cfg.normalization.y_velocity_range = [-1, 1]
    Cfg.normalization.z_velocity_range = [-1, 1]
    Cfg.normalization.foot_height_range = [0.02, 0.08]
    Cfg.normalization.height_measurements_z_bias_range = [-0.05, 0.05]

    num_dof = Cfg.env.num_dof
    Cfg.env.privileged_observation_components = [
        ['friction',        1,      True],
        ['ground_friction', None,   False],
        ['restitution',     1,      True],
        ['base_mass',       1,      False],
        ['com_displacement ', 3,    False],
        ['motor_strength',  num_dof, False],
        ['motor_offset',    num_dof, False],
        ['body_height',     1,      False],
        ['body_velocity',   3,      True],
        ['body_angular_velocity', 3, False],
        ['gravity',         3,      False],
        ['clock_inputs',    4,      False],
        ['desired_contact_states', 4, False],
        ['contact_states',  4,      True],
        ['feet_height',     4,      False],
        ['height_measurements_bias', 3, False],
        ['height_measurements_z_bias', 1, True],
        ['zero',            1,      False]   # additional privileged observation that will always be zero, to use when disabling the estimator
    ]
    

    #--------------------------
    # Observations and commands
    #--------------------------

    num_dof = Cfg.env.num_dof
    Cfg.env.policy_observation_components = [
        ['commands',             None, True],  # size = cfg.commands.num_commands
        ['global_linear_vel',    3,     False], # linear velocity in global frame of reference
        ['linear_vel',           3,     False],
        ['angular_vel',          3,     False],
        ['projected_gravity',    3,     True],
        ['dof_positions',        num_dof, True], # joint positions
        ['dof_velocities',       num_dof, True],
        ['dof_position_history', num_dof * Cfg.env.dof_history_length, False], # Joint pos history n...n-m, size = 12 * dof_history_length. Assumes joint_positions are false
        ['dof_velocity_history', num_dof * Cfg.env.dof_history_length, False],
        ['last_actions',         num_dof, True], # n-1 actions
        ['action_history',       num_dof * Cfg.env.action_history_length, False], # n-1...n-m actions, size = 12 * action_history_length
        ['timing_parameter',     1,     False],
        ['clock_inputs',         4,     False],
        ['yaw',                  1,     False],
        ['contact_states',       4,     False],
        ['height_measurements',  None,  True], # size = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
    ]

    Cfg.env.estimator_observation_components = [
        ['commands',             None, False],  # size = cfg.commands.num_commands
        ['global_linear_vel',    3,     False], # linear velocity in global frame of reference
        ['linear_vel',           3,     False],
        ['angular_vel',          3,     False],
        ['projected_gravity',    3,     True],
        ['dof_positions',        num_dof, True], # joint positions
        ['dof_velocities',       num_dof, True],
        ['dof_position_history', num_dof * Cfg.env.dof_history_length, False], # Joint pos history n...n-m, size = 12 * dof_history_length. Assumes joint_positions are false
        ['dof_velocity_history', num_dof * Cfg.env.dof_history_length, False],
        ['last_actions',         num_dof, True], # n-1 actions
        ['action_history',       num_dof * Cfg.env.action_history_length, False], # n-1...n-m actions, size = 12 * action_history_length
        ['timing_parameter',     1,     False],
        ['clock_inputs',         4,     False],
        ['yaw',                  1,     False],
        ['contact_states',       4,     False],
        ['height_measurements',  None,  True], # size = len(cfg.terrain.measured_points_x)*len(cfg.terrain.measured_points_y)
    ]

    Cfg.terrain.measure_heights = True
    Cfg.terrain.measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    Cfg.terrain.measured_points_y = [-0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3]

    Cfg.noise_scales.height_measurements = 0 #0.25
    Cfg.terrain.height_measurements_per_step_xy_noise_std = 0.02
    Cfg.terrain.height_measurements_per_step_z_noise_std = 0.01

    Cfg.terrain.height_measurements_per_env_xy_noise_std = 0.03
    Cfg.terrain.height_measurements_per_env_z_noise_std = 0.02
    Cfg.terrain.height_measurements_per_env_noise_prob = 0.50
    Cfg.terrain.height_measurements_per_env_resampling_s = 7

    Cfg.env.num_observation_history = 10 #10 training way slower, 5 still trains ok #30 reduced because of height measurements
    # Cfg.env.sparse_obs_history = [0, 1, 3, 6, 10]
    # Cfg.env.sparse_obs_history = [0, 1, 2, 3, 4]

    # Cfg.env.num_estimator_obs_history = 10
    # Cfg.env.sparse_estimator_obs_history = [0,1,2]


    if Cfg.env.sparse_obs_history is not None:
      Cfg.cfg_ppo.runner.wandb_note = 'Extero, sparse hist ' + ','.join(str(e) for e in Cfg.env.sparse_obs_history)
    else:
      Cfg.cfg_ppo.runner.wandb_note = 'Extero, full hist ' + str(Cfg.env.num_observation_history)

    # commands
    Cfg.commands.num_commands = 3 # change!

    # action_scale=0.25, so this bounds commanded joint offsets to about
    # +/-0.75 rad and prevents rare Gaussian tails from creating torque spikes.
    Cfg.normalization.clip_actions = 3.0
    Cfg.normalization.clip_observations = 100.0

  

    #--------------------------
    # Terrain configuration
    #--------------------------



    Cfg.terrain.curriculum = True
    # Promote terrain difficulty from tracking quality. The legacy rule used
    # net displacement over an episode, which incorrectly demoted successful
    # robots whenever random heading commands made them turn back.
    Cfg.terrain.legacy_curriculum = False

    Cfg.terrain.border_size = 25
    # Rows are explicit physical riser heights.  Keep 8 cm in the mix to
    # preserve the known capability while exposing 10--12.5 cm every reset.
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
    # The old 2 m + one-step criterion could promote a policy that stalled at
    # the first riser.  Require traversal and roughly six risers of elevation.
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
    # Every environment uses the forward ascending staircase.  General terrain
    # robustness is retained in the warm-started policy and can be mixed back
    # after MuJoCo 10/12 cm success is established.
    Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Cfg.terrain.mesh_type = "trimesh"
  

    # wtw additions (here set them to be same as RSL)
    Cfg.terrain.yaw_init_range = 0.03
    Cfg.terrain.difficulty_scale = 1.0
                                      # difficulty: 0 for firstlevel, 1 for last level 
                                      # slope = difficulty * 0.4 (max 0.4)
                                      # step_height = 0.05 + 0.18 * difficulty (max 0.23)
                                      # discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05) (max 0.2)
                                      # rough slope terrain: step=self.cfg.terrain_smoothness
    Cfg.terrain.max_platform_height = 0.23
    Cfg.terrain.max_step_height = 0.23
    Cfg.terrain.terrain_smoothness = 0.005
    # max step_height=0.05 + 0.18*0.75=0.185
    # max discrete_obstacles_height=0.05 + 0.75*(0.25-0.05)=0.185
    Cfg.terrain.x_init_range = 0.02
    Cfg.terrain.y_init_range = 0.02
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False # center robots in the middle of the terrain grid, only makes sense without curriculum 

    
    # this will be averaged with domain_rand.friction_range if randomization is on
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0

    # MuJoCo uses 0.0544 reflected armature on calves and 0.01 elsewhere.
    # Isaac accepts one asset-wide value; 0.03 is the closest conservative
    # compromise and is substantially closer than the previous 0.01.
    Cfg.asset.armature = 0.03

    # A compact CPU physics map can run beside the main GPU training process
    # so the user can inspect the same task in the Isaac Gym viewer.
    if preview:
        Cfg.terrain.border_size = 5
        Cfg.terrain.num_rows = 6
        Cfg.terrain.num_cols = 20
    if sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False



    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, none, smooth flat, rough flat]
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0.5, 0, 0, 0.25, 0.25]
    # Cfg.terrain.curriculum = False
    # Cfg.terrain.max_platform_height = 0.1
    # Cfg.terrain.slope_treshold = 0.25 ##added (maybe needs to be reduced)
    # Cfg.terrain.terrain_noise_magnitude = 0.05
    # Cfg.terrain.border_size = 10.0
    # Cfg.terrain.num_cols = 30
    # Cfg.terrain.num_rows = 30
    # Cfg.terrain.terrain_width = 5.0
    # Cfg.terrain.terrain_length = 5.0

    # Cfg.terrain.center_span = 10 # 5
    # Cfg.terrain.horizontal_scale = 0.1 #0.1 resolution of gridmap


    # terrain configuration
    # Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    # Cfg.domain_rand.tile_height_curriculum = False
    # Cfg.domain_rand.tile_height_update_interval = 1000000
    # Cfg.domain_rand.tile_height_curriculum_step = 0.01
    
    # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, none, smooth flat, rough flat]
    # Cfg.terrain.terrain_proportions = [0, 0, 0, 0, 0.5, 0, 0, 0.25, 0.25]
    # Cfg.terrain.curriculum = False
    # Cfg.terrain.max_platform_height = 0.1
    # Cfg.terrain.slope_treshold = 0.25 ##added (maybe needs to be reduced)
    # Cfg.terrain.terrain_noise_magnitude = 0.05
    # Cfg.terrain.border_size = 10.0
    # Cfg.terrain.mesh_type = "trimesh"
    # # Cfg.terrain.num_cols = 30
    # # Cfg.terrain.num_rows = 30
    # Cfg.terrain.terrain_width = 5.0
    # Cfg.terrain.terrain_length = 5.0
    # Cfg.terrain.x_init_range = 0.2
    # Cfg.terrain.y_init_range = 0.2
    # Cfg.terrain.teleport_thresh = 0.3
    # Cfg.terrain.teleport_robots = False
    # Cfg.terrain.center_robots = True
    # # Cfg.terrain.center_span = 10 # 5
    # Cfg.terrain.horizontal_scale = 0.1 #0.1 resolution of gridmap
     

    # original RSL plane
    # Cfg.terrain.mesh_type = 'plane'
    # Cfg.terrain.teleport_robots = False # else gives error for plane
    # Cfg.terrain.measure_heights = False




    # -----------------
    # Env termination
    # -----------------
    
    # terminate on these conditions
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_body_impact = True
    Cfg.rewards.terminal_body_height = 0.10
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 1.13446  # 65 degrees
    # Reward scales are multiplied by dt=0.02 during environment setup. The
    # special termination reward is applied once, so -50 becomes -1 per fall.
    Cfg.reward_scales.termination = -50.0
    

    # ---------------------
    # Rewards
    # ---------------------

    Cfg.rewards.reward_container_name = "CoRLRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    # Cfg.rewards.sigma_rew_neg = 0.2
    Cfg.rewards.total_reward_function = "leaky_relu_0.25"

    Cfg.rewards.reward_curriculum_factor_init = 0.001
    Cfg.rewards.reward_curriculum_factor_rate = 0.99


    #positive rewards
    Cfg.reward_scales.tracking_lin_vel = 2.0
    Cfg.rewards.tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
    Cfg.rewards.end_sigma_curriculum_iter = 0 # disable curriculum
    Cfg.reward_scales.tracking_ang_vel = 1.0
    Cfg.rewards.tracking_sigma_yaw = 0.25
    Cfg.rewards.end_sigma_yaw_curriculum_iter = 0


    # air time

    Cfg.reward_scales.feet_air_time_rsl = 1.0 #exteroceptive 3.0 # was 2 # 0.75 #0.75 #2.0
    Cfg.rewards.feet_air_time_rsl_period = 0.30 #0.5 # original 0.5
    Cfg.rewards.feet_air_time_rsl_curriculum = True

    Cfg.reward_scales.feet_air_time = 0.0 # 0.5
    Cfg.rewards.use_adaptive_period = False
    Cfg.rewards.contact_condition_scale = -1
    

    #negative rewards
    Cfg.reward_scales.base_height = -0.5
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.orientation = -0.40 #-2.0 # orientation -4.0

    # go1 urdf weight no backpack: 11.308932. Backpack weight: 3.211. Increase by 28%
    Cfg.reward_scales.torques = -0.00012 # best for A1: -0.00025
    Cfg.rewards.torque_hip_weight = 1.0
    Cfg.rewards.torque_pos_thigh_weight = 1.0
    Cfg.rewards.torque_pos_calf_weight = 1.0

    Cfg.reward_scales.dof_pos = -0.025 # rsl disabled, -0.1
    Cfg.rewards.dof_pos_hip_weight = 3.0
    Cfg.rewards.dof_pos_thigh_weight = 1.0
    Cfg.rewards.dof_pos_calf_weight = 0.0

    Cfg.reward_scales.dof_vel = 0
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.feet_slip = -0.025 # rsl disabled -0.025 # -0.05
    Cfg.reward_scales.collision = -0.10
    Cfg.reward_scales.lin_vel_z = -0.15
    Cfg.reward_scales.stair_elevation_progress = 8.0
    Cfg.rewards.stair_elevation_velocity_clip = 1.0
    Cfg.reward_scales.stair_forward_progress = 2.0
    Cfg.rewards.stair_forward_velocity_clip = 1.0
    Cfg.reward_scales.stair_lateral_drift = -8.0
    Cfg.rewards.stair_lateral_displacement_clip = 2.5
    Cfg.reward_scales.stair_heading_drift = -2.0
    Cfg.reward_scales.ang_vel_xy = -0.05

    Cfg.reward_scales.dof_pos_limits = -2.0
    Cfg.rewards.soft_dof_pos_limit = 0.9

    Cfg.rewards.soft_torque_limit = 0.7
    Cfg.reward_scales.torque_limits = -1.0
    


    # disabled 
    # Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0
    Cfg.reward_scales.raibert_heuristic = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -0.0
    Cfg.reward_scales.feet_clearance_ji22 = 0.80
    # The reward implementation adds the 2 cm foot radius, so this produces
    # an 18 cm foot-center target and leaves margin over a 12 cm riser.
    Cfg.rewards.feet_clearance_ji22_target = 0.16
    Cfg.reward_scales.action_smoothness_1 = 0
    Cfg.reward_scales.action_smoothness_2 = 0
    Cfg.reward_scales.feet_impact_vel = -0.0

    # Cfg.reward_scales.dof_pos_stancemode = -0.075#from -0.075
    # Cfg.rewards.use_adaptive_stancemode = True
    # Cfg.rewards.stancemode_multiplier = 15
    # Cfg.rewards.hip_weight = 30
    # Cfg.rewards.thigh_weight = 1
    # Cfg.rewards.calf_weight = 10
    # unused
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.


    #-------------
    # Learing config
    #-------------

    # A mature policy needs small, bounded updates. Checkpoints contain model
    # weights but no optimizer state, so resuming with the original adaptive
    # schedule can make the learning rate jump and destroy an otherwise good
    # controller. Resumed runs therefore use conservative PPO fine tuning.
    Cfg.cfg_ppo.algorithm.schedule = 'fixed' if resume_run else 'adaptive'
    Cfg.cfg_ppo.algorithm.learning_rate = 3.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.critic_learning_rate = 1.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.adaptation_module_learning_rate = 1.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.value_loss_coef = 0.25 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.value_huber_delta = 5.0 if resume_run else None
    Cfg.cfg_ppo.algorithm.clip_param = 0.10 if resume_run else 0.20
    Cfg.cfg_ppo.algorithm.entropy_coef = 0.001 if resume_run else 0.01
    Cfg.cfg_ppo.algorithm.num_learning_epochs = 3 if resume_run else 5
    Cfg.cfg_ppo.algorithm.max_grad_norm = 0.5 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.critic_max_grad_norm = 0.25 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.critic_update_loss_threshold = 25.0 if resume_run else None
    Cfg.cfg_ppo.algorithm.desired_kl = 0.008 if resume_run else 0.01
    Cfg.cfg_ppo.algorithm.hard_kl_limit = 0.020 if resume_run else None
    Cfg.cfg_ppo.algorithm.lr_adaptive_schedule_decay = 1.25
    Cfg.cfg_ppo.algorithm.action_clip = Cfg.normalization.clip_actions

    # Abort before a bad late-stage update can be saved as the new best model.
    # Actor and critic now use separate optimizers, while individual critic
    # minibatches above critic_update_loss_threshold are skipped.  A finite
    # run-level value-loss guard would incorrectly stop on normal episodic
    # return-scale transitions after changing rewards, so Phase-3B relies on
    # the critic minibatch guard plus the policy KL/non-finite guards.
    Cfg.cfg_ppo.runner.divergence_value_loss_threshold = None if resume_run else 100.0
    Cfg.cfg_ppo.runner.divergence_patience = 2
    # Reward and terrain changes invalidate the resumed critic's scale for the
    # first few rollouts.  Critic minibatches above its own loss threshold are
    # already skipped; delay the run-level finite-loss guard while it adapts.
    Cfg.cfg_ppo.runner.divergence_warmup_iterations = 0

    #-------------
    # Commands
    #-------------
    

    Cfg.commands.resampling_time = 6
    Cfg.commands.command_curriculum = True


    # heading command
    Cfg.commands.heading_command = False
    Cfg.commands.heading_range = [-3.14, 3.14]

    Cfg.commands.terrain_conditioned_stair_commands = True
    Cfg.commands.stair_forward_command_range = [0.30, 0.45]
    Cfg.commands.stair_lateral_command_limit = 0.0
    Cfg.commands.stair_yaw_command_limit = 0.0
    
    Cfg.commands.lin_vel_x = [-0.35, 0.55]
    Cfg.commands.limit_vel_x = [-0.70, 0.90]
    Cfg.commands.lin_vel_y = [-0.12, 0.12]
    Cfg.commands.limit_vel_y = [-0.30, 0.30]
    Cfg.commands.ang_vel_yaw = [-0.60, 0.60]
    Cfg.commands.limit_vel_yaw = [-1.0, 1.0]


    # Cfg.commands.lin_vel_x = [-0.5, 0.5]
    # Cfg.commands.limit_vel_x = [-1.0, 1.5] 
    # Cfg.commands.lin_vel_y = [-0.5, 0.5]
    # Cfg.commands.limit_vel_y = [-1.0, 1.0]
    # Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    # Cfg.commands.limit_vel_yaw = [-1.5, 1.5]

    # Cfg.commands.lin_vel_x = [-0.5, 0.5]
    # Cfg.commands.limit_vel_x = [-1.5, 1.5] 
    # Cfg.commands.lin_vel_y = [-0.5, 0.5]
    # Cfg.commands.limit_vel_y = [-1.0, 1.0]
    # Cfg.commands.ang_vel_yaw = [-1.0, 1.0]
    # Cfg.commands.limit_vel_yaw = [-1.5, 1.5]

    Cfg.commands.body_height_cmd = [0.0, 0.0]
    Cfg.commands.gait_frequency_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.0, 0.0]
    Cfg.commands.footswing_height_range = [0.0, 0.0]
    Cfg.commands.body_pitch_range = [0.0, 0.0]
    Cfg.commands.body_roll_range = [0.0, 0.0]
    Cfg.commands.stance_width_range = [0.0, 0.0]
    Cfg.commands.stance_length_range = [0.0, 0.0]
    Cfg.commands.aux_reward_coef_range = [0.0, 0.0]


    Cfg.commands.limit_body_height = [0.0, 0.0]
    Cfg.commands.limit_gait_frequency = [0.0, 0.0]
    Cfg.commands.limit_gait_phase = [0.0, 0.0]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.0, 0.0]
    Cfg.commands.limit_footswing_height = [0.0, 0.0]
    Cfg.commands.limit_body_pitch = [0.0, 0.0]
    Cfg.commands.limit_body_roll = [0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.0, 0.0]
    Cfg.commands.limit_stance_length = [0.0, 0.0]    
    Cfg.commands.limit_aux_reward_coef = [0.0, 0.0]

    Cfg.commands.num_bins_vel_x = 10 ##from 21
    Cfg.commands.num_bins_vel_y = 10 ## from 1
    Cfg.commands.num_bins_vel_yaw = 10 ## from 21
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    #Cfg.commands.gaitwise_curricula = False
    

    
    # cmd vx,vy,vtheta exactly 0 with standing_still_prob
    Cfg.commands.train_standing_still = True
    Cfg.commands.standing_still_prob = 0.05





    env = VelocityTrackingEasyEnv(sim_device=sim_device, headless=headless, cfg=Cfg, debug_viz=debug_viz)
    if preview and not headless:
        origin = env.env_origins[0].detach().cpu().numpy()
        env.set_camera(
            [origin[0] - 2.0, origin[1] - 2.0, origin[2] + 1.3],
            [origin[0], origin[1], origin[2] + 0.3],
        )

    # log the experiment parameters
    logger.log_params(Cfg=vars(Cfg))

    env = HistoryWrapper(env)
    gpu_id = 0
    runner = Runner(env, cfg = Cfg, device=f"cuda:{gpu_id}")
    if resume_run is not None and checkpoint >= 0:
        runner.current_learning_iteration = checkpoint
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True, eval_freq=50)


if __name__ == '__main__':
    import argparse
    import os
    from pathlib import Path
    from ml_logger import logger
    from robodog_gym import MINI_GYM_ROOT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    run_group = (
        f"mybot_v3_sim2sim_phase4_matched_resume_{args.checkpoint:06d}"
        if args.resume_run is not None
        else "mybot_v3_sim2sim_phase4_matched_from_scratch"
    )
    logger.configure(logger.utcnow(f'{run_group}/%Y-%m-%d_%H-%M-%S.%f'),
                     root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
    print("Loggind directory:", os.path.join(logger.root,logger.prefix))
    logger.log_text("""
                charts: 
                - yKey: train/episode/rew_total/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_lin_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_tracking_ang_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_air_time_rsl/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_air_time/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_clearance_ji22/mean
                  xKey: iterations
                - yKey: train/episode/rew_orientation/mean
                  xKey: iterations
                - yKey: train/episode/rew_base_height/mean
                  xKey: iterations
                - yKey: train/episode/rew_stair_elevation_progress/mean
                  xKey: iterations
                - yKey: train/episode/rew_stair_forward_progress/mean
                  xKey: iterations
                - yKey: train/episode/rew_stair_lateral_drift/mean
                  xKey: iterations
                - yKey: train/episode/rew_stair_heading_drift/mean
                  xKey: iterations
                - yKey: train/episode/rew_torques/mean
                  xKey: iterations
                - yKey: train/episode/rew_collision/mean
                  xKey: iterations


                - yKey: train/episode/rew_dof_pos/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_vel/mean
                  xKey: iterations
                - yKey: train/episode/rew_dof_acc/mean
                  xKey: iterations
                - yKey: train/episode/rew_action_rate/mean
                  xKey: iterations
                - yKey: train/episode/rew_ang_vel_xy/mean
                  xKey: iterations
                - yKey: train/episode/rew_lin_vel_z/mean
                  xKey: iterations
                - yKey: train/episode/rew_feet_slip/mean
                  xKey: iterations
                
               
                - yKey: train/episode/command_area_nominal/mean
                  xKey: iterations
                - yKey: train/episode/max_terrain_height/mean
                  xKey: iterations
                - type: video
                  glob: "videos/*.mp4"

                - yKey: mean_surrogate_loss/mean
                  xKey: iterations
                - yKey: mean_value_loss/mean
                  xKey: iterations
                - yKey: adaptation_loss/mean
                  xKey: iterations
                - yKey: mean_adaptation_module_test_loss/mean
                  xKey: iterations
                - yKey: learning_rate/mean
                  xKey: iterations
                - yKey: actor_grad_norm/mean
                  xKey: iterations
                - yKey: critic_grad_norm/mean
                  xKey: iterations
                - yKey: return_std/mean
                  xKey: iterations
                - yKey: action_saturation_fraction/mean
                  xKey: iterations
                - yKey: train/episode/curriculum_stair_success/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    train_mybot_v3_sim2sim_phase4(
        headless=args.headless,
        num_envs=args.num_envs,
        iterations=args.iterations,
        seed=args.seed,
        sim_device=args.sim_device,
        preview=args.preview,
        resume_run=args.resume_run,
        checkpoint=args.checkpoint,
    )
    # Isaac Gym may segfault while Python tears down the PhysX/CUDA objects
    # after every checkpoint has already been synchronously written.  Exit
    # directly at this safe point so a completed run reports success.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
