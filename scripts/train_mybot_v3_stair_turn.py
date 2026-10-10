"""Train Mybot V3 from scratch for stairs and planar velocity commands.

The actor observes a local height map. Terrain and velocity curricula cover
ascending stairs, descending stairs, forward/reverse motion, lateral motion,
and left/right turns. No checkpoint is loaded by this entry point.
"""


def train_mybot_v3_stair_turn(
    headless=True,
    num_envs=4096,
    iterations=60000,
    seed=42,
    sim_device="cuda:0",
    preview=False,
    resume_run=None,
    checkpoint=-1,
    robocon_obstacle="mixed",
    allow_observation_expansion=False,
    save_interval=250,
    divergence_warmup_iterations=0,
    resume_action_std=None,
    freeze_resume_action_std=False,
    observation_adapter_only=False,
    freeze_adaptation_module=False,
    resume_learning_rate=None,
    policy_anchor_coef=0.0,
    low_bar_lateral_init_range=0.0,
    low_bar_yaw_init_range=0.0,
    low_bar_spec_only=False,
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

    if observation_adapter_only:
        if resume_run is None:
            raise ValueError(
                "--observation-adapter-only requires --resume-run")
        if robocon_obstacle != "low_bar":
            raise ValueError(
                "--observation-adapter-only is currently defined only for low_bar")
        if resume_action_std is None:
            raise ValueError(
                "--observation-adapter-only requires --resume-action-std")
    if freeze_adaptation_module and resume_run is None:
        raise ValueError("--freeze-adaptation-module requires --resume-run")
    if resume_learning_rate is not None:
        if resume_run is None:
            raise ValueError("--resume-learning-rate requires --resume-run")
        if resume_learning_rate <= 0.0:
            raise ValueError("--resume-learning-rate must be positive")
    if policy_anchor_coef < 0.0:
        raise ValueError("--policy-anchor-coef must be non-negative")
    if policy_anchor_coef > 0.0 and resume_run is None:
        raise ValueError("--policy-anchor-coef requires --resume-run")

    Cfg.env.num_envs = num_envs
    Cfg.cfg_ppo.seed = seed
    Cfg.cfg_ppo.runner.resume = resume_run is not None
    Cfg.cfg_ppo.runner.resume_path = resume_run
    Cfg.cfg_ppo.runner.checkpoint = checkpoint
    Cfg.cfg_ppo.runner.allow_observation_expansion = allow_observation_expansion
    Cfg.cfg_ppo.runner.resume_action_std = resume_action_std
    Cfg.cfg_ppo.runner.freeze_resume_action_std = freeze_resume_action_std
    Cfg.cfg_ppo.runner.observation_adapter_only = observation_adapter_only
    Cfg.cfg_ppo.runner.freeze_adaptation_module = freeze_adaptation_module
    # The low-bar contract preserves the original distance/lateral/vertical
    # fields and appends heading sin/cos, absolute clearance, and validity.
    Cfg.cfg_ppo.runner.observation_adapter_width = (
        7 if robocon_obstacle == "low_bar" else 0)
    Cfg.cfg_ppo.runner.save_interval = save_interval
    Cfg.cfg_ppo.runner.save_video_interval = 0
    Cfg.cfg_ppo.runner.save_curriculum_plot_interval = 250
    Cfg.cfg_ppo.runner.wandb_logging = False
    Cfg.env.record_video = False
    Cfg.env.export_step_telemetry = False
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
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True # wtw true

    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.4, 1.5]
    Cfg.domain_rand.randomize_restitution = True  # wtw true
    Cfg.domain_rand.restitution_range = [0.0, 0.2]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.0, 1.5]
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.03, 0.03]
    Cfg.domain_rand.randomize_ground_friction = False 
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True  # wtw true
    Cfg.domain_rand.motor_strength_range = [0.9, 1.1]
    Cfg.domain_rand.randomize_motor_offset = True # wtw true
    Cfg.domain_rand.motor_offset_range = [-0.02, 0.02]
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_rigids_after_start = True
    Cfg.domain_rand.randomize_friction_indep = False


    # gravity changes and pushes
    Cfg.domain_rand.randomize_gravity = True # wtw true
    Cfg.domain_rand.gravity_range = [-0.3, 0.3]
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
    Cfg.terrain.max_init_terrain_level = 4 if resume_run else 2
    Cfg.terrain.curriculum_min_progress = 1.0
    Cfg.terrain.curriculum_stair_min_progress = 2.0
    Cfg.terrain.curriculum_stair_min_elevation = 0.06
    Cfg.terrain.terrain_length = 8.
    Cfg.terrain.terrain_width = 8.
    Cfg.terrain.num_rows = 12
    Cfg.terrain.num_cols = 50
    # [smooth slope, rough slope, stairs up, stairs down, discrete,
    #  stepping stones, empty, smooth flat, rough flat]
    Cfg.terrain.terrain_proportions = [0.04, 0.04, 0.34, 0.34, 0.10, 0.0, 0.0, 0.08, 0.06]

    Cfg.terrain.mesh_type = "trimesh"
  

    # wtw additions (here set them to be same as RSL)
    Cfg.terrain.yaw_init_range = 0.15
    Cfg.terrain.difficulty_scale = 0.90
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
    Cfg.terrain.x_init_range = 0.1
    Cfg.terrain.y_init_range = 0.1
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False # center robots in the middle of the terrain grid, only makes sense without curriculum 

    
    # this will be averaged with domain_rand.friction_range if randomization is on
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0

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
     


    # ==================================================================
    # ROBOCON obstacle terrain (reference: robocon_reference/RC_WheelLeg)
    # ------------------------------------------------------------------
    # Phase switch:
    #   "mixed"  -> keep the default mixed terrain curriculum above
    #   "stairs" -> single ROBOCON staircase family, geometry matched to the
    #               competition T-stairs (step_height ~0.10 m)
    # Extra obstacles (low bar / gap bridge / ramp) will be added here as
    # their generators land; see robocon_reference/.../competition_terrains.py.
    # ==================================================================
    Cfg.terrain.robocon_obstacle = robocon_obstacle

    if Cfg.terrain.robocon_obstacle == "stairs":
        # ROBOCON competition staircase: identical geometry to the MuJoCo
        # reference. Tread depth 0.30 m, ladder width 2.0 m, 8 risers,
        # 0.90 m top landing, first riser 0.55 m in front of the spawn.
        Cfg.terrain.sim2sim_straight_stairs = True
        Cfg.terrain.sim2sim_stair_start_offset = 0.55
        Cfg.terrain.sim2sim_stair_tread_depth = 0.30
        Cfg.terrain.sim2sim_stair_width = 2.0
        Cfg.terrain.sim2sim_stair_num_steps = 8
        Cfg.terrain.sim2sim_stair_top_length = 0.90

        # Competition step height is 0.10 m. For RL we widen the *per-step*
        # height and co-vary it with the terrain row (curriculum index), so
        # lower rows teach a smaller riser and the top rows slightly exceed
        # the competition spec. Index i (terrain row) drives the height table
        # consumed by Terrain.make_terrain and the stair-elevation promotion
        # rule in legged_robot._update_terrain_curriculum.
        step_h_min = 0.06   # easiest riser
        step_h_max = 0.14   # hardest riser (competition = 0.10)
        num_levels = Cfg.terrain.num_rows
        if num_levels > 1:
            step_heights = [
                step_h_min + (step_h_max - step_h_min) * i / (num_levels - 1)
                for i in range(num_levels)
            ]
        else:
            step_heights = [step_h_max]
        # Guarantee the exact competition riser is a selectable level.
        step_heights[num_levels // 2] = 0.10
        Cfg.terrain.sim2sim_stair_step_heights = step_heights

        # Train only on the staircase family; disable the mixed curriculum rows.
        Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Stair-aware promotion: require real forward progress and ~6 risers
        # of elevation gain before a robot levels up.
        Cfg.terrain.curriculum_stair_min_progress = 2.80
        Cfg.terrain.curriculum_stair_min_elevation = 0.06
        Cfg.terrain.curriculum_stair_min_elevation_steps = 6.0

    elif Cfg.terrain.robocon_obstacle == "low_bar":
        # ROBOCON low-bar (限高杆 / 矮门): flat ground + a real collision
        # gate (cross-bar + two posts) placed inside each env.  The bar CANNOT
        # be faked with the 2.5D heightfield, so it is a fixed-base, gravity-
        # free actor that the robot collides with.
        Cfg.terrain.robocon_low_bar = True
        Cfg.env.episode_length_s = 10.0
        Cfg.terrain.max_init_terrain_level = 2
        Cfg.terrain.x_init_range = 0.0
        # Phase 1 defaults to a centred nominal task. Pose perturbations are
        # explicit CLI-controlled curriculum stages, so a failed robustness
        # experiment cannot silently become the next run's default.
        Cfg.terrain.y_init_range = low_bar_lateral_init_range
        Cfg.terrain.yaw_init_range = low_bar_yaw_init_range
        # Flat ground everywhere: index 6 = "empty" leaves the height field at 0.
        Cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        Cfg.terrain.low_bar_clearance_min = 0.25       # hardest row (lowest bar)
        Cfg.terrain.low_bar_clearance_max = 0.35       # easiest row (highest bar)
        Cfg.terrain.low_bar_target_clearance = 0.30    # ROBOCON competition spec
        if low_bar_spec_only:
            # M1.3: train the competition clearance and below without changing
            # the pose distribution at the same time, so a bar-contact fix is
            # not confounded with an alignment fix.
            Cfg.terrain.low_bar_clearance_max = Cfg.terrain.low_bar_target_clearance
        Cfg.terrain.low_bar_width = 1.0
        Cfg.terrain.low_bar_thickness = 0.05
        Cfg.terrain.low_bar_x = 2.0                    # bar ~2 m ahead of spawn
        Cfg.terrain.low_bar_contact_force_threshold = 1.0
        Cfg.terrain.low_bar_pass_margin = 0.05
        Cfg.terrain.low_bar_gate_margin = 0.02
        Cfg.terrain.low_bar_stable_roll_pitch = 0.60
        Cfg.terrain.low_bar_stable_base_height = 0.16
        Cfg.terrain.low_bar_recovery_steps = 50        # 1 s at 50 Hz
        Cfg.terrain.low_bar_terminate_on_success = True
        Cfg.terrain.low_bar_alignment_approach = 3.0
        Cfg.terrain.low_bar_alignment_tolerance = 0.15
        Cfg.terrain.low_bar_heading_tolerance = 0.10
        Cfg.terrain.low_bar_heading_weight = 1.0
        Cfg.terrain.low_bar_crouch_margin = 0.08
        Cfg.terrain.low_bar_crouch_shaping_span = 0.20
        # M1.3 three-phase shaping.  body_top is the term that actually
        # constrains the front leg; the crossing-speed term keeps the policy
        # from trading progress for clearance.
        Cfg.terrain.low_bar_body_top_window = 0.35
        # Legs get the strict ceiling (they are the measured colliders); the
        # trunk is allowed above the edge so it cannot dominate the term.
        Cfg.terrain.low_bar_body_top_safety = 0.05
        Cfg.terrain.low_bar_trunk_safety = -0.10
        Cfg.terrain.low_bar_recover_distance = 0.35
        Cfg.terrain.low_bar_recover_span = 0.06
        Cfg.terrain.low_bar_crossing_min_vx = 0.25

        # Stage 1 first proves the task under nominal dynamics. Parameter and
        # sensor randomization are introduced only after clean success reaches
        # roughly 80%, otherwise the sparse pass signal is overwhelmed.
        Cfg.domain_rand.randomize_friction = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.randomize_base_mass = False
        Cfg.domain_rand.randomize_com_displacement = False
        Cfg.domain_rand.randomize_motor_strength = False
        Cfg.domain_rand.randomize_motor_offset = False
        Cfg.domain_rand.randomize_gravity = False
        Cfg.domain_rand.randomize_lag_timesteps = False

        # Difficulty = terrain row, in the OPPOSITE direction to the stairs:
        # row 0 = highest bar (0.35 m, easiest) ... last row = lowest (0.25 m).
        # Force the middle row to the exact competition clearance.
        lb_min = Cfg.terrain.low_bar_clearance_min
        lb_max = Cfg.terrain.low_bar_clearance_max
        lb_levels = Cfg.terrain.num_rows
        if lb_levels > 1:
            lb_table = [lb_max + (lb_min - lb_max) * i / (lb_levels - 1)
                        for i in range(lb_levels)]
        else:
            lb_table = [lb_max]
        if not low_bar_spec_only:
            lb_table[lb_levels // 2] = Cfg.terrain.low_bar_target_clearance
        Cfg.terrain.robocon_low_bar_clearance_by_level = lb_table

        # The policy must be able to SEE the overhead bar (the downward height
        # scanner cannot): append a body-frame bar observation to policy and
        # estimator (must be in both -- estimator activity is keyed on the
        # policy component list).  Widths auto-recompute in Observations.__init__.
        Cfg.env.policy_observation_components.append(['obstacle_ahead', 7, True])
        Cfg.env.estimator_observation_components.append(['obstacle_ahead', 7, True])

        # Keep the bar observation clean: the default noise scale is 1.0 m
        # (noise_level == 1.0), which would swamp a ~3 m signal.
        Cfg.noise_scales.obstacle_ahead = 0.0

        # Dense shaping teaches the lowering motion. Event rewards define the
        # actual task and prevent reward hacking by staying crouched or simply
        # walking around the posts. Event reward functions divide by dt so
        # these coefficients are the true one-shot reward/penalty magnitudes.
        Cfg.reward_scales.low_bar_crouch = 1.0
        Cfg.reward_scales.low_bar_body_top = -100.0
        Cfg.reward_scales.low_bar_recover = 1.5
        Cfg.reward_scales.low_bar_crossing_speed = -4.0
        Cfg.reward_scales.low_bar_alignment = -1.0
        Cfg.reward_scales.low_bar_pass = 3.0
        Cfg.reward_scales.low_bar_success = 7.0
        Cfg.reward_scales.low_bar_collision = -5.0
        Cfg.reward_scales.low_bar_missed_gate = -5.0


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
    Cfg.rewards.feet_air_time_rsl_period = 0.25 #0.5 # original 0.5
    Cfg.rewards.feet_air_time_rsl_curriculum = True

    Cfg.reward_scales.feet_air_time = 0.0 # 0.5
    Cfg.rewards.use_adaptive_period = False
    Cfg.rewards.contact_condition_scale = -1
    

    #negative rewards
    Cfg.reward_scales.base_height = 0 # exteroceptive -10
    Cfg.rewards.base_height_target = 0.30
    Cfg.reward_scales.orientation = -0.25 #-2.0 # orientation -4.0

    # go1 urdf weight no backpack: 11.308932. Backpack weight: 3.211. Increase by 28%
    Cfg.reward_scales.torques = -0.00020 # best for A1: -0.00025
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
    Cfg.reward_scales.collision = -1.
    Cfg.reward_scales.lin_vel_z = -2.0
    Cfg.reward_scales.ang_vel_xy = -0.05

    Cfg.reward_scales.dof_pos_limits = -2.0
    Cfg.rewards.soft_dof_pos_limit = 0.9

    Cfg.rewards.soft_torque_limit = 0.7
    Cfg.reward_scales.torque_limits = -2.0
    


    # disabled 
    # Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0
    Cfg.reward_scales.raibert_heuristic = -0.0
    Cfg.reward_scales.feet_clearance_cmd_linear = -0.0
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
    Cfg.cfg_ppo.algorithm.learning_rate = 2.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.critic_learning_rate = 1.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.adaptation_module_learning_rate = 1.e-5 if resume_run else 1.e-3
    Cfg.cfg_ppo.algorithm.value_loss_coef = 0.5 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.value_huber_delta = 5.0 if resume_run else None
    Cfg.cfg_ppo.algorithm.clip_param = 0.10 if resume_run else 0.20
    Cfg.cfg_ppo.algorithm.entropy_coef = 0.002 if resume_run else 0.01
    if robocon_obstacle == "low_bar" and resume_run:
        Cfg.cfg_ppo.algorithm.entropy_coef = 0.0 if freeze_resume_action_std else 0.0005
    Cfg.cfg_ppo.algorithm.num_learning_epochs = 3 if resume_run else 5
    Cfg.cfg_ppo.algorithm.max_grad_norm = 0.5 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.critic_max_grad_norm = 0.5 if resume_run else 1.0
    Cfg.cfg_ppo.algorithm.desired_kl = 0.008 if resume_run else 0.01
    Cfg.cfg_ppo.algorithm.hard_kl_limit = 0.020 if resume_run else None
    Cfg.cfg_ppo.algorithm.lr_adaptive_schedule_decay = 1.25
    Cfg.cfg_ppo.algorithm.action_clip = Cfg.normalization.clip_actions
    Cfg.cfg_ppo.algorithm.policy_anchor_coef = policy_anchor_coef

    # With a smaller fixed exploration std, a parameter step produces a much
    # larger policy KL. Use a correspondingly smaller actor step so PPO can
    # improve the obstacle policy instead of tripping the hard-KL guard on
    # nearly every minibatch.
    if (robocon_obstacle == "low_bar" and resume_run
            and resume_action_std is not None
            and resume_action_std <= 0.30):
        if resume_action_std <= 0.05:
            conservative_lr = 2.e-7
        elif resume_action_std <= 0.15:
            conservative_lr = 1.e-6
        else:
            conservative_lr = 5.e-6
        Cfg.cfg_ppo.algorithm.learning_rate = conservative_lr
        Cfg.cfg_ppo.algorithm.adaptation_module_learning_rate = conservative_lr
    if observation_adapter_only:
        Cfg.cfg_ppo.algorithm.learning_rate = 2.e-6
    if resume_learning_rate is not None:
        Cfg.cfg_ppo.algorithm.learning_rate = resume_learning_rate

    # Abort before a bad late-stage update can be saved as the new best model.
    Cfg.cfg_ppo.runner.divergence_value_loss_threshold = 25.0 if resume_run else 100.0
    Cfg.cfg_ppo.runner.divergence_patience = 3
    Cfg.cfg_ppo.runner.divergence_warmup_iterations = divergence_warmup_iterations

    #-------------
    # Commands
    #-------------
    

    Cfg.commands.resampling_time = 6
    Cfg.commands.command_curriculum = True


    # heading command
    Cfg.commands.heading_command = True # need for terrain curriculum trainig, robot reaches farther places
    Cfg.commands.heading_range = [-3.14, 3.14]
    
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
    Cfg.commands.standing_still_prob = 0.10

    if robocon_obstacle == "low_bar":
        # An obstacle expert must repeatedly approach the gate. Reverse and
        # large turn commands let the policy avoid the task while collecting
        # ordinary locomotion reward, so keep only realistic approach errors.
        Cfg.commands.heading_command = False
        Cfg.commands.command_curriculum = False
        Cfg.commands.resampling_time = Cfg.env.episode_length_s
        Cfg.commands.lin_vel_x = [0.50, 0.50]
        Cfg.commands.limit_vel_x = [0.50, 0.50]
        # Phase 1 validates a centred, straight pass. Lateral/yaw command
        # perturbations are introduced only after nominal success exceeds the
        # milestone; otherwise velocity tracking explicitly asks the policy to
        # leave the gate centre while the task reward asks it to stay there.
        Cfg.commands.lin_vel_y = [0.0, 0.0]
        Cfg.commands.limit_vel_y = [0.0, 0.0]
        Cfg.commands.ang_vel_yaw = [0.0, 0.0]
        Cfg.commands.limit_vel_yaw = [0.0, 0.0]
        Cfg.commands.train_standing_still = False
        Cfg.commands.standing_still_prob = 0.0





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
    # Random episode ages are useful for stationary locomotion, but they
    # truncate the approach-pass-recovery sequence on the first low-bar
    # rollout and leave a short fine-tuning block with almost no successes.
    runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=robocon_obstacle != "low_bar",
        eval_freq=50,
    )


if __name__ == '__main__':
    import argparse
    import os
    from pathlib import Path
    from ml_logger import logger
    from robodog_gym import MINI_GYM_ROOT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--resume-run", type=str, default=None)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--robocon-obstacle", choices=["mixed", "stairs", "low_bar"],
                        default="mixed")
    parser.add_argument(
        "--low-bar-spec-only",
        action="store_true",
        help=("Restrict the low-bar clearance table to the competition spec and "
              "below (0.25..0.30 m).  Used by the M1.3 bar-contact block."),
    )
    parser.add_argument(
        "--allow-observation-expansion",
        action="store_true",
        help=("Expand only the known CSE input layers when a resumed checkpoint "
              "has fewer per-frame observations; new appended fields start at zero."),
    )
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument(
        "--divergence-warmup-iterations",
        type=int,
        default=0,
        help=("Delay the value-loss divergence guard while a migrated critic "
              "adapts to a changed observation/reward contract."),
    )
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=None,
        help=("Reset every action standard deviation after loading a checkpoint; "
              "use a smaller value for conservative obstacle fine tuning."),
    )
    parser.add_argument(
        "--freeze-resume-action-std",
        action="store_true",
        help="Keep --resume-action-std fixed instead of optimizing it.",
    )
    parser.add_argument(
        "--observation-adapter-only",
        action="store_true",
        help=("Freeze the mature actor and train only the input columns for "
              "the newly appended low-bar obstacle observation."),
    )
    parser.add_argument(
        "--freeze-adaptation-module",
        action="store_true",
        help="Freeze the resumed estimator while fine tuning the full actor.",
    )
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        default=None,
        help="Override the actor learning rate for a resumed run.",
    )
    parser.add_argument(
        "--policy-anchor-coef",
        type=float,
        default=0.0,
        help=("Penalize mean-action drift from the policy loaded at the start "
              "of this run."),
    )
    parser.add_argument(
        "--low-bar-lateral-init-range",
        type=float,
        default=0.0,
        help="Low-bar spawn lateral randomization half-range in metres.",
    )
    parser.add_argument(
        "--low-bar-yaw-init-range",
        type=float,
        default=0.0,
        help="Low-bar spawn yaw randomization half-range in radians.",
    )
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    task_name = "mybot_v3" if args.robocon_obstacle == "mixed" else f"mybot_v3_{args.robocon_obstacle}_expert"
    run_group = (
        f"{task_name}_resume_{args.checkpoint:06d}"
        if args.resume_run is not None
        else f"{task_name}_from_scratch"
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
                - yKey: train/episode/rew_orientation/mean
                  xKey: iterations
                - yKey: train/episode/rew_base_height/mean
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
                - yKey: train/episode/low_bar_success_rate/mean
                  xKey: iterations
                - yKey: train/episode/rew_low_bar_alignment/mean
                  xKey: iterations
                - yKey: train/episode/low_bar_pass_rate/mean
                  xKey: iterations
                - yKey: train/episode/low_bar_collision_rate/mean
                  xKey: iterations
                - yKey: train/episode/low_bar_missed_gate_rate/mean
                  xKey: iterations
                - yKey: train/episode/curriculum_low_bar_success/mean
                  xKey: iterations
                """, filename=".charts.yml", dedent=True)

    train_mybot_v3_stair_turn(
        headless=args.headless,
        num_envs=args.num_envs,
        iterations=args.iterations,
        seed=args.seed,
        sim_device=args.sim_device,
        preview=args.preview,
        resume_run=args.resume_run,
        checkpoint=args.checkpoint,
        robocon_obstacle=args.robocon_obstacle,
        allow_observation_expansion=args.allow_observation_expansion,
        save_interval=args.save_interval,
        divergence_warmup_iterations=args.divergence_warmup_iterations,
        resume_action_std=args.resume_action_std,
        freeze_resume_action_std=args.freeze_resume_action_std,
        observation_adapter_only=args.observation_adapter_only,
        freeze_adaptation_module=args.freeze_adaptation_module,
        resume_learning_rate=args.resume_learning_rate,
        policy_anchor_coef=args.policy_anchor_coef,
        low_bar_lateral_init_range=args.low_bar_lateral_init_range,
        low_bar_yaw_init_range=args.low_bar_yaw_init_range,
        low_bar_spec_only=args.low_bar_spec_only,
    )
