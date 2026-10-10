import torch
import numpy as np
from robodog_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi

class CoRLRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------


    def _reward_tracking_lin_vel(self):
        # sigma curriculum
        # ppo iteration should be simulator step / 24 (num_steps_per_env hardcoded in RunnerArgs)
        # (in output, "iterations" is simulat steps * num envs 4000?)
        iteration = self.env.common_step_counter / self.env.cfg.cfg_ppo.runner.num_steps_per_env
        tracking_sigma = self.env.cfg.rewards.tracking_sigma # TODO  implement curriculum
        target_iteration = self.env.cfg.rewards.end_sigma_curriculum_iter
        start_tracking_sigma = self.env.cfg.rewards.start_tracking_sigma
        if iteration >= target_iteration:
            tracking_sigma = self.env.cfg.rewards.tracking_sigma
        else:
            tracking_sigma = start_tracking_sigma - (start_tracking_sigma - self.env.cfg.rewards.tracking_sigma) * (iteration / target_iteration)
        # if iteration%10==0:
        #     print("Tracking sigma: ", tracking_sigma, " Iteration: ", iteration)
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.pow(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2],4), dim=1)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        #return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)
        return torch.exp(-lin_vel_error / tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # sigma curriculum
        # ppo iteration should be simulator step / 24 (num_steps_per_env hardcoded in RunnerArgs)
        # (in output, "iterations" is simulat steps * num envs 4000?)
        iteration = self.env.common_step_counter / 24
        tracking_sigma_yaw = self.env.cfg.rewards.tracking_sigma_yaw # TODO  implement curriculum
        target_iteration = self.env.cfg.rewards.end_sigma_yaw_curriculum_iter 
        start_tracking_sigma_yaw = self.env.cfg.rewards.start_tracking_sigma_yaw
        if iteration >= target_iteration:
            tracking_sigma_yaw = self.env.cfg.rewards.tracking_sigma_yaw
        else:
            tracking_sigma_yaw = start_tracking_sigma_yaw - (start_tracking_sigma_yaw - self.env.cfg.rewards.tracking_sigma_yaw) * (iteration / target_iteration)
        # if iteration%10==0:
        #     print("Tracking sigma: ", tracking_sigma_yaw, " Iteration: ", iteration)
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        #return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
        return torch.exp(-ang_vel_error / tracking_sigma_yaw)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_stair_elevation_progress(self):
        """Reward signed world-frame elevation progress on ascending stairs.

        Using signed vertical velocity makes this a dense approximation of a
        potential-based height reward: climbing earns reward while slipping
        back down removes it.  Restricting it to ascending-stair columns keeps
        flat-ground locomotion and descending stairs unchanged.
        """
        terrain_proportions = list(self.env.cfg.terrain.terrain_proportions)
        stair_begin = float(sum(terrain_proportions[:2]))
        stair_split = float(sum(terrain_proportions[:3]))
        terrain_choice = (
            self.env.terrain_types.float()
            / float(self.env.cfg.terrain.num_cols)
            + 0.001
        )
        stairs_up = (terrain_choice >= stair_begin) & (terrain_choice < stair_split)
        max_abs_velocity = float(getattr(
            self.env.cfg.rewards, "stair_elevation_velocity_clip", 1.0))
        world_vertical_velocity = torch.clamp(
            self.env.root_states[:self.env.num_envs, 9],
            min=-max_abs_velocity,
            max=max_abs_velocity,
        )
        return stairs_up.float() * world_vertical_velocity

    def _reward_stair_lateral_drift(self):
        """Penalize bypassing ascending stairs by drifting around them."""
        terrain_proportions = list(self.env.cfg.terrain.terrain_proportions)
        stair_begin = float(sum(terrain_proportions[:2]))
        stair_split = float(sum(terrain_proportions[:3]))
        terrain_choice = (
            self.env.terrain_types.float()
            / float(self.env.cfg.terrain.num_cols)
            + 0.001
        )
        stairs_up = (terrain_choice >= stair_begin) & (terrain_choice < stair_split)
        max_abs_displacement = float(getattr(
            self.env.cfg.rewards, "stair_lateral_displacement_clip", 2.5))
        lateral_displacement = torch.clamp(
            self.env.root_states[:self.env.num_envs, 1]
            - self.env.env_origins[:self.env.num_envs, 1],
            min=-max_abs_displacement,
            max=max_abs_displacement,
        )
        return stairs_up.float() * torch.square(lateral_displacement)

    def _reward_stair_heading_drift(self):
        """Penalize turning away from the world-x staircase direction."""
        terrain_proportions = list(self.env.cfg.terrain.terrain_proportions)
        stair_begin = float(sum(terrain_proportions[:2]))
        stair_split = float(sum(terrain_proportions[:3]))
        terrain_choice = (
            self.env.terrain_types.float()
            / float(self.env.cfg.terrain.num_cols)
            + 0.001
        )
        stairs_up = (terrain_choice >= stair_begin) & (terrain_choice < stair_split)
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading_error = torch.atan2(forward[:, 1], forward[:, 0])
        return stairs_up.float() * torch.square(heading_error)

    def _reward_stair_forward_progress(self):
        """Reward signed progress along the matched staircase centreline.

        Velocity tracking alone can settle into walking against the first
        riser.  This potential-style term directly values forward displacement
        and removes the reward again when the robot slides backwards.  The
        lateral-drift penalty prevents collecting it by going around the
        finite-width staircase.
        """
        terrain_proportions = list(self.env.cfg.terrain.terrain_proportions)
        stair_begin = float(sum(terrain_proportions[:2]))
        stair_split = float(sum(terrain_proportions[:3]))
        terrain_choice = (
            self.env.terrain_types.float()
            / float(self.env.cfg.terrain.num_cols)
            + 0.001
        )
        stairs_up = (terrain_choice >= stair_begin) & (terrain_choice < stair_split)
        max_abs_velocity = float(getattr(
            self.env.cfg.rewards, "stair_forward_velocity_clip", 1.0))
        world_forward_velocity = torch.clamp(
            self.env.root_states[:self.env.num_envs, 7],
            min=-max_abs_velocity,
            max=max_abs_velocity,
        )
        return stairs_up.float() * world_forward_velocity

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)

    def _reward_ang_vel_xy_linear(self):
        # Penalize xy axes base angular velocity, linearly
        return torch.sum(torch.abs(self.env.base_ang_vel[:, :2]), dim=1)
    
    def _reward_ang_vel_xy_sqrt(self):
        # Penalize xy axes base angular velocity, with a square root
        return torch.sum(torch.sqrt(torch.abs(self.env.base_ang_vel[:, :2])), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        #TODO: add parameters for pitch and roll scale
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.torque_hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.torque_thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.torque_calf_weight

        return torch.sum(torch.square(self.env.torques) * weights, dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.env.dof_vel) - self.env.dof_vel_limits*self.env.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.env.torques) - self.env.torque_limits*self.env.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_jump(self):
        # body height tracking, no idea why it is called jump
        reference_heights = torch.mean((self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) - self.env.feet_height, dim=1) #avg terrain height under the feet
        body_height = self.env.base_pos[:, 2]  - reference_heights
        if self.env.cfg.commands.num_commands > 3:
            jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        else:
            jump_height_target = self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:self.env.num_envs, 2].unsqueeze(1) - self.env.measured_heights, dim=1)
        return torch.square(base_height - self.env.cfg.rewards.base_height_target)

    def _reward_low_bar_crouch(self):
        """Encourage ducking while a low bar is still ahead of the robot.

        The velocity-tracking task never asks the robot to lower its body, so
        without this term there is no incentive to crouch under the bar.

        bar_bottom (rel[:, 2]) is the bar's ground clearance measured from the
        base: it is 0 when the base sits level with the bar bottom and grows
        positive as the robot ducks below it.  A linear clearance score keeps
        a useful gradient even when the robot starts above the bar; the old
        positive clamp returned exactly zero in that most important failure
        region.  The score is active only while the bar is ahead, so there is
        no incentive to stay crouched after passing.
        """
        rel = self.env.low_bar_relative_state()
        forward = rel[:, 0]                      # + = bar ahead in base frame
        bar_bottom = rel[:, 2]                   # base height below bar bottom
        approach = float(getattr(self.env.cfg.terrain, 'low_bar_crouch_approach', 4.5))
        margin = float(getattr(self.env.cfg.terrain, 'low_bar_crouch_margin', 0.08))
        shaping_span = float(getattr(
            self.env.cfg.terrain, 'low_bar_crouch_shaping_span', 0.20))
        active = (forward > 0.0) & (forward < approach)
        clearance_deficit = torch.clamp(margin - bar_bottom, min=0.0)
        crouch = 1.0 - torch.clamp(
            clearance_deficit / max(shaping_span, 1e-6), 0.0, 1.0)
        return active.float() * crouch

    def _low_bar_window_state(self):
        """Per-body bar-relative x and z plus the per-env bar clearance."""
        env = self.env
        bar_x = env.env_origins[:, 0] + float(
            getattr(env.cfg.terrain, 'low_bar_x', 2.0))
        bodies = env.rigid_body_state.view(env.num_envs, env.num_bodies, 13)
        x_rel = bodies[:, :, 0] - bar_x.unsqueeze(1)
        z = bodies[:, :, 2]
        return x_rel, z, env.low_bar_clearance

    def _reward_low_bar_body_top(self):
        """Penalize any rigid body above the bar's lower edge inside the gate.

        Measured failure mode: the robot hits the bar with the front leg, not
        the trunk.  6 of 18 first contacts were the swinging front foot exactly
        at the bar's lower edge and 12 of 18 the front thigh within a few
        centimetres of it, while the trunk was still ~0.3 m short of the bar.
        _reward_low_bar_crouch only sees the base, so a term over every rigid
        body is the only way to shape the leg that actually collides.

        Returns the worst violation in metres, so the caller's negative scale
        is a penalty per metre above the allowed height.

        Legs and the trunk get different ceilings.  The first M1.3 block used a
        single 0.06 m margin for every body; the trunk then dominated the term
        (it sits near the bar edge anyway) and the penalty saturated at an
        unreachable value, while the collision rate stayed at 5.8%.  The legs
        are the measured colliders, so they keep the strict ceiling and the
        rest of the robot only has to avoid a gross violation.
        """
        env = self.env
        if not getattr(env.cfg.terrain, 'robocon_low_bar', False):
            return torch.zeros(env.num_envs, device=env.device)
        window = float(getattr(env.cfg.terrain, 'low_bar_body_top_window', 0.35))
        leg_safety = float(getattr(env.cfg.terrain, 'low_bar_body_top_safety', 0.05))
        trunk_safety = float(getattr(env.cfg.terrain, 'low_bar_trunk_safety', -0.10))
        x_rel, z, clearance = self._low_bar_window_state()
        inside = (x_rel.abs() <= window).float()
        leg_mask = self._low_bar_leg_mask()
        ceiling = clearance.unsqueeze(1) - (
            leg_mask * leg_safety + (1.0 - leg_mask) * trunk_safety)
        excess = torch.clamp(z - ceiling, min=0.0)
        return (excess * inside).amax(dim=1)

    def _low_bar_leg_mask(self):
        """Per-body mask selecting the leg links, cached on first use.

        The bar-contact trace attributed the first contact to a foot or a
        thigh, so those links carry the strict ceiling.
        """
        cached = getattr(self, '_low_bar_leg_mask_cache', None)
        if cached is not None:
            return cached
        env = self.env
        keys = ('foot', 'calf', 'thigh', 'toe', 'knee', 'shank')
        names = list(getattr(env, 'body_names', []))
        if len(names) == env.num_bodies:
            mask = torch.tensor(
                [1.0 if any(key in str(name).lower() for key in keys) else 0.0
                 for name in names],
                dtype=torch.float, device=env.device)
        else:
            mask = None
        if mask is None:
            # Fall back to applying the strict ceiling everywhere rather than
            # silently disabling the term.
            mask = torch.ones(env.num_bodies, dtype=torch.float, device=env.device)
        self._low_bar_leg_mask_cache = mask
        return mask

    def _reward_low_bar_recover(self):
        """Reward regaining normal height after the gate.

        Without this the cheapest way to satisfy the crossing term is to stay
        flattened for the rest of the episode, which the plan explicitly
        forbids.
        """
        env = self.env
        if not getattr(env.cfg.terrain, 'robocon_low_bar', False):
            return torch.zeros(env.num_envs, device=env.device)
        distance = float(getattr(env.cfg.terrain, 'low_bar_recover_distance', 0.35))
        span = float(getattr(env.cfg.terrain, 'low_bar_recover_span', 0.06))
        rel = env.low_bar_relative_state()
        past = (rel[:, 0] < -distance).float()
        # root_states also holds the low-bar actors, so slice the robots out.
        base_z = env.root_states[:env.num_envs, 2]
        target = env.low_bar_clearance - 0.02
        return past * torch.clamp((base_z - target) / max(span, 1e-6), 0.0, 1.0)

    def _reward_low_bar_crossing_speed(self):
        """Keep forward progress under the bar so crawling is not a strategy."""
        env = self.env
        if not getattr(env.cfg.terrain, 'robocon_low_bar', False):
            return torch.zeros(env.num_envs, device=env.device)
        window = float(getattr(env.cfg.terrain, 'low_bar_body_top_window', 0.35))
        minimum = float(getattr(env.cfg.terrain, 'low_bar_crossing_min_vx', 0.25))
        x_rel, _, _ = self._low_bar_window_state()
        inside = (x_rel.abs() <= window).any(dim=1).float()
        return inside * torch.clamp(minimum - env.base_lin_vel[:, 0], min=0.0)

    def _reward_low_bar_alignment(self):
        """Penalize lateral/yaw drift while approaching the gate.

        The lateral component of the body-frame bar vector grows when the
        robot either walks away from the opening centre or yaws away from it.
        Using it here avoids a policy that learns to evade the cross-bar by
        walking around a post.  The term switches off after crossing, so it
        does not fight the normal post-obstacle gait.
        """
        rel = self.env.low_bar_relative_state()
        forward = rel[:, 0]
        lateral = rel[:, 1]
        approach = float(getattr(
            self.env.cfg.terrain, 'low_bar_alignment_approach', 3.0))
        tolerance = float(getattr(
            self.env.cfg.terrain, 'low_bar_alignment_tolerance', 0.15))
        active = (forward > 0.0) & (forward < approach)
        lateral_error = torch.square(
            lateral / max(tolerance, 1e-6))
        if rel.shape[1] >= 7:
            heading_error = torch.atan2(rel[:, 3], rel[:, 4])
            heading_tolerance = float(getattr(
                self.env.cfg.terrain, 'low_bar_heading_tolerance', 0.10))
            heading_weight = float(getattr(
                self.env.cfg.terrain, 'low_bar_heading_weight', 1.0))
            heading_error = heading_weight * torch.square(
                heading_error / max(heading_tolerance, 1e-6))
        else:
            heading_error = torch.zeros_like(lateral_error)
        error = (lateral_error + heading_error).clamp(max=4.0)
        return active.float() * error

    def _reward_low_bar_pass(self):
        """One-shot reward when every robot body crosses inside the gate."""
        return self.env.low_bar_just_passed.float() / self.env.dt

    def _reward_low_bar_success(self):
        """One-shot reward after a clean pass and stable recovery window."""
        return self.env.low_bar_success_this_step.float() / self.env.dt

    def _reward_low_bar_collision(self):
        """One-shot penalty for any physical contact with bar or posts."""
        return self.env.low_bar_collision_this_step.float() / self.env.dt

    def _reward_low_bar_missed_gate(self):
        """One-shot penalty when the robot walks around instead of underneath."""
        return self.env.low_bar_missed_gate_this_step.float() / self.env.dt

    def _reward_tracking_contacts_shaped_force(self):
        # penalize nonzero contact forces during swing phase
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
            
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        # penalize nonzero xy foot velocities during stance phase 
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
            
        return reward / 4

    def _reward_dof_pos(self):
        # Penalize dof positions
        # return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
        
        # Penalize dof position different from nominal    
        reward = torch.square(self.env.dof_pos - self.env.default_dof_pos) # (env_num x 12) tensor
        
        # Penalize hip joint positions more
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.dof_pos_hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.dof_pos_thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.dof_pos_calf_weight

        return torch.sum(reward * weights, dim=1) # (env_num) tensor


    def _reward_dof_pos_stancemode(self):
        # Penalize dof positions
        reward = torch.square(self.env.dof_pos - self.env.default_dof_pos) # (env_num x 12) tensor
        
        # Penalize hip joint positions more
        velocity_commands = self.env.commands[:, :3]
        weights = torch.ones(self.env.num_envs ,12, device=self.env.device) # (12) tensor
        hip_joint_indices =   [0, 3, 6, 9]
        thigh_joint_indices = [1, 4, 7, 10]
        calf_joint_indices =  [2, 5, 8, 11]
        weights[:,hip_joint_indices] = self.env.cfg.rewards.hip_weight
        weights[:,thigh_joint_indices] = self.env.cfg.rewards.thigh_weight
        weights[:,calf_joint_indices] = self.env.cfg.rewards.calf_weight

        if(self.env.cfg.rewards.use_adaptive_stancemode):
            y_cmd_intesity = torch.abs(velocity_commands[:,1]/self.env.cfg.commands.limit_vel_y[1])
            yaw_cmd_intensity = torch.abs(velocity_commands[:,2]/self.env.cfg.commands.limit_vel_yaw[1])
            cmd_intesity = torch.max(y_cmd_intesity,yaw_cmd_intensity)
            cmd_intesity = torch.min(torch.ones_like(cmd_intesity)*0.5, cmd_intesity)
            weights[:,hip_joint_indices] = weights[:,hip_joint_indices]*(torch.ones_like(cmd_intesity)-cmd_intesity).view(-1,1)
        # todo print weights/reward
        reward = torch.sum(reward * weights, dim=1) # (env_num) tensor

        # use stancemode alternative reward if command vector is below a certain threshold
        if self.env.cfg.commands.train_standing_still:
            velocity_commands = self.env.commands[:, :3] # (env_num x 3) tensor, x/y/yaw velocity commands
            cmd_norm = torch.norm(velocity_commands, dim=1)
            env_stancemode = cmd_norm < 0.01 #0.1
            
            # penalize dof positions more if in stancemode
            reward = torch.where(env_stancemode, reward*self.env.cfg.rewards.stancemode_multiplier, reward)
            
        return reward


    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.env.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_feet_slip(self):
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        self.env.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:2], dim=2).view(self.env.num_envs, -1)) # squared norm of xy velocities, results in 2D tensor
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1) # sum over the four feet, results in 1D tensor
        return rew_slip
    
    def _reward_feet_air_time(self):
        # print("CAREFUL: this reward may be bugged!")
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1. # 1N contact force threshold
        contact_filt = torch.logical_or(contact, self.env.last_contacts_air_time) # shows whether contact was present in the last two steps
       
        transition_reset_condition = contact_filt ^ self.env.last_contacts_filt_air_time # shows whether a transition from contact to no contact or vice versa happened
        # print("The following env 0 feet just transitioned: ", transition_reset_condition[0, :])

        # reset the touchdown and takeoff timers if a transition happens
        self.env.feet_touchdown_air_time = torch.where(transition_reset_condition, torch.zeros_like(self.env.feet_touchdown_air_time), self.env.feet_touchdown_air_time)
        self.env.feet_takeoff_air_time = torch.where(transition_reset_condition, torch.zeros_like(self.env.feet_takeoff_air_time), self.env.feet_takeoff_air_time)

        # increment the timers
        self.env.feet_touchdown_air_time = torch.where(contact_filt, self.env.feet_touchdown_air_time + self.env.dt, self.env.feet_touchdown_air_time)
        self.env.feet_takeoff_air_time = torch.where(~contact_filt, self.env.feet_takeoff_air_time + self.env.dt, self.env.feet_takeoff_air_time)
        
        # save contact states for next iteration
        self.env.last_contacts_air_time = contact
        self.env.last_contacts_filt_air_time = contact_filt

        velocity_commands = self.env.commands[:, :3] # (env_num x 3) tensor, x/y/yaw velocity commands
        cmd_norm = torch.norm(velocity_commands, dim=1)

        # use stancemode alternative reward if command vector is below a certain threshold
        if self.env.cfg.commands.train_standing_still:
            env_stancemode = cmd_norm < 0.001
        else:
            env_stancemode = cmd_norm < 0 # disabled

        # calculate the desired stance time
        air_time_period = torch.ones_like(contact_filt)*self.env.cfg.rewards.feet_air_time_period      

        if(self.env.cfg.rewards.use_adaptive_period):
            cmd_int_x = torch.abs(velocity_commands[:,0])/self.env.cfg.commands.limit_vel_x[0]
            cmd_int_y = torch.abs(velocity_commands[:,1])/self.env.cfg.commands.limit_vel_x[1]
            cmd_intesity = torch.max(cmd_int_x,cmd_int_y)
            cmd_intesity = torch.min(torch.ones_like(cmd_intesity)*0.5, cmd_intesity)
            air_time_period = air_time_period*((torch.ones_like(cmd_intesity)-cmd_intesity).view(-1,1))

        # set reward to 0 if feet remain in one state for too long


        reward_condition_contact = self.env.feet_touchdown_air_time < air_time_period #(desired_stance_time.unsqueeze(1) + 0.05)
        reward_condition_nocontact = self.env.feet_takeoff_air_time <  air_time_period
       

        rew_air_time_contact = torch.where(reward_condition_contact, torch.min(self.env.feet_touchdown_air_time, air_time_period), torch.tensor(0.0, device=self.env.device))
        rew_air_time_nocontact = torch.where(reward_condition_nocontact, torch.min(self.env.feet_takeoff_air_time, air_time_period), torch.tensor(0.0, device=self.env.device))
        # print("The contact reward of env 0 feet are: ", rew_air_time_contact[0, :])
        # print("The nocontact reward of env 0 feet are: ", rew_air_time_nocontact[0, :])

        # calculate the final reward
        rew_air_time_foot_normal = torch.where(contact_filt, rew_air_time_contact, rew_air_time_nocontact)
        rew_air_time_foot_stancemode = torch.clip(self.env.feet_touchdown_air_time - self.env.feet_takeoff_air_time, min=-air_time_period, max=air_time_period)
        # print("The normal reward of env 0 feet are: ", rew_air_time_foot_normal[0, :])
        # print("The stancemode reward of env 0 feet are: ", rew_air_time_foot_stancemode[0, :])
        
        # enforce trotting gait
        # Joint order: FL FR RL RR
        # rew_air_time_foot_normal[:,0] = torch.where(torch.logical_xor(contact_filt[:,0],contact_filt[:,1]),rew_air_time_foot_normal[:,0],rew_air_time_foot_normal[:,0]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,1] = torch.where(torch.logical_xor(contact_filt[:,0],contact_filt[:,1]),rew_air_time_foot_normal[:,1],rew_air_time_foot_normal[:,1]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,2] = torch.where(torch.logical_xor(contact_filt[:,2],contact_filt[:,3]),rew_air_time_foot_normal[:,2],rew_air_time_foot_normal[:,2]*self.env.cfg.rewards.contact_condition_scale)
        # rew_air_time_foot_normal[:,3] = torch.where(torch.logical_xor(contact_filt[:,2],contact_filt[:,3]),rew_air_time_foot_normal[:,3],rew_air_time_foot_normal[:,3]*self.env.cfg.rewards.contact_condition_scale)
        
        if(self.env.cfg.rewards.use_adaptive_period): #normalize reward => small period would give small reward else
            rew_air_time_foot_normal *= torch.square((torch.divide(torch.ones_like(cmd_intesity),torch.ones_like(cmd_intesity)-cmd_intesity)).view(-1,1))


        rew_air_time_foot = torch.where(env_stancemode.unsqueeze(1), rew_air_time_foot_stancemode, rew_air_time_foot_normal)
        # print("The final reward of env 0 feet are: ", rew_air_time_foot[0, :])
        
        rew_air_time = torch.sum(rew_air_time_foot, dim=1)
        # print("The final reward of env 0 is: ", rew_air_time[0])

        
        return rew_air_time
        
    def _reward_feet_air_time_rsl(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts) 
        self.env.last_contacts = contact
        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt
        rew_airTime = torch.sum((self.env.feet_air_time - self.env.cfg.rewards.feet_air_time_rsl_period) * first_contact, dim=1) # reward only on first contact with the ground
        # rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.01 #no reward for zero command
        self.env.feet_air_time *= ~contact_filt

        if self.env.cfg.rewards.feet_air_time_rsl_curriculum:
            rew_airTime *= self.env.reward_curriculum_factor

        return rew_airTime
        


    def _reward_feet_contact_vel(self):
        reference_heights = 0
        near_ground = self.env.foot_positions[:, :, 2] - reference_heights < 0.03
        foot_velocities = torch.square(torch.norm(self.env.foot_velocities[:, :, 0:3], dim=2).view(self.env.num_envs, -1))
        rew_contact_vel = torch.sum(near_ground * foot_velocities, dim=1)
        return rew_contact_vel

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.env.contact_forces[:, self.env.feet_indices, :],
                                     dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) # reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states) # penalize height only when contact is not desired (swing phase)
        rew_foot_clearance = torch.sum(rew_foot_clearance, dim=1)
        
        return rew_foot_clearance
    
    def _reward_feet_clearance_ji22(self):
        # get filtered contact states
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts_feet_clearance_ji22) ##why or? shouldn't it be an and ?
        self.env.last_contacts_feet_clearance_ji22 = contact
        
        target_height = self.env.cfg.rewards.feet_clearance_ji22_target + 0.02 # target + offset for foot radius 2cm
        
        rew_foot_clearance = torch.square(torch.where(self.env.feet_height < target_height, self.env.feet_height, torch.full_like(self.env.feet_height, fill_value=target_height)))/target_height**2 
        rew_foot_clearance = torch.where(contact_filt, torch.zeros_like(rew_foot_clearance), rew_foot_clearance) # set reward to 0 if foot is in contact
        
        
        
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_thigh_angle(self):
        """ Penalize knee height (or thigh joint angle) to avoid collisions with backpack
        """

        max_thigh_angle = self.env.cfg.rewards.max_thigh_angle

        weights = torch.zeros(12, device=self.env.device) # (12) tensor
        thigh_joint_indices = [1, 4, 7, 10]
        weights[thigh_joint_indices] = 1.0

        #thigh_joint_indices =  torch.tensor([1, 4, 7, 10],device=self.env.device)
        
        rew_thigh_angle = torch.clamp(self.env.dof_pos - max_thigh_angle, min=0, max=None)*weights
    
        #print(self.env.dof_pos[:,1])

        return torch.sum(rew_thigh_angle, dim=1)
        
        

    def _reward_feet_impact_vel(self):
        prev_foot_velocities = self.env.prev_foot_velocities[:, :, 2].view(self.env.num_envs, -1)
        contact_states = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1) > 1.0 ##why: non need to filter? 

        rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))

        return torch.sum(rew_foot_impact_vel, dim=1)


    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        # Actually nope, also do pitch control!
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
