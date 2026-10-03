import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from robodog_gym_learn.ppo_cse import ActorCritic
from robodog_gym_learn.ppo_cse import RolloutStorage
from robodog_gym_learn.ppo_cse import caches



class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, cfg_ppo, device='cpu'):

        self.device = device

        self.cfg_ppo = cfg_ppo

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        # The actor and critic have disjoint networks. Keep their optimizers and
        # gradient clipping separate so a return outlier in the critic cannot
        # shrink the actor update through one global gradient norm.
        self.actor_parameters = list(self.actor_critic.actor_body.parameters()) + \
                                list(self.actor_critic.adaptation_module.parameters()) + \
                                [self.actor_critic.std]
        self.critic_parameters = list(self.actor_critic.critic_body.parameters())
        self.optimizer = optim.Adam(self.actor_parameters, lr=self.cfg_ppo.algorithm.learning_rate)
        critic_learning_rate = getattr(
            self.cfg_ppo.algorithm, 'critic_learning_rate', self.cfg_ppo.algorithm.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=critic_learning_rate)
        self.adaptation_module_optimizer = optim.Adam(
            self.actor_critic.adaptation_module.parameters(),
            lr=self.cfg_ppo.algorithm.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=self.cfg_ppo.algorithm.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = self.cfg_ppo.algorithm.learning_rate
        self.last_kl_mean = 0.0
        self.last_kl_max = 0.0
        self.skipped_policy_updates = 0
        self.invalid_policy_updates = 0
        self.invalid_critic_updates = 0
        self.actor_grad_norm = 0.0
        self.critic_grad_norm = 0.0
        self.return_mean = 0.0
        self.return_std = 0.0
        self.return_abs_max = 0.0
        self.action_saturation_fraction = 0.0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, estimator_obs_shape, privileged_obs_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, estimator_obs_shape, privileged_obs_shape,
                                      action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, estimator_obs, privileged_obs):
        # Important to detech and clone tensors obtained from environment step function
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs,estimator_obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = self.transition.observations
        self.transition.estimator_observations = estimator_obs.detach().clone()
        self.transition.privileged_observations = privileged_obs.detach().clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Important to detech and clone tensors obtained from environment step function 
        # (if not modified by non-in-place operations)
        self.transition.rewards = rewards.detach().clone()
        self.transition.dones = dones.detach().clone()
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.cfg_ppo.algorithm.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, self.cfg_ppo.algorithm.gamma, self.cfg_ppo.algorithm.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        kl_values = []
        skipped_policy_updates = 0
        invalid_policy_updates = 0
        invalid_critic_updates = 0
        actor_grad_norms = []
        critic_grad_norms = []

        # Record target and action-tail statistics before storage is cleared.
        # These make reward drift and action clipping visible in the local log.
        with torch.inference_mode():
            self.return_mean = float(self.storage.returns.mean().item())
            self.return_std = float(self.storage.returns.std(unbiased=False).item())
            self.return_abs_max = float(self.storage.returns.abs().max().item())
            action_clip = getattr(self.cfg_ppo.algorithm, 'action_clip', None)
            if action_clip is not None and action_clip > 0:
                self.action_saturation_fraction = float(
                    (self.storage.actions.abs() >= action_clip).float().mean().item())
            else:
                self.action_saturation_fraction = 0.0
        generator = self.storage.mini_batch_generator(self.cfg_ppo.algorithm.num_mini_batches, self.cfg_ppo.algorithm.num_learning_epochs)
        for obs_batch, critic_obs_batch, estimator_obs_batch, privileged_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_batch, estimator_obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Measure KL for every schedule. In fixed-rate fine tuning it acts
            # as a hard trust-region guard: once the current policy has moved too
            # far from the rollout policy, the remaining destructive update is
            # skipped and the next rollout starts from the bounded policy.
            skip_policy_update = False
            desired_kl = self.cfg_ppo.algorithm.desired_kl
            if desired_kl is not None:
                with torch.inference_mode():
                    safe_sigma = torch.clamp(sigma_batch, min=1.e-6)
                    safe_old_sigma = torch.clamp(old_sigma_batch, min=1.e-6)
                    kl = torch.sum(
                        torch.log(safe_sigma / safe_old_sigma) + (
                                torch.square(safe_old_sigma) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(safe_sigma)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    kl_value = float(kl_mean.item())
                    kl_values.append(kl_value)

                    if self.cfg_ppo.algorithm.schedule == 'adaptive' and np.isfinite(kl_value):
                        if kl_value > desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / self.cfg_ppo.algorithm.lr_adaptive_schedule_decay)
                        elif 0.0 < kl_value < desired_kl / 2.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                    hard_kl_limit = getattr(self.cfg_ppo.algorithm, 'hard_kl_limit', None)
                    if hard_kl_limit is not None and (not np.isfinite(kl_value) or kl_value > hard_kl_limit):
                        skip_policy_update = True

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.cfg_ppo.algorithm.clip_param,
                                                                               1.0 + self.cfg_ppo.algorithm.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss. Keep MSE as the diagnostic metric so it is
            # comparable with older runs, while optionally using Huber loss for
            # the critic update to bound the influence of rare return outliers.
            if self.cfg_ppo.algorithm.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-self.cfg_ppo.algorithm.clip_param,
                                                                          self.cfg_ppo.algorithm.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
                value_error = value_batch - returns_batch
                value_error_clipped = value_clipped - returns_batch
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
                value_error = value_batch - returns_batch
                value_error_clipped = None

            huber_delta = getattr(self.cfg_ppo.algorithm, 'value_huber_delta', None)
            if huber_delta is not None and huber_delta > 0:
                def huber(error):
                    abs_error = error.abs()
                    return torch.where(
                        abs_error <= huber_delta,
                        0.5 * error.pow(2),
                        huber_delta * (abs_error - 0.5 * huber_delta))

                value_objective = huber(value_error)
                if value_error_clipped is not None:
                    value_objective = torch.max(value_objective, huber(value_error_clipped))
                value_objective = value_objective.mean()
            else:
                value_objective = value_loss

            actor_loss = surrogate_loss - self.cfg_ppo.algorithm.entropy_coef * entropy_batch.mean()
            critic_loss = self.cfg_ppo.algorithm.value_loss_coef * value_objective

            # Actor gradient step. The KL guard only applies to the policy; the
            # critic is still allowed to fit the rollout targets.
            if not bool(torch.isfinite(actor_loss).item()):
                invalid_policy_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
            elif skip_policy_update:
                skipped_policy_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.actor_parameters, self.cfg_ppo.algorithm.max_grad_norm)
                if bool(torch.isfinite(grad_norm).item()):
                    actor_grad_norms.append(float(grad_norm.item()))
                    self.optimizer.step()
                else:
                    invalid_policy_updates += 1
                    self.optimizer.zero_grad(set_to_none=True)

            # Critic gradient step with its own learning rate and clipping.
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_update_loss_threshold = getattr(
                self.cfg_ppo.algorithm, 'critic_update_loss_threshold', None)
            critic_loss_is_finite = bool(torch.isfinite(critic_loss).item())
            critic_loss_is_bounded = (
                critic_update_loss_threshold is None or
                float(value_loss.detach().item()) <= float(critic_update_loss_threshold))
            if critic_loss_is_finite and critic_loss_is_bounded:
                critic_loss.backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    getattr(self.cfg_ppo.algorithm, 'critic_max_grad_norm',
                            self.cfg_ppo.algorithm.max_grad_norm))
                if bool(torch.isfinite(critic_grad_norm).item()):
                    critic_grad_norms.append(float(critic_grad_norm.item()))
                    self.critic_optimizer.step()
                else:
                    invalid_critic_updates += 1
                    self.critic_optimizer.zero_grad(set_to_none=True)
            else:
                invalid_critic_updates += 1

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step

            for epoch in range(self.cfg_ppo.algorithm.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.adaptation_module(estimator_obs_batch)
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                # print("The adaptation module has target: ", adaptation_target)
                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long)
                if self.cfg_ppo.algorithm.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])



                self.adaptation_module_optimizer.zero_grad(set_to_none=True)
                if bool(torch.isfinite(adaptation_loss).item()):
                    adaptation_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.adaptation_module.parameters(),
                                             self.cfg_ppo.algorithm.max_grad_norm)
                    self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()

        num_updates = self.cfg_ppo.algorithm.num_learning_epochs * self.cfg_ppo.algorithm.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * self.cfg_ppo.algorithm.num_adaptation_module_substeps)
        self.storage.clear()
        finite_kl_values = [value for value in kl_values if np.isfinite(value)]
        self.last_kl_mean = float(np.mean(finite_kl_values)) if finite_kl_values else 0.0
        self.last_kl_max = float(np.max(finite_kl_values)) if finite_kl_values else 0.0
        self.skipped_policy_updates = skipped_policy_updates
        self.invalid_policy_updates = invalid_policy_updates
        self.invalid_critic_updates = invalid_critic_updates
        self.actor_grad_norm = float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0
        self.critic_grad_norm = float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
