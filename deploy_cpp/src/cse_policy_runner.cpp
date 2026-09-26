/**
 * @file cse_policy_runner.cpp
 * @brief CSE policy runner implementation.
 */

#include "cse_policy_runner.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace deploy {
namespace {

torch::Tensor tensor_from_array(const std::array<float, NUM_JOINTS> &arr,
                                torch::Device device) {
  return torch::from_blob(const_cast<float *>(arr.data()), {1, NUM_JOINTS},
                          torch::kFloat32)
      .clone()
      .to(device);
}

template <size_t N>
torch::Tensor tensor_from_float_array(const std::array<float, N> &arr,
                                      torch::Device device) {
  return torch::from_blob(const_cast<float *>(arr.data()),
                          {1, static_cast<long>(N)}, torch::kFloat32)
      .clone()
      .to(device);
}

void require_shape(const torch::Tensor &tensor,
                   const std::vector<int64_t> &shape,
                   const std::string &name) {
  if (tensor.dim() != static_cast<int64_t>(shape.size())) {
    throw std::runtime_error(name + " has wrong rank");
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (tensor.size(static_cast<int64_t>(i)) != shape[i]) {
      throw std::runtime_error(name + " has wrong shape");
    }
  }
}

} // namespace

CSEPolicyRunner::CSEPolicyRunner(const RobotRuntimeConfig &config)
    : config_(config), device_(config.device) {
  std::cout << "[CSEPolicyRunner] Loading adaptation module: "
            << config_.adaptation_module_path << std::endl;
  std::cout << "[CSEPolicyRunner] Loading body: " << config_.body_path
            << std::endl;

  adaptation_module_ = torch::jit::load(config_.adaptation_module_path, device_);
  body_ = torch::jit::load(config_.body_path, device_);
  adaptation_module_.eval();
  body_.eval();

  policy_history_ =
      torch::zeros({CSE_HISTORY_LENGTH, CSE_POLICY_OBS_PER_STEP},
                   torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  estimator_history_ =
      torch::zeros({CSE_HISTORY_LENGTH, CSE_ESTIMATOR_OBS_PER_STEP},
                   torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  last_actions_ =
      torch::zeros({1, NUM_ACTIONS},
                   torch::TensorOptions().dtype(torch::kFloat32).device(device_));
  policy_dof_pos_ = tensor_from_array(config_.policy_dof_pos, device_);

  validate_model_shapes();
}

void CSEPolicyRunner::validate_model_shapes() {
  torch::NoGradGuard no_grad;
  auto est = torch::zeros({1, CSE_ESTIMATOR_OBS},
                          torch::TensorOptions().dtype(torch::kFloat32)
                              .device(device_));
  auto latent = adaptation_module_.forward({est}).toTensor();
  require_shape(latent, {1, CSE_LATENT_DIM}, "adaptation_module output");

  auto body_in = torch::zeros({1, CSE_BODY_INPUT_DIM},
                              torch::TensorOptions().dtype(torch::kFloat32)
                                  .device(device_));
  auto action = body_.forward({body_in}).toTensor();
  require_shape(action, {1, NUM_ACTIONS}, "body output");
  std::cout << "[CSEPolicyRunner] Shape check OK: estimator "
            << CSE_ESTIMATOR_OBS << " -> latent " << CSE_LATENT_DIM
            << ", body " << CSE_BODY_INPUT_DIM << " -> " << NUM_ACTIONS
            << std::endl;
}

void CSEPolicyRunner::reset() {
  policy_history_.zero_();
  estimator_history_.zero_();
  last_actions_.zero_();
  infer_count_ = 0;
}

torch::Tensor CSEPolicyRunner::compute_policy_obs(
    const std::array<float, 3> &commands,
    const std::array<float, 3> &projected_gravity,
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel,
    const std::array<float, NUM_HEIGHT_POINTS> &height_distances) {
  std::array<float, 3> cmd = commands;
  const float cmd_xy_norm = std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1]);
  if (cmd_xy_norm < config_.cmd_deadband) {
    cmd[0] = 0.0f;
    cmd[1] = 0.0f;
  }

  auto t_cmd = tensor_from_float_array(cmd, device_);
  auto t_gravity = tensor_from_float_array(projected_gravity, device_);
  auto t_pos = tensor_from_array(dof_pos, device_);
  auto t_vel = tensor_from_array(dof_vel, device_);
  auto t_height = tensor_from_float_array(height_distances, device_);

  std::array<float, 3> command_scale = {
      config_.command_lin_vel_scale, config_.command_lin_vel_scale,
      config_.command_yaw_scale};
  auto t_cmd_scale = tensor_from_float_array(command_scale, device_);

  auto height_obs =
      torch::clamp(config_.height_bias - t_height, -1.0f, 1.0f) *
      config_.height_scale;

  auto obs = torch::cat(
      {t_cmd * t_cmd_scale,
       t_gravity * config_.projected_gravity_scale,
       (t_pos - policy_dof_pos_) * config_.dof_pos_scale,
       t_vel * config_.dof_vel_scale,
       last_actions_,
       height_obs},
      1);
  return torch::clamp(obs, -config_.clip_obs, config_.clip_obs);
}

torch::Tensor CSEPolicyRunner::compute_estimator_obs(
    const std::array<float, 3> &projected_gravity,
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel,
    const std::array<float, NUM_HEIGHT_POINTS> &height_distances) {
  auto t_gravity = tensor_from_float_array(projected_gravity, device_);
  auto t_pos = tensor_from_array(dof_pos, device_);
  auto t_vel = tensor_from_array(dof_vel, device_);
  auto t_height = tensor_from_float_array(height_distances, device_);
  auto height_obs =
      torch::clamp(config_.height_bias - t_height, -1.0f, 1.0f) *
      config_.height_scale;

  auto obs = torch::cat(
      {t_gravity * config_.projected_gravity_scale,
       (t_pos - policy_dof_pos_) * config_.dof_pos_scale,
       t_vel * config_.dof_vel_scale,
       last_actions_,
       height_obs},
      1);
  return torch::clamp(obs, -config_.clip_obs, config_.clip_obs);
}

void CSEPolicyRunner::update_history(const torch::Tensor &policy_obs,
                                     const torch::Tensor &estimator_obs) {
  policy_history_ = torch::cat(
      {policy_history_.index(
           {torch::indexing::Slice(1, torch::indexing::None),
            torch::indexing::Slice()}),
       policy_obs.reshape({1, CSE_POLICY_OBS_PER_STEP})},
      0);
  estimator_history_ = torch::cat(
      {estimator_history_.index(
           {torch::indexing::Slice(1, torch::indexing::None),
            torch::indexing::Slice()}),
       estimator_obs.reshape({1, CSE_ESTIMATOR_OBS_PER_STEP})},
      0);
}

std::array<float, NUM_JOINTS> CSEPolicyRunner::target_from_actions(
    const std::array<float, NUM_ACTIONS> &actions) const {
  std::array<float, NUM_JOINTS> target{};
  for (int i = 0; i < NUM_JOINTS; ++i) {
    target[i] = config_.policy_dof_pos[i] + actions[i] * config_.action_scale;
    target[i] =
        std::clamp(target[i], config_.joint_pos_lower[i],
                   config_.joint_pos_upper[i]);
  }
  return target;
}

void CSEPolicyRunner::step(
    const std::array<float, 3> &commands,
    const std::array<float, 3> &projected_gravity,
    const std::array<float, NUM_JOINTS> &dof_pos,
    const std::array<float, NUM_JOINTS> &dof_vel,
    const std::array<float, NUM_HEIGHT_POINTS> &height_distances,
    std::array<float, NUM_JOINTS> &target_dof_pos,
    std::array<float, NUM_ACTIONS> &actions) {
  torch::NoGradGuard no_grad;

  auto policy_obs =
      compute_policy_obs(commands, projected_gravity, dof_pos, dof_vel,
                         height_distances);
  auto estimator_obs =
      compute_estimator_obs(projected_gravity, dof_pos, dof_vel,
                            height_distances);
  update_history(policy_obs, estimator_obs);

  auto policy_hist = policy_history_.reshape({1, CSE_POLICY_OBS});
  auto estimator_hist = estimator_history_.reshape({1, CSE_ESTIMATOR_OBS});
  auto latent = adaptation_module_.forward({estimator_hist}).toTensor();
  auto body_input = torch::cat({policy_hist, latent}, 1);
  auto action_tensor = body_.forward({body_input}).toTensor();
  action_tensor =
      torch::clamp(action_tensor, -config_.clip_actions, config_.clip_actions);
  last_actions_ = action_tensor.clone();

  auto action_cpu = action_tensor.cpu().contiguous();
  const float *ptr = action_cpu.data_ptr<float>();
  std::copy(ptr, ptr + NUM_ACTIONS, actions.begin());
  target_dof_pos = target_from_actions(actions);

  ++infer_count_;
  if (config_.debug_print_policy &&
      infer_count_ % static_cast<uint64_t>(config_.debug_print_interval) ==
          0U) {
    auto latent_cpu = latent.cpu().contiguous();
    auto height_cpu = policy_obs.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(CSE_POLICY_OBS_PER_STEP - NUM_HEIGHT_POINTS,
                                CSE_POLICY_OBS_PER_STEP)})
                          .cpu()
                          .contiguous();
    std::cout << "\n[CSEPolicyRunner] step=" << infer_count_
              << " action[0:3]=[" << actions[0] << ", " << actions[1]
              << ", " << actions[2] << "] latent[0]="
              << latent_cpu.data_ptr<float>()[0] << " height_obs[0]="
              << height_cpu.data_ptr<float>()[0] << std::endl;
  }
}

} // namespace deploy
