/**
 * @file cse_policy_runner.h
 * @brief CSE two-stage policy inference with observation history.
 */
#pragma once

#include <array>
#include <string>

#include <torch/script.h>
#include <torch/torch.h>

#include "robot_runtime_config.h"

namespace deploy {

class CSEPolicyRunner {
public:
  explicit CSEPolicyRunner(const RobotRuntimeConfig &config);

  void reset();

  torch::Tensor compute_policy_obs(
      const std::array<float, 3> &commands,
      const std::array<float, 3> &projected_gravity,
      const std::array<float, NUM_JOINTS> &dof_pos,
      const std::array<float, NUM_JOINTS> &dof_vel,
      const std::array<float, NUM_HEIGHT_POINTS> &height_distances);

  torch::Tensor compute_estimator_obs(
      const std::array<float, 3> &projected_gravity,
      const std::array<float, NUM_JOINTS> &dof_pos,
      const std::array<float, NUM_JOINTS> &dof_vel,
      const std::array<float, NUM_HEIGHT_POINTS> &height_distances);

  void step(const std::array<float, 3> &commands,
            const std::array<float, 3> &projected_gravity,
            const std::array<float, NUM_JOINTS> &dof_pos,
            const std::array<float, NUM_JOINTS> &dof_vel,
            const std::array<float, NUM_HEIGHT_POINTS> &height_distances,
            std::array<float, NUM_JOINTS> &target_dof_pos,
            std::array<float, NUM_ACTIONS> &actions);

private:
  void update_history(const torch::Tensor &policy_obs,
                      const torch::Tensor &estimator_obs);
  std::array<float, NUM_JOINTS>
  target_from_actions(const std::array<float, NUM_ACTIONS> &actions) const;
  void validate_model_shapes();

  RobotRuntimeConfig config_;
  torch::Device device_;
  torch::jit::script::Module adaptation_module_;
  torch::jit::script::Module body_;

  torch::Tensor policy_history_;    // (10, 119), oldest -> newest
  torch::Tensor estimator_history_; // (10, 116), oldest -> newest
  torch::Tensor last_actions_;      // (1, 12)
  torch::Tensor policy_dof_pos_;    // (1, 12)

  uint64_t infer_count_ = 0;
};

} // namespace deploy
