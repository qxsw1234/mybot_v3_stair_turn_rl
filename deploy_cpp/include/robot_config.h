/**
 * @file robot_config.h
 * @brief Compile-time dimensions for mybot CSE deployment.
 */
#pragma once

namespace deploy {

constexpr int NUM_JOINTS = 12;
constexpr int NUM_ACTIONS = 12;

constexpr int NUM_HEIGHT_POINTS_X = 11;
constexpr int NUM_HEIGHT_POINTS_Y = 7;
constexpr int NUM_HEIGHT_POINTS =
    NUM_HEIGHT_POINTS_X * NUM_HEIGHT_POINTS_Y;

constexpr int CSE_POLICY_OBS_PER_STEP =
    3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_ACTIONS + NUM_HEIGHT_POINTS;
constexpr int CSE_ESTIMATOR_OBS_PER_STEP =
    3 + NUM_JOINTS + NUM_JOINTS + NUM_ACTIONS + NUM_HEIGHT_POINTS;
constexpr int CSE_HISTORY_LENGTH = 10;
constexpr int CSE_POLICY_OBS = CSE_POLICY_OBS_PER_STEP * CSE_HISTORY_LENGTH;
constexpr int CSE_ESTIMATOR_OBS =
    CSE_ESTIMATOR_OBS_PER_STEP * CSE_HISTORY_LENGTH;
constexpr int CSE_LATENT_DIM = 10;
constexpr int CSE_BODY_INPUT_DIM = CSE_POLICY_OBS + CSE_LATENT_DIM;

} // namespace deploy
