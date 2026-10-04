#!/usr/bin/env python3
"""Run the current MyBot V3 CSE policy directly in MuJoCo.

This is the lightweight sim2sim path. It intentionally does not depend on
ROS2, LibTorch, or the Unitree SDK: Python loads the same TorchScript files
used by deployment, while MuJoCo runs the robot dynamics.

Run from the project root, for example:

    python scripts/sim2sim_mujoco.py --headless --duration 5 --vx 0.2
    python scripts/sim2sim_mujoco.py --terrain-mode stairs --vx 0.2
    python scripts/sim2sim_mujoco.py --terrain-mode three_stairs --interactive

The three_stairs mode contains five parallel lanes: 8, 10, 12, 15, and 20 cm.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


NUM_JOINTS = 12
NUM_HEIGHT_POINTS = 77
HISTORY_LENGTH = 10
POLICY_OBS_PER_STEP = 119
ESTIMATOR_OBS_PER_STEP = 116
MAX_STAIR_STEPS = 8
THREE_STAIR_LANES = 5
THREE_STAIR_STEPS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "deploy_cpp/config/robots/mybot_v3_cse_sim.yaml"
DEFAULT_BODY = PROJECT_ROOT / "deploy_cpp/policy/body_latest.jit"
DEFAULT_ADAPTATION = PROJECT_ROOT / "deploy_cpp/policy/adaptation_module_latest.jit"


class KeyboardController:
    """Keyboard velocity commands plus arrow-key external disturbances."""

    def __init__(self, disturbance_force_n: float, vx: float, vy: float, yaw: float):
        from pynput import keyboard

        self.keyboard = keyboard
        self.keys = {
            key: False
            for key in (
                "w", "s", "a", "d", "q", "e", "u", "o",
                "1", "2", "3", "4", "5", "r",
            )
        }
        self.special_keys = {
            key: False
            for key in (
                keyboard.Key.up,
                keyboard.Key.down,
                keyboard.Key.left,
                keyboard.Key.right,
            )
        }
        self.disturbance_force_n = float(disturbance_force_n)
        self.vx = float(vx)
        self.vy = float(vy)
        self.yaw = float(yaw)
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
        if key in self.special_keys:
            self.special_keys[key] = True
            return
        try:
            name = key.char.lower()
            if name in self.keys:
                self.keys[name] = True
        except AttributeError:
            pass

    def on_release(self, key):
        if key in self.special_keys:
            self.special_keys[key] = False
            return
        try:
            name = key.char.lower()
            if name in self.keys:
                self.keys[name] = False
        except AttributeError:
            pass

    def command(self):
        return np.array(
            [
                self.vx if self.keys["w"] else (-self.vx if self.keys["s"] else 0.0),
                self.vy if self.keys["a"] else (-self.vy if self.keys["d"] else 0.0),
                self.yaw if self.keys["q"] else (-self.yaw if self.keys["e"] else 0.0),
            ],
            dtype=np.float32,
        )

    def disturbance(self) -> np.ndarray:
        """Return a world-frame force while an arrow key is held.

        Arrow keys apply horizontal force to the base body. U/O apply a
        vertical force, which is useful for checking contact recovery without
        changing the commanded velocity.
        """
        force = np.zeros(3, dtype=np.float32)
        magnitude = self.disturbance_force_n
        if self.special_keys[self.keyboard.Key.up]:
            force[0] += magnitude
        if self.special_keys[self.keyboard.Key.down]:
            force[0] -= magnitude
        if self.special_keys[self.keyboard.Key.left]:
            force[1] += magnitude
        if self.special_keys[self.keyboard.Key.right]:
            force[1] -= magnitude
        if self.keys.get("u", False):
            force[2] += magnitude
        if self.keys.get("o", False):
            force[2] -= magnitude
        return force

    def close(self):
        self.listener.stop()

    def consume(self, name: str) -> bool:
        """Consume a one-shot key such as a lane selector or reset."""
        if self.keys.get(name, False):
            self.keys[name] = False
            return True
        return False


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def apply_policy_profile(config: dict, profile: str):
    """Apply control/normalization values from a compatible older policy."""
    if profile in ("config", "current"):
        return
    if profile == "may2026":
        # The May 2026 mybot_v3 CSE run used 40/40/80 Nm/rad gains and
        # normalization.clip_actions=20.  Keep this opt-in so the current
        # 059000 stair-turn policy remains unchanged by default.
        config["kp_joint"] = [40.0, 40.0, 80.0] * 4
        config["kd_joint"] = [1.0, 1.0, 2.0] * 4
        config["clip_actions"] = 20.0
        return
    raise ValueError(f"unknown policy profile: {profile}")


def quat_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0, 0, -1] into the body frame."""
    w, x, y, z = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


class MybotV3Sim:
    def __init__(self, config: dict, body_path: Path, adaptation_path: Path,
                 terrain_mode: str, package_root: Path):
        self.config = config
        self.terrain_mode = terrain_mode.lower()
        if self.terrain_mode not in ("flat", "stairs", "stair", "three_stairs"):
            raise ValueError("terrain mode must be flat, stairs, or three_stairs")

        if int(config.get("num_of_dofs", NUM_JOINTS)) != NUM_JOINTS:
            raise RuntimeError("this CSE policy expects exactly 12 DoFs")
        if int(config["decimation"]) <= 0:
            raise RuntimeError("decimation must be positive")
        expected_control_dt = float(config["dt"]) * int(config["decimation"])
        if abs(float(config["control_dt"]) - expected_control_dt) > 1e-6:
            raise RuntimeError(
                f"control_dt {config['control_dt']} != dt * decimation "
                f"({expected_control_dt})"
            )
        for key in (
            "kp_joint",
            "kd_joint",
            "torque_limits",
            "policy_dof_pos",
            "joint_pos_lower",
            "joint_pos_upper",
            "joint_names",
            "joint_controller_names",
        ):
            if len(config[key]) != NUM_JOINTS:
                raise RuntimeError(f"{key} must contain {NUM_JOINTS} values")

        xml_value = Path(config["mujoco_xml_relpath"])
        xml_path = xml_value if xml_value.is_absolute() else package_root / xml_value
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "body"
        )
        if self.base_body_id < 0:
            raise RuntimeError("base body not found in MuJoCo XML: body")
        expected_dt = float(config["dt"])
        if abs(float(self.model.opt.timestep) - expected_dt) > 1e-6:
            raise RuntimeError(
                f"MuJoCo XML timestep {self.model.opt.timestep} != config dt {expected_dt}"
            )

        self.decimation = int(config["decimation"])
        self.control_dt = float(config["control_dt"])
        self.kp = np.asarray(config["kp_joint"], dtype=np.float32) * float(
            config.get("kp_scale", 1.0))
        self.kd = np.asarray(config["kd_joint"], dtype=np.float32) * float(
            config.get("kd_scale", 1.0))
        self.torque_limits = np.asarray(config["torque_limits"], dtype=np.float32)
        self.policy_dof_pos = np.asarray(config["policy_dof_pos"], dtype=np.float32)
        self.action_lag_steps = int(config.get("action_lag_steps", 0))
        if self.action_lag_steps < 0:
            raise RuntimeError("action_lag_steps must be non-negative")
        self.action_lag_buffer = [
            np.zeros(NUM_JOINTS, dtype=np.float32)
            for _ in range(self.action_lag_steps + 1)
        ]
        self.lower = np.asarray(config["joint_pos_lower"], dtype=np.float32)
        self.upper = np.asarray(config["joint_pos_upper"], dtype=np.float32)
        self.joint_names = list(config["joint_names"])
        self.actuator_names = list(config["joint_controller_names"])

        self.joint_qpos_idx = []
        self.joint_qvel_idx = []
        self.actuator_idx = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                raise RuntimeError(f"joint not found in MuJoCo XML: {joint_name}")
            self.joint_qpos_idx.append(int(self.model.jnt_qposadr[joint_id]))
            self.joint_qvel_idx.append(int(self.model.jnt_dofadr[joint_id]))
        for actuator_name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise RuntimeError(
                    f"actuator not found in MuJoCo XML: {actuator_name}"
                )
            self.actuator_idx.append(int(actuator_id))

        # Diagnostic/validation knobs for isolating passive drivetrain
        # mismatch from policy quality.  A value of 1.0 preserves the XML.
        # These are applied only to the twelve policy joints, never the free
        # base DoFs.
        damping_scale = float(config.get("joint_damping_scale", 1.0))
        armature_scale = float(config.get("joint_armature_scale", 1.0))
        for dof_index in self.joint_qvel_idx:
            self.model.dof_damping[dof_index] *= damping_scale
            self.model.dof_armature[dof_index] *= armature_scale

        self.height_x = np.asarray(config["height_points_x"], dtype=np.float32)
        self.height_y = np.asarray(config["height_points_y"], dtype=np.float32)
        if self.height_x.size * self.height_y.size != NUM_HEIGHT_POINTS:
            raise RuntimeError("height grid must contain 77 points")

        self.stair_geom_ids = self._configure_terrain()

        mujoco.mj_resetData(self.model, self.data)
        for index, qpos_index in enumerate(self.joint_qpos_idx):
            self.data.qpos[qpos_index] = self.policy_dof_pos[index]
        mujoco.mj_forward(self.model, self.data)

        self.body = torch.jit.load(str(body_path), map_location="cpu").eval()
        self.adaptation = torch.jit.load(
            str(adaptation_path), map_location="cpu"
        ).eval()
        self.policy_history = torch.zeros(
            (HISTORY_LENGTH, POLICY_OBS_PER_STEP), dtype=torch.float32
        )
        self.estimator_history = torch.zeros(
            (HISTORY_LENGTH, ESTIMATOR_OBS_PER_STEP), dtype=torch.float32
        )
        self.last_actions = torch.zeros((1, NUM_JOINTS), dtype=torch.float32)
        self.infer_count = 0
        self.initial_base = np.asarray(self.data.qpos[:3], dtype=np.float32).copy()
        self.min_base_z = float(self.data.qpos[2])
        self.max_base_z = float(self.data.qpos[2])
        self.max_tilt_rad = 0.0
        self.action_abs_sum = 0.0
        self.action_saturated_count = 0
        self.action_value_count = 0
        self.fell = False

        self._validate_models()

    @staticmethod
    def _tensor_output(value, name: str) -> torch.Tensor:
        """Accept a plain TorchScript tensor and fail clearly on bad exports."""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 1:
            if isinstance(value[0], torch.Tensor):
                return value[0]
        raise RuntimeError(f"{name} must return a single tensor")

    def _validate_models(self):
        """Check the exported JIT interface before the first physics step."""
        with torch.inference_mode():
            estimator_history = torch.zeros(
                (1, HISTORY_LENGTH * ESTIMATOR_OBS_PER_STEP), dtype=torch.float32
            )
            latent = self._tensor_output(
                self.adaptation(estimator_history), "adaptation module"
            )
            if tuple(latent.shape) != (1, 10):
                raise RuntimeError(
                    f"adaptation module output must be [1, 10], got {tuple(latent.shape)}"
                )

            body_input = torch.zeros(
                (1, HISTORY_LENGTH * POLICY_OBS_PER_STEP + 10),
                dtype=torch.float32,
            )
            action = self._tensor_output(self.body(body_input), "body module")
            if tuple(action.shape) != (1, NUM_JOINTS):
                raise RuntimeError(
                    f"body output must be [1, 12], got {tuple(action.shape)}"
                )

    def _configure_terrain(self):
        def geom_id(name: str, required: bool = True) -> int:
            value = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name))
            if value < 0 and required:
                raise RuntimeError(f"stair geom not found in MuJoCo XML: {name}")
            return value

        def disable(value: int):
            if value >= 0:
                self.model.geom_contype[value] = 0
                self.model.geom_conaffinity[value] = 0
                self.model.geom_rgba[value, 3] = 0.0

        def enable(value: int, position, size, friction: float):
            self.model.geom_contype[value] = 1
            self.model.geom_conaffinity[value] = 1
            self.model.geom_rgba[value, 3] = 1.0
            self.model.geom_friction[value, 0] = friction
            self.model.geom_pos[value] = position
            self.model.geom_size[value] = size

        # Disable every generated stair first. This keeps the old one-lane
        # terrain mode compatible while making the new three-lane mode explicit.
        legacy_step_ids = [geom_id(f"sim_stair_{index}") for index in range(MAX_STAIR_STEPS)]
        legacy_top_id = geom_id("sim_stair_top")
        three_step_ids = [
            geom_id(f"sim_three_stair_{index}")
            for index in range(THREE_STAIR_LANES * THREE_STAIR_STEPS)
        ]
        three_top_ids = [
            geom_id(f"sim_three_stair_top_{lane}")
            for lane in range(THREE_STAIR_LANES)
        ]
        for value in legacy_step_ids + [legacy_top_id] + three_step_ids + three_top_ids:
            disable(value)

        if self.terrain_mode == "flat":
            return []

        stair_friction = float(self.config.get("stairs_friction", 1.0))
        if stair_friction <= 0:
            raise RuntimeError("stairs_friction must be positive")

        if self.terrain_mode in ("stairs", "stair"):
            requested_steps = int(self.config.get("stairs_num_steps", MAX_STAIR_STEPS))
            if requested_steps < 1 or requested_steps > MAX_STAIR_STEPS:
                raise RuntimeError(
                    f"stairs_num_steps must be in [1, {MAX_STAIR_STEPS}]"
                )
            step_height = float(self.config.get("stairs_step_height", 0.10))
            tread_depth = float(self.config.get("stairs_tread_depth", 0.30))
            x_start = float(self.config.get("stairs_x_start", 0.55))
            width = float(self.config.get("stairs_width", 2.0))
            top_length = float(self.config.get("stairs_top_length", 0.90))
            if step_height <= 0 or tread_depth <= 0 or width <= 0 or top_length <= 0:
                raise RuntimeError("stair dimensions must be positive")

            active_ids = []
            for index, value in enumerate(legacy_step_ids):
                height = (index + 1) * step_height
                if index < requested_steps:
                    enable(
                        value,
                        (x_start + (index + 0.5) * tread_depth, 0.0, 0.5 * height),
                        (0.5 * tread_depth, 0.5 * width, 0.5 * height),
                        stair_friction,
                    )
                    active_ids.append(value)

            top_height = requested_steps * step_height
            enable(
                legacy_top_id,
                (x_start + requested_steps * tread_depth + 0.5 * top_length, 0.0,
                 0.5 * top_height),
                (0.5 * top_length, 0.5 * width, 0.5 * top_height),
                stair_friction,
            )
            active_ids.append(legacy_top_id)
            return active_ids

        num_steps = int(self.config.get("three_stairs_num_steps", THREE_STAIR_STEPS))
        if num_steps != THREE_STAIR_STEPS:
            raise RuntimeError(f"three_stairs_num_steps must be {THREE_STAIR_STEPS}")
        step_heights = [float(value) for value in self.config.get(
            "three_stairs_step_heights", [0.08, 0.10, 0.12]
        )]
        lane_y = [float(value) for value in self.config.get(
            "three_stairs_lane_y", [-2.6, 0.0, 2.6, 5.2, 7.8]
        )]
        if len(step_heights) != THREE_STAIR_LANES or len(lane_y) != THREE_STAIR_LANES:
            raise RuntimeError("three_stairs_step_heights and three_stairs_lane_y need 5 values")
        if any(value <= 0 for value in step_heights):
            raise RuntimeError("three_stairs_step_heights must be positive")
        tread_depth = float(self.config.get("three_stairs_tread_depth", 0.30))
        x_start = float(self.config.get("three_stairs_x_start", 0.55))
        width = float(self.config.get("three_stairs_width", 2.0))
        top_length = float(self.config.get("three_stairs_top_length", 0.90))
        if tread_depth <= 0 or width <= 0 or top_length <= 0:
            raise RuntimeError("three-stair dimensions must be positive")

        active_ids = []
        for lane in range(THREE_STAIR_LANES):
            step_height = step_heights[lane]
            for index in range(THREE_STAIR_STEPS):
                value = three_step_ids[lane * THREE_STAIR_STEPS + index]
                height = (index + 1) * step_height
                enable(
                    value,
                    (x_start + (index + 0.5) * tread_depth, lane_y[lane], 0.5 * height),
                    (0.5 * tread_depth, 0.5 * width, 0.5 * height),
                    stair_friction,
                )
                active_ids.append(value)

            top_height = THREE_STAIR_STEPS * step_height
            top_id = three_top_ids[lane]
            enable(
                top_id,
                (x_start + THREE_STAIR_STEPS * tread_depth + 0.5 * top_length,
                 lane_y[lane], 0.5 * top_height),
                (0.5 * top_length, 0.5 * width, 0.5 * top_height),
                stair_friction,
            )
            active_ids.append(top_id)

        return active_ids

    def joint_pos(self) -> np.ndarray:
        return np.asarray(
            [self.data.qpos[index] for index in self.joint_qpos_idx], dtype=np.float32
        )

    def joint_vel(self) -> np.ndarray:
        return np.asarray(
            [self.data.qvel[index] for index in self.joint_qvel_idx], dtype=np.float32
        )

    def teleport(self, x: float, y: float, yaw: float = 0.0):
        """Reset the robot to its standing pose at a chosen test lane."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (x, y, 0.35)
        half = 0.5 * yaw
        self.data.qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
        for index, qpos_index in enumerate(self.joint_qpos_idx):
            self.data.qpos[qpos_index] = self.policy_dof_pos[index]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.policy_history.zero_()
        self.estimator_history.zero_()
        self.last_actions.zero_()
        self.action_lag_buffer = [
            np.zeros(NUM_JOINTS, dtype=np.float32)
            for _ in range(self.action_lag_steps + 1)
        ]
        self.infer_count = 0
        self.initial_base = np.asarray(self.data.qpos[:3], dtype=np.float32).copy()
        self.min_base_z = float(self.data.qpos[2])
        self.max_base_z = float(self.data.qpos[2])
        self.max_tilt_rad = 0.0
        self.action_abs_sum = 0.0
        self.action_saturated_count = 0
        self.action_value_count = 0
        self.fell = False

    def _ground_height_at(self, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        gz = np.zeros_like(wx, dtype=np.float32)
        for geom_id in self.stair_geom_ids:
            center = self.model.geom_pos[geom_id]
            half_size = self.model.geom_size[geom_id]
            inside = (
                (np.abs(wx - center[0]) <= half_size[0])
                & (np.abs(wy - center[1]) <= half_size[1])
            )
            top_z = np.float32(center[2] + half_size[2])
            gz = np.where(inside, np.maximum(gz, top_z), gz)
        return gz

    def height_distances(self) -> np.ndarray:
        base_x, base_y, base_z = self.data.qpos[:3]
        w, x, y, z = self.data.qpos[3:7]
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cy, sy = np.cos(yaw), np.sin(yaw)
        grid_x, grid_y = np.meshgrid(self.height_x, self.height_y, indexing="ij")
        world_x = base_x + cy * grid_x - sy * grid_y
        world_y = base_y + sy * grid_x + cy * grid_y

        # Match training: heightmap grid + min-of-3 sampling (horiz_scale=0.1, border=25)
        HS = 0.1
        BORDER = 25.0
        ix = np.floor((world_x + BORDER) / HS).astype(np.int64)
        iy = np.floor((world_y + BORDER) / HS).astype(np.int64)

        def _h_at_grid(gx, gy):
            wx = gx * HS - BORDER
            wy = gy * HS - BORDER
            return self._ground_height_at(wx, wy)

        h1 = _h_at_grid(ix, iy)
        h2 = _h_at_grid(ix + 1, iy)
        h3 = _h_at_grid(ix, iy + 1)
        ground_z = np.minimum(np.minimum(h1, h2), h3)
        return (base_z - ground_z).astype(np.float32).reshape(-1)

    def _observations(self, command: np.ndarray):
        command = command.astype(np.float32).copy()
        if np.linalg.norm(command[:2]) < float(self.config["cmd_deadband"]):
            command[:2] = 0.0
        gravity = quat_projected_gravity(np.asarray(self.data.qpos[3:7]))
        q = self.joint_pos()
        dq = self.joint_vel()
        distances = self.height_distances()
        height_obs = np.clip(
            float(self.config["height_bias"]) - distances, -1.0, 1.0
        ) * float(self.config["height_scale"])
        command_obs = command * np.asarray(
            [
                self.config["command_lin_vel_scale"],
                self.config["command_lin_vel_scale"],
                self.config["command_yaw_scale"],
            ],
            dtype=np.float32,
        )
        q_obs = (q - self.policy_dof_pos) * float(self.config["dof_pos_scale"])
        dq_obs = dq * float(self.config["dof_vel_scale"])
        policy_step = np.concatenate(
            [command_obs, gravity, q_obs, dq_obs, self.last_actions.numpy()[0], height_obs]
        )
        estimator_step = np.concatenate(
            [gravity, q_obs, dq_obs, self.last_actions.numpy()[0], height_obs]
        )
        policy_step = np.clip(policy_step, -float(self.config["clip_obs"]), float(self.config["clip_obs"]))
        estimator_step = np.clip(estimator_step, -float(self.config["clip_obs"]), float(self.config["clip_obs"]))
        return (
            torch.from_numpy(policy_step).view(1, -1),
            torch.from_numpy(estimator_step).view(1, -1),
        )

    @torch.no_grad()
    def infer(self, command: np.ndarray) -> np.ndarray:
        policy_step, estimator_step = self._observations(command)
        self.policy_history = torch.cat(
            [self.policy_history[1:], policy_step], dim=0
        )
        self.estimator_history = torch.cat(
            [self.estimator_history[1:], estimator_step], dim=0
        )
        latent = self._tensor_output(
            self.adaptation(self.estimator_history.reshape(1, -1)),
            "adaptation module",
        )
        body_input = torch.cat([self.policy_history.reshape(1, -1), latent], dim=1)
        action = self._tensor_output(self.body(body_input), "body module")
        action = torch.clamp(
            action,
            -float(self.config["clip_actions"]),
            float(self.config["clip_actions"]),
        )
        self.last_actions = action.clone()
        self.infer_count += 1
        action_np = action.cpu().numpy()[0].astype(np.float32)
        clip = float(self.config["clip_actions"])
        self.action_abs_sum += float(np.abs(action_np).sum())
        self.action_saturated_count += int(
            np.count_nonzero(np.abs(action_np) >= clip - 1e-5)
        )
        self.action_value_count += NUM_JOINTS
        return action_np

    def target_from_action(self, action: np.ndarray, action_scale: float) -> np.ndarray:
        """Apply the same delayed position-target queue used during training."""
        scaled_action = np.asarray(action, dtype=np.float32) * float(action_scale)
        self.action_lag_buffer = self.action_lag_buffer[1:] + [scaled_action.copy()]
        return self.policy_dof_pos + self.action_lag_buffer[0]

    def step_control(
        self, target: np.ndarray, external_force: Optional[np.ndarray] = None
    ):
        if external_force is None:
            external_force = np.zeros(3, dtype=np.float32)
        external_force = np.asarray(external_force, dtype=np.float32).reshape(3)
        for _ in range(self.decimation):
            q = self.joint_pos()
            dq = self.joint_vel()
            torque = self.kp * (target - q) - self.kd * dq
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            for index, actuator_id in enumerate(self.actuator_idx):
                gear = abs(float(self.model.actuator_gear[actuator_id, 0]))
                self.data.ctrl[actuator_id] = torque[index] / max(gear, 1e-6)
            # xfrc_applied is persistent in MuJoCo, so overwrite it on every
            # physics step to ensure a released disturbance key removes force.
            self.data.xfrc_applied[self.base_body_id, :] = 0.0
            self.data.xfrc_applied[self.base_body_id, :3] = external_force
            mujoco.mj_step(self.model, self.data)

        gravity = quat_projected_gravity(np.asarray(self.data.qpos[3:7]))
        tilt = math.atan2(
            float(np.linalg.norm(gravity[:2])),
            max(-float(gravity[2]), 1e-6),
        )
        self.max_tilt_rad = max(self.max_tilt_rad, tilt)
        self.min_base_z = min(self.min_base_z, float(self.data.qpos[2]))
        self.max_base_z = max(self.max_base_z, float(self.data.qpos[2]))
        fall_height = float(self.config.get("fall_base_height", 0.15))
        fall_tilt = math.radians(float(self.config.get("fall_tilt_deg", 65.0)))
        self.fell = self.fell or float(self.data.qpos[2]) < fall_height or tilt > fall_tilt

    def summary(self) -> dict:
        base = np.asarray(self.data.qpos[:3], dtype=np.float32)
        return {
            "terrain": self.terrain_mode,
            "control_steps": self.infer_count,
            "sim_time_s": self.infer_count * self.control_dt,
            "base_xyz": [float(value) for value in base],
            "displacement_xyz": [float(value) for value in (base - self.initial_base)],
            "mean_abs_action": (
                self.action_abs_sum / self.action_value_count
                if self.action_value_count
                else 0.0
            ),
            "action_saturation_fraction": (
                self.action_saturated_count / self.action_value_count
                if self.action_value_count
                else 0.0
            ),
            "base_z_min": self.min_base_z,
            "base_z_max": self.max_base_z,
            "max_tilt_deg": math.degrees(self.max_tilt_rad),
            "fell": self.fell,
        }


def run(args):
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    apply_policy_profile(config, args.policy_profile)
    for argument, key in (
        (args.stairs_step_height, "stairs_step_height"),
        (args.stairs_tread_depth, "stairs_tread_depth"),
        (args.stairs_width, "stairs_width"),
        (args.stairs_x_start, "stairs_x_start"),
        (args.joint_damping_scale, "joint_damping_scale"),
        (args.joint_armature_scale, "joint_armature_scale"),
        (args.action_scale, "action_scale"),
        (args.clip_actions, "clip_actions"),
        (args.kp_scale, "kp_scale"),
        (args.kd_scale, "kd_scale"),
        (args.stairs_friction, "stairs_friction"),
    ):
        if argument is not None:
            config[key] = float(argument)
    # Deployment YAML paths are relative to deploy_cpp/, not the repository
    # root (the same convention used by the ROS2 deploy node).
    package_root = config_path.parents[2]
    if args.terrain_mode:
        terrain_mode = args.terrain_mode
    else:
        terrain_mode = config.get("terrain_mode", "flat")

    sim = MybotV3Sim(
        config,
        args.body.expanduser().resolve(),
        args.adaptation.expanduser().resolve(),
        terrain_mode,
        package_root,
    )
    controller = None
    if args.interactive:
        try:
            disturbance_force_n = (
                float(config.get("external_force_n", 0.0))
                if args.disturbance_force is None
                else float(args.disturbance_force)
            )
            controller = KeyboardController(
                disturbance_force_n,
                float(config.get("interactive_vx", 0.4)),
                float(config.get("interactive_vy", 0.3)),
                float(config.get("interactive_yaw", 0.8)),
            )
        except Exception as exc:
            raise RuntimeError(
                "--interactive requires a desktop keyboard listener (pynput)"
            ) from exc
        command = np.zeros(3, dtype=np.float32)
        print(
            "[sim2sim] keyboard: W/S forward, A/D lateral, Q/E yaw; "
            f"arrows external force={disturbance_force_n:.1f} N, U/O vertical force; "
            "1/2/3/4/5 select 8/10/12/15/20 cm lanes, R reset"
        )
    else:
        command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    print(
        f"[sim2sim] terrain={terrain_mode} command="
        f"[{command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}] "
        f"dt={config['dt']} decimation={config['decimation']} "
        f"policy_profile={args.policy_profile}"
    )
    if terrain_mode == "three_stairs":
        print(
            "[sim2sim] stair lanes: "
            f"y={config.get('three_stairs_lane_y', [-2.6, 0.0, 2.6, 5.2, 7.8])}, "
            f"step_heights_m={config.get('three_stairs_step_heights', [0.08, 0.10, 0.12, 0.15, 0.20])}, "
            f"steps_each={config.get('three_stairs_num_steps', THREE_STAIR_STEPS)}"
        )

    def tick(step_count):
        if controller is not None:
            if controller.consume("r"):
                sim.teleport(0.0, 0.0)
                print("[sim2sim] reset to origin")
            lane_y = config.get(
                "three_stairs_lane_y", [-2.6, 0.0, 2.6, 5.2, 7.8]
            )
            for lane_index, key in enumerate(("1", "2", "3", "4", "5")):
                if controller.consume(key):
                    sim.teleport(0.0, float(lane_y[lane_index]))
                    print(
                        f"[sim2sim] teleport to lane {key} "
                        f"({config.get('three_stairs_step_heights')[lane_index] * 100:.0f} cm, "
                        f"y={float(lane_y[lane_index]):+.1f})"
                    )
            command[:] = controller.command()
        action = sim.infer(command)
        target = sim.target_from_action(action, float(config["action_scale"]))
        disturbance = (
            controller.disturbance()
            if controller is not None
            else np.zeros(3, dtype=np.float32)
        )
        sim.step_control(target, disturbance)
        if step_count % 50 == 0:
            base = sim.data.qpos[:3]
            print(
                f"[sim2sim] step={step_count} base="
                f"[{base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f}] "
                f"action=[{action.min():.2f}, {action.max():.2f}]"
            )

    total_steps = max(1, int(args.duration / sim.control_dt))
    if args.headless:
        try:
            for step_count in range(total_steps):
                tick(step_count)
                if args.stop_on_fall and sim.fell:
                    print(f"[sim2sim] stop_on_fall at step={step_count}")
                    break
        finally:
            if controller is not None:
                controller.close()
    else:
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                for step_count in range(total_steps):
                    if not viewer.is_running():
                        break
                    start = time.time()
                    tick(step_count)
                    viewer.sync()
                    remaining = sim.control_dt - (time.time() - start)
                    if remaining > 0:
                        time.sleep(remaining)
                    if args.stop_on_fall and sim.fell:
                        print(f"[sim2sim] stop_on_fall at step={step_count}")
                        break
        finally:
            if controller is not None:
                controller.close()

    summary = sim.summary()
    print(f"[sim2sim] summary={json.dumps(summary, ensure_ascii=False)}")
    if args.metrics_json:
        metrics_path = args.metrics_json.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[sim2sim] metrics={metrics_path}")
    if args.fail_on_fall and sim.fell:
        raise RuntimeError("sim2sim detected a fall")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument(
        "--policy-profile",
        choices=("config", "current", "may2026"),
        default="config",
        help="override control normalization for a compatible older policy",
    )
    parser.add_argument(
        "--terrain-mode", choices=("flat", "stairs", "three_stairs"), default=None
    )
    parser.add_argument("--stairs-step-height", type=float, default=None)
    parser.add_argument("--stairs-tread-depth", type=float, default=None)
    parser.add_argument("--stairs-width", type=float, default=None)
    parser.add_argument("--stairs-x-start", type=float, default=None)
    parser.add_argument("--joint-damping-scale", type=float, default=None)
    parser.add_argument("--joint-armature-scale", type=float, default=None)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--clip-actions", type=float, default=None)
    parser.add_argument("--kp-scale", type=float, default=None)
    parser.add_argument("--kd-scale", type=float, default=None)
    parser.add_argument("--stairs-friction", type=float, default=None)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--disturbance-force",
        type=float,
        default=None,
        help="external force magnitude in N for interactive arrow/U/O keys",
    )
    parser.add_argument(
        "--stop-on-fall",
        action="store_true",
        help="stop the rollout when base height or tilt indicates a fall",
    )
    parser.add_argument(
        "--fail-on-fall",
        action="store_true",
        help="return a non-zero exit code if a fall is detected",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="write rollout summary metrics to this JSON file",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
