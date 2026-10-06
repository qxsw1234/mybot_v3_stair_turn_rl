#!/usr/bin/env python3
"""Run the best MyBot V3 sim2sim policy on the full RoboCon map.

The viewer is interactive by default.  Click the MuJoCo window and use:

  W/S forward/back, A/D lateral, Q/E yaw
  1..8 teleport to useful competition-map test positions
  R reset to the currently selected position
  arrows and U/O apply horizontal/vertical disturbances

Use ``--headless`` for a short non-graphical smoke test.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sim2sim_mujoco import (  # noqa: E402
    KeyboardController,
    MybotV3Sim,
    apply_policy_profile,
    load_config,
)


DEFAULT_CONFIG = PROJECT_ROOT / "deploy_cpp/config/robots/mybot_v3_cse_sim.yaml"
DEFAULT_MAP_XML = PROJECT_ROOT / "deploy_cpp/robot/mybot_v3/xml/mybot_v3_1hao.xml"
CKPT_DIR = (
    PROJECT_ROOT
    / "runs/mybot_v3_sim2sim_phase3b_resume_059500"
    / "2026-10-03_07-04-56.647780/checkpoints"
)
DEFAULT_BODY = CKPT_DIR / "body_best_sim2sim.jit"
DEFAULT_ADAPTATION = CKPT_DIR / "adaptation_module_best_sim2sim.jit"


# Coordinates match mybot_v3_1hao.xml.  Each pose is placed on ordinary floor
# immediately before an obstacle, so reset never starts the robot in collision.
SPAWNS = {
    "start": (-5.50, 1.90, math.pi / 2.0, "地面启动区"),
    "t_stairs": (-1.95, -0.04, 0.0, "T 字台阶西侧入口"),
    "wall": (-4.30, -0.01, 0.0, "高墙入口"),
    "low_bar": (-5.85, 3.91, 0.0, "限高杆入口"),
    "slope": (-3.95, 2.87, 0.0, "大斜坡入口"),
    "slalom": (-5.85, 0.01, math.pi, "直角绕杆入口"),
    "gravel": (-9.10, 2.97, 0.0, "砂砾碎木坑入口"),
    "bridge": (3.42, -0.65, math.pi / 2.0, "木桥入口"),
}
SPAWN_KEYS = dict(zip("12345678", SPAWNS))


class RoboconMapSim(MybotV3Sim):
    """MyBot policy runner with the complete static RoboCon map enabled."""

    def __init__(self, config, body_path, adaptation_path, package_root, map_xml):
        self.map_xml = map_xml
        config = dict(config)
        config["mujoco_xml_relpath"] = str(map_xml)
        # The selected sim2sim checkpoint was validated without an artificial
        # command/action delay in the MuJoCo bridge.
        config["action_lag_steps"] = 0
        super().__init__(
            config, body_path, adaptation_path, "stairs", package_root
        )
        self._configure_aligned_damped_dynamics()

    def _configure_terrain(self):
        """Disable generated test stairs and index full-map collision boxes."""
        active_ids = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
            )
            if name and (
                name.startswith("sim_stair_")
                or name.startswith("sim_three_stair_")
            ):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_rgba[geom_id, 3] = 0.0
                continue

            # Only static world boxes contribute to the learned 77-point
            # terrain-height observation.  Cylinders (poles/low bar) still
            # collide physically but are intentionally excluded from the
            # ground-height scan, matching their role as obstacles rather
            # than walkable ground.
            if (
                int(self.model.geom_bodyid[geom_id]) == 0
                and int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
                and int(self.model.geom_contype[geom_id]) != 0
            ):
                active_ids.append(geom_id)
        if not active_ids:
            raise RuntimeError("RoboCon map contains no active world collision boxes")
        return active_ids

    def _configure_aligned_damped_dynamics(self):
        """Use the MuJoCo dynamics profile selected by Phase3b evaluation."""
        for joint_index, dof_index in enumerate(self.joint_qvel_idx):
            kind = joint_index % 3
            self.model.dof_damping[dof_index] = 1.0 if kind == 0 else 2.0
            self.model.dof_frictionloss[dof_index] = 0.3 if kind == 2 else 0.1
            self.model.dof_armature[dof_index] = 0.01

    def _top_z_of_box(self, geom_id, world_x, world_y):
        """Return exact top intersections for an arbitrarily oriented box."""
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(
            self.data.geom_xmat[geom_id], dtype=np.float64
        ).reshape(3, 3)
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        world_x = np.asarray(world_x, dtype=np.float64)
        world_y = np.asarray(world_y, dtype=np.float64)
        lower = np.full(world_x.shape, -1e9)
        upper = np.full(world_x.shape, +1e9)
        miss = np.zeros(world_x.shape, dtype=bool)
        for axis in range(3):
            column = rotation[:, axis]
            a = (
                column[0] * (world_x - center[0])
                + column[1] * (world_y - center[1])
                - column[2] * center[2]
            )
            b = column[2]
            if abs(b) < 1e-12:
                miss |= np.abs(a) > size[axis]
            else:
                t1 = (-size[axis] - a) / b
                t2 = (+size[axis] - a) / b
                lower = np.maximum(lower, np.minimum(t1, t2))
                upper = np.minimum(upper, np.maximum(t1, t2))
        miss |= lower > upper
        return np.where(miss, -1e9, upper)

    def height_distances(self):
        """Build the policy height observation from the full oriented map."""
        base_x, base_y, base_z = (float(value) for value in self.data.qpos[:3])
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )
        cy, sy = math.cos(yaw), math.sin(yaw)
        grid_x, grid_y = np.meshgrid(self.height_x, self.height_y, indexing="ij")
        world_x = base_x + cy * grid_x - sy * grid_y
        world_y = base_y + sy * grid_x + cy * grid_y
        ground_z = np.zeros_like(world_x, dtype=np.float64)
        for geom_id in self.stair_geom_ids:
            ground_z = np.maximum(
                ground_z, self._top_z_of_box(geom_id, world_x, world_y)
            )
        return (base_z - ground_z).astype(np.float32).reshape(-1)


def teleport_to(sim, spawn_name):
    x, y, yaw, label = SPAWNS[spawn_name]
    sim.teleport(x, y, yaw)
    print(
        f"[robocon] 传送到 {label}: "
        f"x={x:+.2f}, y={y:+.2f}, yaw={math.degrees(yaw):+.0f} deg"
    )


def settle(sim, seconds, viewer=None):
    """Let contacts settle in the nominal standing pose before policy control."""
    target = np.clip(sim.policy_dof_pos, sim.lower, sim.upper)
    steps = max(1, int(seconds / sim.control_dt))
    for _ in range(steps):
        sim.step_control(target)
        if viewer is not None:
            viewer.sync()


class AutoStandController:
    """Blend between the locomotion policy and a quiet nominal PD stand."""

    def __init__(self, nominal, control_dt, deadband, delay, blend_seconds):
        self.nominal = np.asarray(nominal, dtype=np.float32).copy()
        self.control_dt = float(control_dt)
        self.deadband = float(deadband)
        self.delay = max(0.0, float(delay))
        self.blend_seconds = max(self.control_dt, float(blend_seconds))
        self.reset()

    def reset(self):
        # Start quietly.  There is no reason to run the gait policy before the
        # user has issued the first motion command.
        self.standing = True
        self.idle_seconds = self.delay
        self.blend_elapsed = self.blend_seconds
        self.transition_from = self.nominal.copy()
        self.applied_target = self.nominal.copy()

    def update_mode(self, command):
        """Update command-idle timing and return a mode-change label."""
        moving = float(np.linalg.norm(command)) > self.deadband
        if moving:
            self.idle_seconds = 0.0
            if self.standing:
                self.standing = False
                self.transition_from = self.applied_target.copy()
                self.blend_elapsed = 0.0
                return "policy"
            return None

        if self.standing:
            return None
        self.idle_seconds += self.control_dt
        if self.idle_seconds + 1e-9 >= self.delay:
            self.standing = True
            self.transition_from = self.applied_target.copy()
            self.blend_elapsed = 0.0
            return "stand"
        return None

    def blend_target(self, desired):
        """Cubic-smooth a newly selected controller into the applied target."""
        desired = np.asarray(desired, dtype=np.float32)
        if self.blend_elapsed < self.blend_seconds:
            self.blend_elapsed = min(
                self.blend_seconds, self.blend_elapsed + self.control_dt
            )
            alpha = self.blend_elapsed / self.blend_seconds
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - alpha) * self.transition_from + alpha * desired
        else:
            target = desired
        self.applied_target = np.asarray(target, dtype=np.float32).copy()
        return self.applied_target


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--map-xml", type=Path, default=DEFAULT_MAP_XML)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument("--spawn", choices=tuple(SPAWNS), default="start")
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=None)
    parser.add_argument("--yaw-rate", type=float, default=None)
    parser.add_argument("--action-scale", type=float, default=None)
    parser.add_argument("--disturbance-force", type=float, default=None)
    parser.add_argument(
        "--stand-delay",
        type=float,
        default=0.20,
        help="seconds at zero command before switching to quiet PD stand",
    )
    parser.add_argument(
        "--stand-blend",
        type=float,
        default=0.35,
        help="seconds used to blend between policy and quiet stand",
    )
    parser.add_argument(
        "--disable-auto-stand",
        action="store_true",
        help="always apply the locomotion policy, including at zero command",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--headless-command",
        nargs=3,
        type=float,
        metavar=("VX", "VY", "YAW"),
        default=(0.0, 0.0, 0.0),
    )
    return parser


def main():
    args = build_parser().parse_args()
    config_path = args.config.expanduser().resolve()
    map_xml = args.map_xml.expanduser().resolve()
    body_path = args.body.expanduser().resolve()
    adaptation_path = args.adaptation.expanduser().resolve()
    for required in (config_path, map_xml, body_path, adaptation_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = load_config(config_path)
    apply_policy_profile(config, "config")
    if args.action_scale is not None:
        config["action_scale"] = float(args.action_scale)
    package_root = config_path.parents[2]
    sim = RoboconMapSim(
        config, body_path, adaptation_path, package_root, map_xml
    )
    teleport_to(sim, args.spawn)

    print(f"[robocon] map={map_xml}")
    print(f"[robocon] body={body_path}")
    print(f"[robocon] adaptation={adaptation_path}")
    print(
        f"[robocon] height-map boxes={len(sim.stair_geom_ids)}, "
        f"physics geoms={sim.model.ngeom}, duration={args.duration:.1f}s"
    )

    action_scale = float(config["action_scale"])
    total_steps = max(1, int(args.duration / sim.control_dt))
    selected_spawn = args.spawn
    controller = None
    auto_stand = None
    if not args.disable_auto_stand:
        auto_stand = AutoStandController(
            sim.policy_dof_pos,
            sim.control_dt,
            float(config.get("cmd_deadband", 0.05)),
            args.stand_delay,
            args.stand_blend,
        )

    def control_step(command, disturbance=None, announce=False):
        mode_change = auto_stand.update_mode(command) if auto_stand else None
        if announce and mode_change == "stand":
            print("[robocon] 自动站立: 固定关节目标，停止步态策略输出")
        elif announce and mode_change == "policy":
            print("[robocon] 运动控制: 平滑切回 RL 策略")

        if auto_stand is not None and auto_stand.standing:
            # Keep recurrent histories current without evaluating or counting
            # a gait action that is not physically applied.  Recording zero as
            # the previous action makes the next hand-off match the held pose.
            sim.last_actions.zero_()
            policy_step, estimator_step = sim._observations(
                np.zeros(3, dtype=np.float32)
            )
            sim.policy_history[:-1].copy_(sim.policy_history[1:].clone())
            sim.policy_history[-1].copy_(policy_step[0])
            sim.estimator_history[:-1].copy_(sim.estimator_history[1:].clone())
            sim.estimator_history[-1].copy_(estimator_step[0])
            sim.infer_count += 1
            sim.action_lag_buffer = [
                np.zeros(12, dtype=np.float32)
                for _ in range(sim.action_lag_steps + 1)
            ]
            desired = sim.policy_dof_pos
        else:
            action = sim.infer(command)
            desired = np.clip(
                sim.target_from_action(action, action_scale), sim.lower, sim.upper
            )
        target = auto_stand.blend_target(desired) if auto_stand else desired
        sim.step_control(target, disturbance)
        return mode_change

    if args.headless:
        command = np.asarray(args.headless_command, dtype=np.float32)
        settle(sim, min(0.8, args.duration * 0.25))
        for step in range(total_steps):
            control_step(command)
            if step % 50 == 0:
                print(
                    f"[robocon] step={step} xyz="
                    f"{np.asarray(sim.data.qpos[:3]).round(3).tolist()}"
                )
    else:
        disturbance_force = (
            float(config.get("external_force_n", 0.0))
            if args.disturbance_force is None
            else float(args.disturbance_force)
        )
        controller = KeyboardController(
            disturbance_force,
            float(args.vx if args.vx is not None else config.get("interactive_vx", 0.4)),
            float(args.vy if args.vy is not None else config.get("interactive_vy", 0.3)),
            float(
                args.yaw_rate
                if args.yaw_rate is not None
                else config.get("interactive_yaw", 0.8)
            ),
        )
        controller.keys.update({"6": False, "7": False, "8": False})
        print(
            "[robocon] 键盘: W/S 前后, A/D 横移, Q/E 转向, R 当前点复位; "
            "1 启动区, 2 T台阶, 3 高墙, 4 限高杆, "
            "5 大斜坡, 6 绕杆, 7 碎木坑, 8 木桥"
        )
        print(
            f"[robocon] 方向键施加 {disturbance_force:.1f}N 水平扰动, "
            "U/O 施加竖直扰动"
        )
        if auto_stand is not None:
            print(
                f"[robocon] 自动站立已启用: 松键 {args.stand_delay:.2f}s 后进入, "
                f"切换平滑时间 {args.stand_blend:.2f}s"
            )
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.lookat[:] = sim.data.qpos[:3]
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -32.0
                settle(sim, 0.8, viewer)
                for step in range(total_steps):
                    if not viewer.is_running():
                        break
                    start_time = time.time()
                    if controller.consume("r"):
                        teleport_to(sim, selected_spawn)
                        if auto_stand is not None:
                            auto_stand.reset()
                    for key, spawn_name in SPAWN_KEYS.items():
                        if controller.consume(key):
                            selected_spawn = spawn_name
                            teleport_to(sim, selected_spawn)
                            if auto_stand is not None:
                                auto_stand.reset()

                    command = controller.command()
                    control_step(
                        command, controller.disturbance(), announce=True
                    )
                    viewer.cam.lookat[:] = sim.data.qpos[:3]
                    viewer.sync()
                    if step % 100 == 0:
                        xyz = np.asarray(sim.data.qpos[:3]).round(3).tolist()
                        print(
                            f"[robocon] step={step} xyz={xyz} "
                            f"cmd={command.round(2).tolist()} fell={sim.fell}"
                        )
                    remaining = sim.control_dt - (time.time() - start_time)
                    if remaining > 0.0:
                        time.sleep(remaining)
        finally:
            controller.close()

    summary = sim.summary()
    summary.update(
        {
            "map": str(map_xml),
            "body": str(body_path),
            "adaptation": str(adaptation_path),
            "spawn": selected_spawn,
        }
    )
    print(f"[robocon] summary={json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
