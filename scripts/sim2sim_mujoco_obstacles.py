#!/usr/bin/env python3
"""RoboCon 2026 competition-obstacle scenes for MyBot V3 MuJoCo testing.

Two obstacle courses aligned with the rulebook V2.0 dimensions (M0 report
`M0_REPORT_20261005.md`):

  * ``t_stairs``  T字形台阶: platform 1.0x1.0 m at 0.40 m, riser 0.10 m,
    tread 0.30 m, three-step flights on -x / +x / -y; overall 2.8 m x 1.9 m.
  * ``slope``     大斜坡: 3.0 m run at 11.3 deg (rise 0.5994 m), width 2.0 m,
    0.8 m plateau, mirrored descent; small 14 deg ramps are part of the
    bridge obstacles, not this one.

Headless smoke test:
    python scripts/sim2sim_mujoco_obstacles.py --obstacle t_stairs --duration 6
Interactive viewer (camera follows the robot):
    python scripts/sim2sim_mujoco_obstacles.py --obstacle slope --viewer
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sim2sim_mujoco import MybotV3Sim, apply_policy_profile, load_config  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "deploy_cpp/config/robots/mybot_v3_cse_sim.yaml"
CKPT_DIR = (
    PROJECT_ROOT
    / "runs/mybot_v3_sim2sim_phase3b_resume_059500/2026-10-03_07-04-56.647780/checkpoints"
)
DEFAULT_BODY = CKPT_DIR / "body_059950.jit"
DEFAULT_ADAPTATION = CKPT_DIR / "adaptation_module_059950.jit"
SCENE_DIR = PROJECT_ROOT / "tmp/mujoco_obstacles"
BASELINE_XML = PROJECT_ROOT / "deploy_cpp/robot/mybot_v3/xml/mybot_v3.xml.bak_matched"
ALIGNED_XML = PROJECT_ROOT / "deploy_cpp/robot/mybot_v3/xml/mybot_v3.xml"

SPAWN_HEIGHT = 0.35
FRICTION = "1 0.005 0.0001"

# ---------------------------------------------------------------------------
# Obstacle geometry (rulebook V2.0 / M0 report)
# ---------------------------------------------------------------------------
T_STAIRS = {
    "riser": 0.10,
    "tread": 0.30,
    "platform_xy": 1.0,
    "height": 0.40,
    "flight_width": 1.0,
    "steps_per_flight": 3,
    "color": "0.85 0.58 0.12 1",  # orange latex paint
}
SLOPE = {
    "angle_deg": 11.3,
    "run": 3.0,          # horizontal run per face
    "width": 2.0,        # total width (y)
    "plateau_len": 0.8,
    "thickness": 0.20,   # slab thickness (cosmetic/physical, half = 0.10)
    "color": "0.85 0.58 0.12 1",
}

T_STAIRS_SPAWN = (-2.30, 0.0)   # facing +x, closet flight starts at x=-1.40
SLOPE_SPAWN = (-0.70, 0.0)      # facing +x, ramp starts at x=0.0


def t_stairs_geom_specs():
    """Return list of (name, pos, size) boxes for the T-stairs obstacle."""
    riser = T_STAIRS["riser"]
    tread = T_STAIRS["tread"]
    half_plat = 0.5 * T_STAIRS["platform_xy"]
    height = T_STAIRS["height"]
    width = 0.5 * T_STAIRS["flight_width"]
    specs = [("tstair_platform", (0.0, 0.0, 0.5 * height), (half_plat, half_plat, 0.5 * height))]

    flight_levels = [height - (k + 1) * riser for k in range(T_STAIRS["steps_per_flight"])]
    for k, level in enumerate(flight_levels):  # 0.30, 0.20, 0.10
        offset = half_plat + (k + 0.5) * tread
        specs.append(
            (f"tstair_left_{k + 1}", (-offset, 0.0, 0.5 * level), (0.5 * tread, width, 0.5 * level))
        )
        specs.append(
            (f"tstair_right_{k + 1}", (+offset, 0.0, 0.5 * level), (0.5 * tread, width, 0.5 * level))
        )
        specs.append(
            (f"tstair_stem_{k + 1}", (0.0, -offset, 0.5 * level), (half_plat, 0.5 * tread, 0.5 * level))
        )
    return specs


def slope_geom_specs():
    """Return list of (name, pos, size, quat) boxes for the big-slope obstacle."""
    theta = math.radians(SLOPE["angle_deg"])
    run = SLOPE["run"]
    width = 0.5 * SLOPE["width"]
    plateau = SLOPE["plateau_len"]
    half_t = 0.5 * SLOPE["thickness"]
    rise = run * math.tan(theta)
    surface = math.hypot(run, rise)

    w = math.cos(0.5 * theta)  # half-angle quaternion component
    q_up = (w, 0.0, -math.sin(0.5 * theta), 0.0)   # rotate about y so +x climbs
    q_down = (w, 0.0, +math.sin(0.5 * theta), 0.0)

    def slab_center(face_center, q, sign):
        # top-face offset in world = R(q) @ (0, 0, +half_t)
        s2 = math.sin(0.5 * theta)
        c2 = math.cos(0.5 * theta)
        # R for q=(w,0,±s2,0): y-rotation by ∓theta
        cz = math.cos(theta)
        sz = math.sin(theta)
        if sign > 0:   # rotation by -theta
            R = np.array([[cz, 0, -sz], [0, 1, 0], [sz, 0, cz]])
        else:          # rotation by +theta
            R = np.array([[cz, 0, sz], [0, 1, 0], [-sz, 0, cz]])
        offset = R @ np.array([0.0, 0.0, half_t])
        return tuple(np.asarray(face_center) - offset)

    up_face_center = (0.5 * run, 0.0, 0.5 * rise)
    plateau_center = (run + 0.5 * plateau, 0.0, 0.5 * rise)
    down_face_center = (run + plateau + 0.5 * run, 0.0, 0.5 * rise)
    size = (0.5 * surface, width, half_t)

    return [
        ("slope_up", slab_center(up_face_center, q_up, +1), size, q_up),
        ("slope_plateau", plateau_center, (0.5 * plateau, width, 0.5 * rise), None),
        ("slope_down", slab_center(down_face_center, q_down, -1), size, q_down),
    ]


def obstacle_geoms_xml(obstacle: str) -> list:
    lines = []
    if obstacle == "t_stairs":
        for name, pos, size in t_stairs_geom_specs():
            lines.append(
                f'        <geom name="{name}" type="box" '
                f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" '
                f'contype="1" conaffinity="1" friction="{FRICTION}" '
                f'rgba="{T_STAIRS["color"]}" />'
            )
    elif obstacle == "slope":
        for name, pos, size, quat in slope_geom_specs():
            quat_attr = ""
            if quat is not None:
                quat_attr = f'quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}" '
            lines.append(
                f'        <geom name="{name}" type="box" '
                f'{quat_attr}'
                f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" '
                f'contype="1" conaffinity="1" friction="{FRICTION}" '
                f'rgba="{SLOPE["color"]}" />'
            )
    else:
        raise ValueError(f"unknown obstacle: {obstacle}")
    return lines


def build_scene_xml(base_text: str, xml_path: Path, obstacle: str) -> str:
    """Strip the built-in placeholder staircase and inject the obstacle."""
    text, removed = re.subn(
        r'^\s*<geom name="sim_(?:stair|three_stair)_[^"]*"[^>]*/>\s*\n',
        "",
        base_text,
        flags=re.M,
    )
    if removed < 9:
        raise RuntimeError(
            f"expected >=9 sim_stair placeholder geoms in robot XML, removed {removed}"
        )
    mesh_dir = (xml_path.parent / "../meshes").resolve()
    text, subs = re.subn(r'meshdir="[^"]*"', f'meshdir="{mesh_dir}"', text, count=1)
    if subs != 1:
        raise RuntimeError("meshdir attribute not found in robot XML")

    anchor = '<geom name="floor"'
    insert_at = text.index("\n", text.index(anchor)) + 1
    header = (
        f"        <!-- {obstacle} competition scene "
        "(generated by sim2sim_mujoco_obstacles.py) -->\n"
    )
    body = "\n".join(obstacle_geoms_xml(obstacle))
    return text[:insert_at] + header + body + "\n" + text[insert_at:]


class ObstacleSim(MybotV3Sim):
    """MybotV3Sim with one RoboCon competition obstacle in the scene."""

    OBSTACLE_SPAWNS = {"t_stairs": T_STAIRS_SPAWN, "slope": SLOPE_SPAWN}

    def __init__(self, config, body_path, adaptation_path, package_root, profile, obstacle):
        if obstacle not in self.OBSTACLE_SPAWNS:
            raise ValueError(f"obstacle must be one of {list(self.OBSTACLE_SPAWNS)}")
        self.obstacle = obstacle
        self.profile = profile
        xml_path = BASELINE_XML if profile == "baseline" else ALIGNED_XML
        if not xml_path.is_file():
            raise RuntimeError(f"MuJoCo source XML is missing: {xml_path}")
        scene_text = build_scene_xml(xml_path.read_text(), xml_path, obstacle)
        SCENE_DIR.mkdir(parents=True, exist_ok=True)
        scene_path = SCENE_DIR / f"mybot_v3_{obstacle}_{os.getpid()}.xml"
        scene_path.write_text(scene_text)
        self.scene_path = scene_path
        config = dict(config)
        config["action_lag_steps"] = 0
        config["mujoco_xml_relpath"] = str(scene_path)
        super().__init__(config, body_path, adaptation_path, "stairs", package_root)
        if profile not in ("baseline", "dynamics"):
            self._configure_single_variable_profile(profile)

    # -- profile handling mirrors sim2sim_mujoco_3stairs.py ----------------
    def _configure_single_variable_profile(self, profile):
        for joint_index, dof_index in enumerate(self.joint_qvel_idx):
            kind = joint_index % 3
            baseline_damping = 1.0 if kind == 0 else 2.0
            baseline_armature = 0.0544 if kind == 2 else 0.01
            aligned_frictionloss = 0.3 if kind == 2 else 0.1
            self.model.dof_damping[dof_index] = (
                0.0 if profile == "damping" else baseline_damping
            )
            self.model.dof_frictionloss[dof_index] = (
                aligned_frictionloss
                if profile in ("friction", "aligned_damped")
                else 0.2
            )
            self.model.dof_armature[dof_index] = (
                0.01
                if profile in ("armature", "aligned_damped")
                else baseline_armature
            )
        if profile != "aligned_damped":
            for geom_id in range(self.model.ngeom):
                if self.model.geom_bodyid[geom_id] != 0 and self.model.geom_contype[geom_id]:
                    self.model.geom_friction[geom_id] = (0.6, 0.3, 0.3)

    def _configure_terrain(self):
        names = []
        if self.obstacle == "t_stairs":
            names = [spec[0] for spec in t_stairs_geom_specs()]
        elif self.obstacle == "slope":
            names = [spec[0] for spec in slope_geom_specs()]
        active_ids = []
        self.geom_name_by_id = {}
        for name in names:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise RuntimeError(f"scene geom missing: {name}")
            self.model.geom_contype[geom_id] = 1
            self.model.geom_conaffinity[geom_id] = 1
            active_ids.append(int(geom_id))
            self.geom_name_by_id[int(geom_id)] = name
        return active_ids

    def teleport(self, x: float, y: float, yaw: float = 0.0):
        qpos = np.asarray(self.data.qpos).copy()
        qpos[0:3] = (x, y, SPAWN_HEIGHT)
        half = 0.5 * yaw
        qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        for index, qpos_index in enumerate(self.joint_qpos_idx):
            self.data.qpos[qpos_index] = self.policy_dof_pos[index]
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.policy_history.zero_()
        self.estimator_history.zero_()
        self.last_actions.zero_()
        self.action_lag_buffer = [
            np.zeros(12, dtype=np.float32) for _ in range(self.action_lag_steps + 1)
        ]
        self.initial_base = np.asarray(self.data.qpos[:3], dtype=np.float32).copy()
        self.min_base_z = float(self.data.qpos[2])
        self.max_base_z = float(self.data.qpos[2])
        self.max_tilt_rad = 0.0
        self.fell = False

    def _top_z_of_geom(self, geom_id: int, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        """Exact top-surface height of an oriented box under each query point."""
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        rot = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        wx = np.asarray(wx, dtype=np.float64)
        wy = np.asarray(wy, dtype=np.float64)
        t_lo = np.full(wx.shape, -1e9)
        t_hi = np.full(wx.shape, +1e9)
        miss = np.zeros(wx.shape, dtype=bool)
        for axis in range(3):
            col = rot[:, axis]
            a = col[0] * (wx - center[0]) + col[1] * (wy - center[1]) - col[2] * center[2]
            b = col[2]
            if abs(b) < 1e-12:
                miss |= np.abs(a) > size[axis]
            else:
                t1 = (-size[axis] - a) / b
                t2 = (+size[axis] - a) / b
                lo = np.minimum(t1, t2)
                hi = np.maximum(t1, t2)
                t_lo = np.maximum(t_lo, lo)
                t_hi = np.minimum(t_hi, hi)
        miss |= t_lo > t_hi
        out = np.where(miss, -1e9, t_hi)
        return out

    def height_distances(self) -> np.ndarray:
        """Height map query using exact oriented-box tops (replaces AABB query)."""
        base_x, base_y, base_z = (float(v) for v in self.data.qpos[:3])
        w, x, y, z = self.data.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cy, sy = math.cos(yaw), math.sin(yaw)
        grid_x, grid_y = np.meshgrid(self.height_x, self.height_y, indexing="ij")
        world_x = base_x + cy * grid_x - sy * grid_y
        world_y = base_y + sy * grid_x + cy * grid_y

        ground_z = np.zeros_like(world_x, dtype=np.float64)
        for geom_id in self.stair_geom_ids:
            top = self._top_z_of_geom(geom_id, world_x, world_y)
            ground_z = np.maximum(ground_z, top)
        return (base_z - ground_z).astype(np.float32).reshape(-1)

    # -- helpers for evaluation ---------------------------------------------
    def foot_body_ids(self):
        return [self.model.body(f"{side}_foot").id for side in ("FL", "FR", "RL", "RR")]

    def contact_labels(self):
        """Return set of obstacle geom names currently in contact with a foot."""
        foot_bodies = set(self.foot_body_ids())
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        hit = set()
        floor_hit = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for geoma, geomb in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if geoma in self.geom_name_by_id and self.model.geom_bodyid[geomb] in foot_bodies:
                    hit.add(self.geom_name_by_id[geoma])
                if geoma == floor_id and self.model.geom_bodyid[geomb] in foot_bodies:
                    floor_hit = True
        return hit, floor_hit


def run_headless(sim: ObstacleSim, vx: float, duration: float, action_scale: float,
                 stand_seconds: float = 1.0):
    """Stand briefly, then walk forward for `duration`; return metrics dict."""
    motion_kp = sim.kp.copy()
    motion_kd = sim.kd.copy()
    stand_target = np.clip(
        sim.policy_dof_pos
        + np.array([-0.05, 0.02, 0.10, +0.05, 0.02, 0.10,
                    -0.08, 0.02, 0.11, +0.08, 0.02, 0.11], dtype=np.float32),
        sim.lower,
        sim.upper,
    )
    sim.kp[:] = 2.0 * motion_kp
    sim.kd[:] = 2.0 * motion_kd
    last_target = sim.policy_dof_pos.copy()
    for _ in range(max(1, int(stand_seconds / sim.control_dt))):
        last_target += 0.12 * (stand_target - last_target)
        sim.step_control(last_target, None)
    sim.kp[:] = motion_kp
    sim.kd[:] = motion_kd

    command = np.array([vx, 0.0, 0.0], dtype=np.float32)
    touched = set()
    floor_steps = 0
    steps = max(1, int(duration / sim.control_dt))
    for _ in range(steps):
        action = sim.infer(command)
        target = sim.target_from_action(action, action_scale)
        last_target = np.clip(target, sim.lower, sim.upper)
        sim.step_control(last_target, None)
        hit, floor_hit = sim.contact_labels()
        touched |= hit
        floor_steps += int(floor_hit)
        if sim.fell:
            break
    summary = sim.summary()
    summary["touched"] = sorted(touched)
    summary["floor_contact_steps"] = floor_steps
    summary["obstacle"] = sim.obstacle
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obstacle", choices=("t_stairs", "slope"), default="t_stairs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--profile", default="aligned_damped")
    parser.add_argument("--spawn-x", type=float, default=None)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    apply_policy_profile(config, "config")
    package_root = config_path.parents[2]

    sim = ObstacleSim(
        config,
        args.body.expanduser().resolve(),
        args.adaptation.expanduser().resolve(),
        package_root,
        args.profile,
        args.obstacle,
    )
    spawn_x, spawn_y = ObstacleSim.OBSTACLE_SPAWNS[args.obstacle]
    if args.spawn_x is not None:
        spawn_x = args.spawn_x
    if args.spawn_y:
        spawn_y = args.spawn_y
    sim.teleport(spawn_x, spawn_y, 0.0)
    action_scale = float(config["action_scale"])

    print(f"[obstacles] scene={sim.scene_path}")
    print(f"[obstacles] obstacle={args.obstacle} profile={args.profile} spawn=({spawn_x}, {spawn_y})")

    if args.viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            # stand a moment
            motion_kp, motion_kd = sim.kp.copy(), sim.kd.copy()
            stand_target = np.clip(sim.policy_dof_pos, sim.lower, sim.upper)
            last_target = sim.policy_dof_pos.copy()
            for _ in range(int(1.0 / sim.control_dt)):
                last_target += 0.12 * (stand_target - last_target)
                sim.step_control(last_target, None)
                viewer.sync()
            command = np.array([args.vx, 0.0, 0.0], dtype=np.float32)
            steps = max(1, int(args.duration / sim.control_dt))
            for _ in range(steps):
                if sim.fell:
                    break
                action = sim.infer(command)
                target = sim.target_from_action(action, action_scale)
                last_target = np.clip(target, sim.lower, sim.upper)
                sim.step_control(last_target, None)
                viewer.cam.lookat[:] = sim.data.qpos[:3]
                viewer.cam.distance = 3.5
                viewer.cam.elevation = -18
                viewer.sync()
        summary = sim.summary()
        print(summary)
    else:
        summary = run_headless(sim, args.vx, args.duration, action_scale)
        print(f"final xyz = {[round(v, 3) for v in summary['base_xyz']]}")
        print(f"touched   = {summary['touched']}")
        print(f"fell      = {summary['fell']}  max_tilt={summary['max_tilt_deg']:.1f} deg "
              f"floor_contact_steps={summary['floor_contact_steps']}")


if __name__ == "__main__":
    main()
