#!/usr/bin/env python3
"""Zero-training MuJoCo baselines for the first RoboCon 2027 skills.

Rule-sized scenes are built for:

* slalom:  ``059950`` locomotion policy driven by a pure-pursuit S-weave;
* beam:    ``059950`` locomotion policy driven by pure-pursuit centre-line
  tracking;
* low_frame: ``063296`` low-bar policy with its seven-field obstacle input.

No training happens here.  The script classifies each checkpoint as directly
reusable, needing fine-tune, or needing a new expert, and records the rule
events, completion time and contact telemetry the M1 gate needs.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mujoco
import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from robodog_gym.envs.robocon_2027 import ObstacleAttempt, load_rules  # noqa: E402
from sim2sim_mujoco import (  # noqa: E402
    HISTORY_LENGTH,
    NUM_JOINTS,
    MybotV3Sim,
    apply_policy_profile,
    load_config,
    quat_projected_gravity,
)


DEFAULT_CONFIG = PROJECT_ROOT / "deploy_cpp/config/robots/mybot_v3_cse_sim.yaml"
BASE_XML = PROJECT_ROOT / "deploy_cpp/robot/mybot_v3/xml/mybot_v3.xml"
SCENE_DIR = PROJECT_ROOT / "tmp/robocon_2027"

LOCOMOTION_CKPT = (
    PROJECT_ROOT
    / "runs/mybot_v3_sim2sim_phase3b_resume_059500"
    / "2026-10-03_07-04-56.647780/checkpoints"
)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> Dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        check=True, capture_output=True).stdout.decode().strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=PROJECT_ROOT,
        check=True, capture_output=True).stdout
    return {
        "commit": head,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "dirty": bool(diff),
    }


LOW_FRAME_CKPT = (
    PROJECT_ROOT
    / "runs/mybot_v3_low_bar_expert_resume_062097"
    / "2026-10-10_13-36-23.241602/checkpoints"
)

MODELS = {
    "slalom": (LOCOMOTION_CKPT / "body_059950.jit", LOCOMOTION_CKPT / "adaptation_module_059950.jit"),
    "beam": (LOCOMOTION_CKPT / "body_059950.jit", LOCOMOTION_CKPT / "adaptation_module_059950.jit"),
    "low_frame": (LOW_FRAME_CKPT / "body_latest.jit", LOW_FRAME_CKPT / "adaptation_module_latest.jit"),
}

FRICTION = "1 0.005 0.0001"
GREEN = "0.10 0.65 0.25 1"

SPAWNS = {
    # Start outside the required zone, then explicitly drive through it.
    "slalom": (-1.30, 0.0, 0.0),
    "beam": (-0.70, 0.0, 0.0),
    "low_frame": (-2.00, 0.0, 0.0),
}

# Rulebook table 1: six 100 mm posts, 900 mm centre spacing, L-shaped layout.
SLALOM_POLES = (
    (0.0, 0.0),
    (0.9, 0.0),
    (1.8, 0.0),
    (2.7, 0.0),
    (2.7, 0.9),
    (2.7, 1.8),
)
# 300 mm circles at both ends and outside the corner of the L.
SLALOM_ZONES = {
    "start": (-0.40, 0.0),
    "corner": (3.00, -0.30),
    "finish": (2.70, 2.20),
}
# Centre-line weave, built from the measured keep-out radius.  A straight pass
# test showed the robot starts clipping post 1 once the base is closer than
# about 0.33 m, so the reference rides an exclusion circle of radius 0.37 m
# around every post.  Post 1 is taken on +y, post 2 on -y, post 3 on +y and
# post 4 on -y so the outside corner zone (3.00, -0.30) is reached tangentially.
# Posts 5 and 6 are then taken on +x and -x respectively.
SLALOM_WAYPOINTS = (
    (-1.20, 0.00),
    (-0.70, 0.00),
    (-0.40, 0.14),
    (-0.30, 0.24),
    (-0.18, 0.33),
    (-0.06, 0.38),
    (0.00, 0.39),
    (0.06, 0.38),
    (0.18, 0.33),
    (0.30, 0.24),
    (0.45, 0.00),
    (0.60, -0.24),
    (0.72, -0.33),
    (0.84, -0.38),
    (0.90, -0.39),
    (0.96, -0.38),
    (1.08, -0.33),
    (1.20, -0.24),
    (1.35, 0.00),
    (1.50, 0.24),
    (1.62, 0.33),
    (1.74, 0.38),
    (1.80, 0.39),
    (1.86, 0.38),
    (1.98, 0.33),
    (2.10, 0.24),
    (2.25, 0.00),
    (2.40, -0.24),
    (2.52, -0.33),
    (2.64, -0.38),
    (2.70, -0.39),
    (2.76, -0.38),
    (2.88, -0.34),
    (3.00, -0.30),
    (3.08, 0.00),
    (3.10, 0.20),
    (3.09, 0.45),
    (3.09, 0.90),
    (3.00, 1.15),
    (2.78, 1.35),
    (2.50, 1.50),
    (2.36, 1.65),
    (2.32, 1.80),
    (2.26, 2.00),
    (2.40, 2.26),
    (2.70, 2.20),
)
BEAM_WAYPOINTS = (
    (-0.60, 0.0),
    (0.00, 0.0),
    (0.60, 0.0),
    (1.50, 0.0),
    (2.40, 0.0),
    (3.00, 0.0),
    (3.45, 0.0),
)
LOW_FRAME_WAYPOINTS = ((-1.20, 0.0), (0.40, 0.0), (1.40, 0.0))

PATH_BY_OBSTACLE = {
    "slalom": SLALOM_WAYPOINTS,
    "beam": BEAM_WAYPOINTS,
    "low_frame": LOW_FRAME_WAYPOINTS,
}

PURSUIT = {
    # ``speed`` is the commanded body-frame forward speed.  The low-level
    # policy tracks roughly half of the commanded value at walking pace, so
    # these numbers are deliberately above the intended ground speed.
    "slalom": dict(speed=0.28, k_cross=4.0, k_yaw=1.4, vy_max=0.40, yaw_max=0.45, k_slow=3.0, heading_slow=3.0, lookahead_m=0.0, curvature_slow=0.0),
    # The measured stance is 0.405 m wide, the bridge is 0.400 m wide, so the
    # centre line error budget is only about +-0.15 m.  Give lateral recovery
    # strong priority over forward progress.
    "beam": dict(speed=0.24, k_cross=9.0, k_yaw=1.8, vy_max=0.35, yaw_max=0.40, k_slow=0.0),
    "low_frame": dict(speed=0.22, k_cross=2.5, k_yaw=1.2, vy_max=0.20, yaw_max=0.40, k_slow=3.0),
}

STAND_OFFSET = np.array(
    [-0.05, 0.02, 0.10, +0.05, 0.02, 0.10,
     -0.08, 0.02, 0.11, +0.08, 0.02, 0.11],
    dtype=np.float32,
)


def _box(name: str, pos: Sequence[float], size: Sequence[float], surface: bool = False) -> str:
    prefix = "rc27_surface_" if surface else "rc27_obstacle_"
    return (
        f'        <geom name="{prefix}{name}" type="box" '
        f'pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
        f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" '
        f'contype="1" conaffinity="1" friction="{FRICTION}" rgba="{GREEN}" />'
    )


def obstacle_geoms_xml(obstacle: str, rules: Dict) -> List[str]:
    spec = rules["obstacles"][obstacle]
    if obstacle == "beam":
        return [
            _box(
                "beam",
                (0.5 * spec["length_m"], 0.0, 0.5 * spec["height_m"]),
                (0.5 * spec["length_m"], 0.5 * spec["width_m"], 0.5 * spec["height_m"]),
                surface=True,
            )
        ]
    if obstacle == "low_frame":
        length = float(spec["length_m"])
        inner_width = float(spec["inner_width_m"])
        clearance = float(spec["inner_height_m"])
        beam_thickness = 0.05
        half_t = 0.5 * beam_thickness
        post_y = 0.5 * inner_width + half_t
        # Match the Isaac Gym curriculum asset exactly: one full roof and four
        # corner posts.  Continuous side walls used by the old baseline were a
        # stricter but different obstacle and therefore invalid for sim2sim.
        return [
            _box("frame_roof", (0.5 * length, 0.0, clearance + half_t),
                 (0.5 * length, 0.5 * inner_width + beam_thickness, half_t)),
            _box("frame_entry_pos_y", (0.0, post_y, 0.5 * clearance),
                 (half_t, half_t, 0.5 * clearance)),
            _box("frame_entry_neg_y", (0.0, -post_y, 0.5 * clearance),
                 (half_t, half_t, 0.5 * clearance)),
            _box("frame_exit_pos_y", (length, post_y, 0.5 * clearance),
                 (half_t, half_t, 0.5 * clearance)),
            _box("frame_exit_neg_y", (length, -post_y, 0.5 * clearance),
                 (half_t, half_t, 0.5 * clearance)),
        ]
    if obstacle == "slalom":
        radius = 0.5 * float(spec["pole_diameter_m"])
        half_height = 0.5 * float(spec["pole_height_m"])
        return [
            (
                f'        <geom name="rc27_obstacle_pole_{index}" type="cylinder" '
                f'pos="{x:.6f} {y:.6f} {half_height:.6f}" '
                f'size="{radius:.6f} {half_height:.6f}" contype="1" '
                f'conaffinity="1" friction="{FRICTION}" rgba="{GREEN}" />'
            )
            for index, (x, y) in enumerate(SLALOM_POLES, start=1)
        ]
    raise ValueError(f"unsupported baseline obstacle: {obstacle}")


def low_frame_gate_overlap(
    lowers: np.ndarray,
    uppers: np.ndarray,
    frame_length_m: float,
    inner_width_m: float,
    gate_margin_m: float = 0.02,
) -> Tuple[bool, bool, float]:
    """Check only robot collision geoms currently overlapping the opening."""
    overlapping = (uppers[:, 0] >= 0.0) & (lowers[:, 0] <= frame_length_m)
    if not np.any(overlapping):
        return False, False, 0.0
    lateral_extent = float(np.max(np.maximum(
        np.abs(lowers[overlapping, 1]),
        np.abs(uppers[overlapping, 1]),
    )))
    opening_half_width = 0.5 * inner_width_m - gate_margin_m
    return True, lateral_extent >= opening_half_width, lateral_extent


def aggregate_contact_snapshots(snapshots: List[Dict], physics_dt: float) -> Dict:
    """Keep end-step foot contacts but retain every physics-step frame peak."""
    if not snapshots:
        raise ValueError("at least one physics-step contact snapshot is required")
    combined = dict(snapshots[-1])
    combined["obstacle_bodies"] = set().union(
        *(snapshot["obstacle_bodies"] for snapshot in snapshots))
    combined["obstacle_geoms"] = set().union(
        *(snapshot["obstacle_geoms"] for snapshot in snapshots))
    combined["max_obstacle_force_n"] = max(
        snapshot["max_obstacle_force_n"] for snapshot in snapshots)
    combined["max_contacting_robot_top_z"] = max(
        snapshot["max_contacting_robot_top_z"] for snapshot in snapshots)
    combined["obstacle_impulse_ns"] = physics_dt * sum(
        snapshot["total_obstacle_force_n"] for snapshot in snapshots)
    most_floor_feet = max(snapshots, key=lambda snapshot: len(snapshot["floor_feet"]))
    combined["max_simultaneous_floor_feet"] = set(most_floor_feet["floor_feet"])
    combined["surface_contact_points"] = [
        point for snapshot in snapshots
        for point in snapshot["surface_contact_points"]
    ]
    return combined


def build_scene_xml(
    obstacle: str, rules: Dict, low_frame_training_payload: bool = False,
    low_frame_isaac_dynamics: bool = False,
) -> Path:
    text = BASE_XML.read_text(encoding="utf-8")
    text, removed = re.subn(
        r'^\s*<geom name="sim_(?:stair|three_stair)_[^"]*"[^>]*/>\s*\n',
        "",
        text,
        flags=re.M,
    )
    if removed < 9:
        raise RuntimeError(f"expected stair placeholders in {BASE_XML}, removed {removed}")
    mesh_dir = (BASE_XML.parent / "../meshes").resolve()
    text, substitutions = re.subn(
        r'meshdir="[^"]*"', f'meshdir="{mesh_dir}"', text, count=1
    )
    if substitutions != 1:
        raise RuntimeError("meshdir attribute not found in base XML")
    if obstacle == "low_frame" and low_frame_training_payload:
        # The training URDF welds a 1 kg payload at the body origin.
        base_body = '<body name="body"'
        body_start = text.index(base_body)
        insert_at = text.index("\n", body_start) + 1
        payload = (
            '            <body name="rc27_training_payload" pos="0 0 0">\n'
            '                <inertial pos="0 0 0" mass="1.0" '
            'diaginertia="0.0001 0.0001 0.0001" />\n'
            '            </body>\n'
            '            <body name="rc27_training_imu" pos="0 0 0">\n'
            '                <inertial pos="0 0 0" mass="0.001" '
            'diaginertia="0.0001 0.0001 0.0001" />\n'
            '            </body>\n'
        )
        text = text[:insert_at] + payload + text[insert_at:]
    if obstacle == "low_frame" and low_frame_isaac_dynamics:
        rr_pattern = (
            r'(<body name="RR_calf"[^>]*>\s*)'
            r'<inertial[^>]+/>'
        )
        rr_inertial = (
            'pos="0.00126062 -0.00001358 -0.06701153" '
            'mass="0.23880158" '
            'fullinertia="0.00202452 0.00207123 0.00011991 '
            '0.00000000 0.00025658 0.00000027"'
        )
        text, rr_count = re.subn(
            rr_pattern, lambda match: match.group(1) +
            f'<inertial {rr_inertial} />', text, count=1)
        if rr_count != 1:
            raise RuntimeError("RR calf inertial not found in base XML")
    anchor = '<geom name="floor"'
    insert_at = text.index("\n", text.index(anchor)) + 1
    body = "\n".join(obstacle_geoms_xml(obstacle, rules))
    text = text[:insert_at] + body + "\n" + text[insert_at:]
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    path = SCENE_DIR / f"{obstacle}_{os.getpid()}_{time.time_ns()}.xml"
    path.write_text(text, encoding="utf-8")
    return path


class Robocon2027Sim(MybotV3Sim):
    def __init__(
        self,
        obstacle: str,
        rules: Dict,
        body_path: Path = None,
        adaptation_path: Path = None,
        low_frame_crouch_center: bool = False,
        low_frame_center_thigh: float = 1.40,
        low_frame_center_calf: float = -2.15,
        low_frame_zero_damping: bool = False,
        low_frame_training_payload: bool = False,
        low_frame_isaac_dynamics: bool = False,
        low_frame_damping_scale: float = 1.0,
    ):
        self.obstacle = obstacle
        self.rules = rules
        self.scene_path = build_scene_xml(
            obstacle, rules, low_frame_training_payload,
            low_frame_isaac_dynamics)
        self.scene_xml_sha256 = sha256_file(self.scene_path)
        config = load_config(DEFAULT_CONFIG)
        apply_policy_profile(config, "config")
        config = dict(config)
        config["mujoco_xml_relpath"] = str(self.scene_path)
        config["action_lag_steps"] = 0
        if obstacle == "low_frame":
            spec = rules["obstacles"]["low_frame"]
            config.update(
                obstacle_ahead_x=0.0,
                obstacle_ahead_y=0.0,
                obstacle_clearance_m=float(spec["inner_height_m"]),
                obstacle_thickness_m=0.05,
                obstacle_heading_rad=0.0,
                obstacle_ahead_valid=1.0,
            )
            if low_frame_crouch_center:
                crouch = [
                    0.10, low_frame_center_thigh, low_frame_center_calf,
                    -0.10, low_frame_center_thigh, low_frame_center_calf,
                    0.10, low_frame_center_thigh, low_frame_center_calf,
                    -0.10, low_frame_center_thigh, low_frame_center_calf,
                ]
                config["policy_dof_pos"] = crouch
                config["default_dof_pos"] = crouch
                config["standup_target_pos"] = crouch
                config["clip_actions"] = 4.5
        default_body, default_adaptation = MODELS[obstacle]
        body = Path(body_path) if body_path is not None else default_body
        adaptation = (
            Path(adaptation_path)
            if adaptation_path is not None else default_adaptation)
        for path in (body, adaptation):
            if not path.is_file():
                raise FileNotFoundError(path)
        super().__init__(config, body, adaptation, "stairs", DEFAULT_CONFIG.parents[2])
        self._configure_aligned_damped_dynamics()
        if obstacle == "low_frame":
            for dof_index in self.joint_qvel_idx:
                self.model.dof_damping[dof_index] *= low_frame_damping_scale
        if obstacle == "low_frame" and low_frame_isaac_dynamics:
            # Isaac Gym AssetOptions.armature adds 0.01 to every link's
            # rotational inertia, while its actual DOF armature is zero.
            for body_id in range(1, self.model.nbody):
                self.model.body_inertia[body_id, :] += 0.01
            for dof_index in self.joint_qvel_idx:
                self.model.dof_armature[dof_index] = 0.0
            # Inertia and DOF armature feed MuJoCo's precomputed inverse
            # weights; refresh them after mutating the compiled model.
            mujoco.mj_setConst(self.model, self.data)
        if obstacle == "low_frame" and low_frame_zero_damping:
            for dof_index in self.joint_qvel_idx:
                self.model.dof_damping[dof_index] = 0.0
        self.obstacle_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                "rc27_obstacle_"
            )
        }
        self.surface_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                "rc27_surface_"
            )
        }
        self.foot_body_by_name = {
            side: int(self.model.body(f"{side}_foot").id)
            for side in ("FL", "FR", "RL", "RR")
        }
        self.foot_name_by_body = {value: key for key, value in self.foot_body_by_name.items()}
        self.foot_geom_ids = {
            side: [
                geom_id
                for geom_id in range(self.model.ngeom)
                if int(self.model.geom_bodyid[geom_id]) == body_id
            ]
            for side, body_id in self.foot_body_by_name.items()
        }
        self.floor_geom_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )

    def _configure_terrain(self):
        active = []
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("rc27_surface_"):
                active.append(int(geom_id))
        return active

    def _configure_aligned_damped_dynamics(self):
        for joint_index, dof_index in enumerate(self.joint_qvel_idx):
            kind = joint_index % 3
            self.model.dof_damping[dof_index] = 1.0 if kind == 0 else 2.0
            self.model.dof_frictionloss[dof_index] = 0.3 if kind == 2 else 0.1
            self.model.dof_armature[dof_index] = 0.01

    def step_control(
        self, target: np.ndarray, external_force=None, on_substep=None
    ):
        if on_substep is None:
            return super().step_control(target, external_force)
        # Mirror MybotV3Sim.step_control so low-frame contacts can be sampled
        # after each 5 ms physics step, without changing other evaluators.
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
            self.data.xfrc_applied[self.base_body_id, :] = 0.0
            self.data.xfrc_applied[self.base_body_id, :3] = external_force
            mujoco.mj_step(self.model, self.data)
            on_substep(self)

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
        self.fell = (
            self.fell or float(self.data.qpos[2]) < fall_height
            or tilt > fall_tilt
        )

    def switch_policy(
        self,
        body_path: Path,
        adaptation_path: Path,
        policy_dof_pos: Sequence[float],
        clip_actions: float,
    ) -> None:
        """Hot-switch policy contracts without resetting physical state.

        The low-frame expert uses the 126/123 observation contract and a
        crouched action centre, while the general locomotion policy uses the
        legacy 119/116 contract and normal standing centre. Only recurrent
        policy state is reset here; qpos/qvel and contact state stay continuous.
        """
        body_path = Path(body_path)
        adaptation_path = Path(adaptation_path)
        for path in (body_path, adaptation_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        self.body = torch.jit.load(str(body_path), map_location="cpu").eval()
        self.adaptation = torch.jit.load(
            str(adaptation_path), map_location="cpu"
        ).eval()
        self._configure_model_observation_widths()
        self._validate_models()

        centre = np.asarray(policy_dof_pos, dtype=np.float32)
        if centre.shape != (NUM_JOINTS,):
            raise ValueError(
                f"policy_dof_pos must have shape ({NUM_JOINTS},), got {centre.shape}"
            )
        self.policy_dof_pos = centre.copy()
        self.config["policy_dof_pos"] = centre.tolist()
        self.config["clip_actions"] = float(clip_actions)
        self.policy_history = torch.zeros(
            (HISTORY_LENGTH, self.policy_obs_per_step), dtype=torch.float32
        )
        self.estimator_history = torch.zeros(
            (HISTORY_LENGTH, self.estimator_obs_per_step), dtype=torch.float32
        )
        self.last_actions = torch.zeros((1, NUM_JOINTS), dtype=torch.float32)
        self.action_lag_buffer = [
            np.zeros(NUM_JOINTS, dtype=np.float32)
            for _ in range(self.action_lag_steps + 1)
        ]

    def contact_snapshot(self) -> Dict:
        floor_feet = set()
        surface_feet = set()
        surface_contact_points = []
        obstacle_bodies = set()
        obstacle_geoms = set()
        max_obstacle_force_n = 0.0
        total_obstacle_force_n = 0.0
        max_contacting_robot_top_z = 0.0
        contact_force = np.zeros(6, dtype=np.float64)
        relevant = self.obstacle_geom_ids | self.surface_geom_ids
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            mujoco.mj_contactForce(self.model, self.data, index, contact_force)
            for static_geom, robot_geom in (
                (int(contact.geom1), int(contact.geom2)),
                (int(contact.geom2), int(contact.geom1)),
            ):
                body_id = int(self.model.geom_bodyid[robot_geom])
                foot_name = self.foot_name_by_body.get(body_id)
                if static_geom == self.floor_geom_id and foot_name:
                    floor_feet.add(foot_name)
                if static_geom in self.surface_geom_ids and foot_name:
                    surface_feet.add(foot_name)
                    surface_contact_points.append({
                        "foot": foot_name,
                        "xyz": [float(value) for value in contact.pos],
                    })
                if static_geom in relevant and body_id != 0:
                    body_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    )
                    obstacle_bodies.add(body_name or f"body_{body_id}")
                    static_name = mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, static_geom)
                    obstacle_geoms.add(
                        static_name or f"geom_{static_geom}")
                    robot_type = int(self.model.geom_type[robot_geom])
                    if robot_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                        rotation = np.asarray(
                            self.data.geom_xmat[robot_geom], dtype=np.float64
                        ).reshape(3, 3)
                        half_size = np.asarray(
                            self.model.geom_size[robot_geom], dtype=np.float64)
                        top_z = float(
                            self.data.geom_xpos[robot_geom, 2]
                            + np.abs(rotation[2]) @ half_size)
                        max_contacting_robot_top_z = max(
                            max_contacting_robot_top_z, top_z)
                    normal_force_n = abs(float(contact_force[0]))
                    max_obstacle_force_n = max(
                        max_obstacle_force_n, normal_force_n
                    )
                    total_obstacle_force_n += normal_force_n
        return {
            "floor_feet": floor_feet,
            "surface_feet": surface_feet,
            "surface_contact_points": surface_contact_points,
            "max_simultaneous_floor_feet": set(floor_feet),
            "obstacle_bodies": obstacle_bodies,
            "obstacle_geoms": obstacle_geoms,
            "max_obstacle_force_n": max_obstacle_force_n,
            "total_obstacle_force_n": total_obstacle_force_n,
            "max_contacting_robot_top_z": max_contacting_robot_top_z,
        }

    def stance_points(self) -> np.ndarray:
        """Base centre plus the four foot centres, in the world XY plane."""
        points = np.zeros((5, 2), dtype=float)
        points[0] = self.data.qpos[:2]
        for index, side in enumerate(("FL", "FR", "RL", "RR"), start=1):
            geom_ids = self.foot_geom_ids[side]
            points[index] = np.mean(self.data.geom_xpos[geom_ids][:, :2], axis=0)
        return points

    def robot_collision_aabb(
        self, per_geom: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """World AABB over active robot collision geoms.

        This mirrors the Isaac low-frame evaluator's whole-robot pass test;
        using base x alone gives different answers for different postures.
        """
        lowers = []
        uppers = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) == 0:
                continue
            if (int(self.model.geom_contype[geom_id]) == 0 and
                    int(self.model.geom_conaffinity[geom_id]) == 0):
                continue
            center = np.asarray(
                self.data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(
                self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            size = np.asarray(
                self.model.geom_size[geom_id], dtype=np.float64)
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                extent = np.abs(rotation) @ size
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                extent = np.full(3, size[0], dtype=np.float64)
            else:
                # Conservative enclosing sphere for capsules/cylinders.  The
                # current robot collision model uses boxes, but this keeps the
                # evaluator safe if a later URDF export changes a foot geom.
                radius = float(size[0] + (size[1] if size.size > 1 else 0.0))
                extent = np.full(3, radius, dtype=np.float64)
            lowers.append(center - extent)
            uppers.append(center + extent)
        if not lowers:
            raise RuntimeError("robot has no active collision geometry")
        lower_array = np.stack(lowers)
        upper_array = np.stack(uppers)
        if per_geom:
            return lower_array, upper_array
        return np.min(lower_array, axis=0), np.max(upper_array, axis=0)

    def pole_clearance(self, poles: Sequence[Tuple[float, float]], pole_radius: float) -> float:
        """Smallest foot-to-post clearance in the current step (metres)."""
        feet = self.stance_points()[1:]
        best = float("inf")
        for pole_x, pole_y in poles:
            distance = float(np.min(np.linalg.norm(feet - np.array([pole_x, pole_y]), axis=1)))
            best = min(best, distance - pole_radius)
        return best

    def close(self):
        try:
            self.scene_path.unlink()
        except FileNotFoundError:
            pass


def yaw_from_qpos(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3:7]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def convex_hull(points: np.ndarray) -> List[Tuple[float, float]]:
    """Andrew monotone chain; enough for a five-point stance polygon."""
    unique = sorted({(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points})
    if len(unique) <= 2:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: List[Tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    sign = 0
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        value = (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)
        if abs(value) < 1e-9:
            continue
        current = 1 if value > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True


def resample_path(points: Sequence[Tuple[float, float]], step: float = 0.02) -> np.ndarray:
    """Centripetal Catmull-Rom through the waypoints, sampled every ``step``."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    extended = np.vstack([2.0 * pts[0] - pts[1], pts, 2.0 * pts[-1] - pts[-2]])
    chunks = []
    for index in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[index - 1], extended[index], extended[index + 1], extended[index + 2]
        count = max(2, int(math.ceil(float(np.linalg.norm(p2 - p1)) / step)))
        t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        chunk = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t ** 3
        )
        chunks.append(chunk)
    return np.vstack(chunks + [pts[-1][None, :]])


def waypoint_command(sim: Robocon2027Sim, target: Tuple[float, float], max_vx: float) -> np.ndarray:
    x, y = (float(value) for value in sim.data.qpos[:2])
    yaw = yaw_from_qpos(np.asarray(sim.data.qpos))
    dx, dy = target[0] - x, target[1] - y
    cy, sy = math.cos(yaw), math.sin(yaw)
    body_x = cy * dx + sy * dy
    body_y = -sy * dx + cy * dy
    heading_error = math.atan2(body_y, max(body_x, 1e-6))
    return np.array(
        [
            np.clip(0.9 * body_x, -0.15, max_vx),
            np.clip(0.9 * body_y, -0.25, 0.25),
            np.clip(1.8 * heading_error, -0.8, 0.8),
        ],
        dtype=np.float32,
    )


def pursuit_command(
    sim: Robocon2027Sim,
    path: np.ndarray,
    index: int,
    params: Dict,
) -> Tuple[np.ndarray, int]:
    """Tangent + cross-track path tracking.

    A classic pure-pursuit look-ahead cuts the inside of every bend by roughly
    ``L**2 * curvature / 2``.  The slalom weave has a curvature near 5 1/m at
    its peaks, so a 0.35 m look-ahead lagged the reference by over 0.3 m and the
    feet clipped the posts.  Instead, feed the path tangent through and correct
    the signed cross-track error directly, which has no look-ahead lag.
    """
    xy = np.asarray(sim.data.qpos[:2], dtype=float)
    yaw = yaw_from_qpos(np.asarray(sim.data.qpos))
    # Global nearest point: a forward-only window that also advanced ``index``
    # created positive feedback and sent the index running away to the goal.
    local = int(np.argmin(np.linalg.norm(path - xy, axis=1)))
    tangent_index = min(
        local + int(round(float(params.get("lookahead_m", 0.0)) / 0.02)),
        len(path) - 1,
    )
    low = max(tangent_index - 1, 0)
    high = min(tangent_index + 1, len(path) - 1)
    if high <= low:
        high = min(low + 1, len(path) - 1)
    tangent = path[high] - path[low]
    tangent_norm = float(np.linalg.norm(tangent))
    tangent = tangent / tangent_norm if tangent_norm > 1e-9 else np.array([1.0, 0.0])
    local_low = max(local - 1, 0)
    local_high = min(local + 1, len(path) - 1)
    local_tangent = path[local_high] - path[local_low]
    local_tangent /= max(float(np.linalg.norm(local_tangent)), 1e-9)
    normal = np.array([-local_tangent[1], local_tangent[0]])
    cross_track = float((xy - path[local]) @ normal)

    # Point the body along the path tangent and crab sideways to remove the
    # cross-track error.  Deriving the heading from the corrected velocity
    # instead made the robot under-rotate on the weave, so its swing foot
    # (0.43 m reach, only 0.11 m apex) swept straight through the posts.
    desired_heading = math.atan2(float(tangent[1]), float(tangent[0]))
    heading_error = (desired_heading - yaw + math.pi) % (2.0 * math.pi) - math.pi
    speed = float(params["speed"])
    speed_scale = 1.0 / (
        1.0 + float(params.get("heading_slow", 3.0)) * abs(heading_error))
    speed_scale /= 1.0 + float(params.get("k_slow", 0.0)) * abs(cross_track)
    curvature_gain = float(params.get("curvature_slow", 0.0))
    if curvature_gain:
        back = max(local - 5, 0)
        ahead = min(local + 5, len(path) - 1)
        v1 = path[local] - path[back]
        v2 = path[ahead] - path[local]
        if np.linalg.norm(v1) > 1e-9 and np.linalg.norm(v2) > 1e-9:
            bend = math.atan2(
                float(v1[0] * v2[1] - v1[1] * v2[0]),
                float(v1 @ v2),
            )
            arc = 0.5 * (float(np.linalg.norm(v1)) + float(np.linalg.norm(v2)))
            speed_scale /= 1.0 + curvature_gain * abs(bend) / max(arc, 1e-6)
    command = np.array(
        [
            float(np.clip(speed * speed_scale, 0.0, speed)),
            float(np.clip(-params["k_cross"] * cross_track, -params["vy_max"], params["vy_max"])),
            float(np.clip(params["k_yaw"] * heading_error, -params["yaw_max"], params["yaw_max"])),
        ],
        dtype=np.float32,
    )
    return command, local


def slalom_projected_passes(
    before: Tuple[float, float], after: Tuple[float, float],
) -> List[Tuple[int, str, float]]:
    """Forward projection crossings: x for poles 1-4, y for poles 5-6."""
    passes = []
    for index, (pole_x, pole_y) in enumerate(SLALOM_POLES, start=1):
        axis = 0 if index <= 4 else 1
        line = pole_x if axis == 0 else pole_y
        first, last = float(before[axis]), float(after[axis])
        if first <= line < last:
            fraction = (line - first) / (last - first)
            crossing = np.asarray(before) + fraction * (
                np.asarray(after) - np.asarray(before))
            offset = float(crossing[1] - pole_y) if axis == 0 else float(
                crossing[0] - pole_x)
            side = "left" if offset > 0.05 else ("right" if offset < -0.05 else "center")
            passes.append((index, side, offset))
    return passes


def stand(sim: Robocon2027Sim, seconds: float = 1.0) -> np.ndarray:
    motion_kp = sim.kp.copy()
    motion_kd = sim.kd.copy()
    sim.kp[:] = 2.0 * motion_kp
    sim.kd[:] = 2.0 * motion_kd
    target = sim.policy_dof_pos.copy()
    desired = (
        sim.policy_dof_pos
        if sim.obstacle == "low_frame"
        else np.clip(sim.policy_dof_pos + STAND_OFFSET, sim.lower, sim.upper)
    )
    for _ in range(max(1, int(seconds / sim.control_dt))):
        target += 0.12 * (desired - target)
        sim.step_control(target)
    sim.kp[:] = motion_kp
    sim.kd[:] = motion_kd
    sim.policy_history.zero_()
    sim.estimator_history.zero_()
    sim.last_actions.zero_()
    return target


def _classify(obstacle: str, success_rate: float) -> str:
    if success_rate >= 0.90:
        return "direct_reuse_candidate"
    if success_rate >= 0.50:
        return "fine_tune_candidate"
    return {
        "slalom": "high_level_planner_tuning_required",
        "beam": "bridge_controller_or_fine_tune_required",
        "low_frame": "expert_fine_tune_required",
    }[obstacle]


def run_trial(
    obstacle: str,
    rules: Dict,
    max_seconds: float,
    controller: str = "pursuit",
    spawn_perturbation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    body_path: Path = None,
    adaptation_path: Path = None,
    low_frame_crouch_center: bool = False,
    violent_contact_force_threshold_n: float = 250.0,
    low_frame_center_thigh: float = 1.40,
    low_frame_center_calf: float = -2.15,
    low_frame_openloop_recovery: bool = False,
    low_frame_recovery_duration_s: float = 1.0,
    low_frame_locomotion_recovery: bool = False,
    low_frame_locomotion_blend_duration_s: float = 0.0,
    low_frame_locomotion_switch_delay_s: float = 0.0,
    low_frame_nominal_init: bool = False,
    low_frame_straight_command: bool = False,
    low_frame_zero_damping: bool = False,
    low_frame_unclipped_expert_target: bool = False,
    low_frame_training_payload: bool = False,
    low_frame_isaac_dynamics: bool = False,
    low_frame_isaac_reset: bool = False,
    low_frame_crouch_residual_scale: float = 0.0,
    low_frame_dynamics_trace: bool = False,
    low_frame_damping_scale: float = 1.0,
    recovery_body_path: Path = None,
    recovery_adaptation_path: Path = None,
    beam_expert_body_path: Path = None,
    beam_expert_adaptation_path: Path = None,
    beam_switch_blend_duration_s: float = 0.3,
    beam_finish_extra_m: float = None,
    frame_callback=None,
) -> Dict:
    beam_modes_enabled = (beam_expert_body_path is not None or
                          beam_expert_adaptation_path is not None)
    if beam_modes_enabled and (obstacle != "beam" or
                               beam_expert_body_path is None or
                               beam_expert_adaptation_path is None):
        raise ValueError("beam mode switching needs both expert JIT files and obstacle=beam")
    if beam_switch_blend_duration_s < 0:
        raise ValueError("beam_switch_blend_duration_s must be nonnegative")
    if beam_finish_extra_m is not None and beam_finish_extra_m < 0.25:
        raise ValueError("beam_finish_extra_m must be at least 0.25 m")
    sim = Robocon2027Sim(
        obstacle, rules, body_path, adaptation_path,
        low_frame_crouch_center=low_frame_crouch_center,
        low_frame_center_thigh=low_frame_center_thigh,
        low_frame_center_calf=low_frame_center_calf,
        low_frame_zero_damping=low_frame_zero_damping,
        low_frame_training_payload=low_frame_training_payload,
        low_frame_isaac_dynamics=low_frame_isaac_dynamics,
        low_frame_damping_scale=low_frame_damping_scale)
    try:
        spawn_x, spawn_y, spawn_yaw = SPAWNS[obstacle]
        spawn_x += spawn_perturbation[0]
        spawn_y += spawn_perturbation[1]
        spawn_yaw += spawn_perturbation[2]
        sim.teleport(spawn_x, spawn_y, spawn_yaw)
        if obstacle == "low_frame" and low_frame_crouch_center:
            sim.data.qpos[2] = 0.21
            mujoco.mj_forward(sim.model, sim.data)
        last_target = sim.policy_dof_pos.copy()
        if not (obstacle == "low_frame" and low_frame_nominal_init):
            last_target = stand(sim)
        if obstacle == "low_frame" and low_frame_isaac_reset:
            # Isaac's env.reset() takes one zero-action control step, then
            # HistoryWrapper inserts that first observation into empty history.
            sim.step_control(sim.policy_dof_pos.copy())
            policy_step, estimator_step = sim._observations(
                np.array([PURSUIT["low_frame"]["speed"], 0.0, 0.0],
                         dtype=np.float32))
            sim.policy_history = torch.cat(
                [sim.policy_history[1:], policy_step], dim=0)
            sim.estimator_history = torch.cat(
                [sim.estimator_history[1:], estimator_step], dim=0)

        initial_policy_obs = (
            sim.policy_history.reshape(-1).detach().cpu().numpy().tolist()
            if obstacle == "low_frame" and low_frame_dynamics_trace else None)
        initial_estimator_obs = (
            sim.estimator_history.reshape(-1).detach().cpu().numpy().tolist()
            if obstacle == "low_frame" and low_frame_dynamics_trace else None)

        attempt = ObstacleAttempt(obstacle)
        pole_radius = 0.5 * float(rules["obstacles"]["slalom"]["pole_diameter_m"])
        raw_path = PATH_BY_OBSTACLE[obstacle]
        if obstacle == "beam":
            beam_length = float(rules["obstacles"]["beam"]["length_m"])
            beam_entry_x = min(0.55, 0.25 * beam_length)
            beam_entry_stage_end_x = min(1.0, 0.35 * beam_length)
            beam_active_end_x = beam_length - 0.25
        path = resample_path(raw_path) if controller == "pursuit" else np.asarray(raw_path, dtype=float)
        params = PURSUIT[obstacle]
        index = 0
        max_contact_force_n = 0.0
        min_pole_clearance_m = float("inf")
        max_centreline_error_m = 0.0
        bridge_active = False
        bridge_min_edge_margin_m = float("inf")
        bridge_min_edge_margin_by_foot = {}
        bridge_min_support_center_margin_m = float("inf")
        bridge_min_support_center_margin_by_foot = {}
        bridge_foot_contact_trace = []
        bridge_first_outside_ground_x_m = None
        contact_steps = 0
        contact_events: List[Dict] = []
        completion_time = None
        crossed_step = None
        entered_gate = False
        outside_gate = False
        max_gate_lateral_extent_m = 0.0
        locomotion_switch_step = None
        recovery_stable_steps = 0
        recovery_start_target = None
        locomotion_recovery_switched = False
        recovery_handoff = None
        recovery_trace: List[List[float]] = []
        max_recovery_stable_steps = 0
        expert_target_clipped_steps = 0
        expert_target_clipped_joints = 0
        expert_target_max_overrun_rad = 0.0
        expert_target_clip_by_joint = np.zeros(NUM_JOINTS, dtype=np.int32)
        recovery_stand_target = np.array(
            [
                0.10, 0.80, -1.50, -0.10, 0.80, -1.50,
                0.10, 0.80, -1.50, -0.10, 0.80, -1.50,
            ],
            dtype=np.float32,
        )
        trajectory: List[List[float]] = []
        dynamics_trace: List[Dict] = []
        beam_mode = "locomotion"
        beam_switch_events: List[Dict] = []
        beam_blend_start_step = None
        beam_blend_start_target = None
        beam_exit_confirmed = False
        beam_locomotion_config = None
        if beam_modes_enabled:
            beam_locomotion_config = load_config(DEFAULT_CONFIG)
            apply_policy_profile(beam_locomotion_config, "config")
        finish_zone = SLALOM_ZONES["finish"]
        steps = max(1, int(max_seconds / sim.control_dt))
        for step in range(steps):
            x, y, z = (float(value) for value in sim.data.qpos[:3])
            if controller == "pursuit":
                command, index = pursuit_command(sim, path, index, params)
            else:
                target_xy = path[min(index, len(path) - 1)]
                tolerance = 0.20 if obstacle != "slalom" else 0.15
                if math.hypot(target_xy[0] - x, target_xy[1] - y) <= tolerance and index < len(path) - 1:
                    index += 1
                    target_xy = path[index]
                command = waypoint_command(sim, tuple(target_xy), max_vx=params["speed"])

            if obstacle == "low_frame" and low_frame_straight_command:
                command = np.array(
                    [PURSUIT["low_frame"]["speed"], 0.0, 0.0],
                    dtype=np.float32)
            if obstacle == "low_frame" and crossed_step is not None:
                command = np.zeros(3, dtype=np.float32)

            if beam_modes_enabled:
                next_mode = None
                if beam_mode == "locomotion" and not beam_switch_events and x >= -0.15:
                    next_mode = "beam_expert"
                elif beam_mode == "beam_expert" and x >= beam_length + 0.30:
                    next_mode = "locomotion"
                if next_mode is not None:
                    model_paths = (
                        (beam_expert_body_path, beam_expert_adaptation_path)
                        if next_mode == "beam_expert" else
                        (body_path or MODELS["beam"][0],
                         adaptation_path or MODELS["beam"][1]))
                    sim.switch_policy(
                        *model_paths,
                        beam_locomotion_config["policy_dof_pos"],
                        float(beam_locomotion_config["clip_actions"]),
                    )
                    beam_mode = next_mode
                    beam_blend_start_step = step
                    beam_blend_start_target = np.asarray(last_target).copy()
                    beam_switch_events.append({
                        "mode": next_mode,
                        "time_s": round(step * sim.control_dt, 3),
                        "base_x_m": round(x, 4),
                        "base_y_m": round(y, 4),
                        "start_target_rad": [round(float(v), 4) for v in last_target],
                    })

            action = sim.infer(command)
            if (obstacle == "low_frame" and crossed_step is None and
                    low_frame_crouch_residual_scale > 0.0):
                # Same smooth diagnostic residual as eval_low_bar_isaac.py:
                # reach a 1.60 rad thigh target before the front enters,
                # then release it after the rear has left the frame.
                frame_length = float(rules["obstacles"]["low_frame"]["length_m"])
                ramp_in = np.clip((x + 0.80) / 0.40, 0.0, 1.0)
                ramp_out = np.clip(
                    (frame_length + 0.70 - x) / 0.35, 0.0, 1.0)
                phase = float(min(ramp_in, ramp_out))
                action = action.copy()
                action[[1, 4, 7, 10]] += (
                    low_frame_crouch_residual_scale * phase *
                    (1.60 - low_frame_center_thigh) /
                    float(sim.config["action_scale"]))
                action = np.clip(action, -4.5, 4.5)
                sim.last_actions = torch.from_numpy(action).view(1, -1)
            raw_target = sim.target_from_action(
                action, float(sim.config["action_scale"]))
            clipped_target = np.clip(raw_target, sim.lower, sim.upper)
            if obstacle == "low_frame" and not locomotion_recovery_switched:
                clipped = np.abs(raw_target - clipped_target) > 1e-6
                expert_target_clipped_steps += int(np.any(clipped))
                expert_target_clipped_joints += int(np.count_nonzero(clipped))
                expert_target_clip_by_joint += clipped.astype(np.int32)
                expert_target_max_overrun_rad = max(
                    expert_target_max_overrun_rad,
                    float(np.abs(raw_target - clipped_target).max()))
            target = (
                raw_target if (obstacle == "low_frame" and
                               low_frame_unclipped_expert_target and
                               not locomotion_recovery_switched)
                else clipped_target)
            if beam_modes_enabled and beam_blend_start_step is not None:
                elapsed = (step - beam_blend_start_step + 1) * sim.control_dt
                alpha = (min(1.0, elapsed / beam_switch_blend_duration_s)
                         if beam_switch_blend_duration_s > 0 else 1.0)
                if step == beam_blend_start_step:
                    beam_switch_events[-1]["raw_target_jump_max_rad"] = round(
                        float(np.max(np.abs(target - beam_blend_start_target))), 4)
                target = np.clip(
                    (1.0 - alpha) * beam_blend_start_target + alpha * target,
                    sim.lower, sim.upper)
                if step == beam_blend_start_step:
                    beam_switch_events[-1]["applied_target_jump_max_rad"] = round(
                        float(np.max(np.abs(target - beam_blend_start_target))), 4)
                if alpha >= 1.0:
                    beam_blend_start_step = None
            if (obstacle == "low_frame" and locomotion_recovery_switched and
                    locomotion_switch_step is not None):
                if step == locomotion_switch_step + 1:
                    raw_jump = np.asarray(target) - recovery_start_target
                    recovery_handoff["raw_target_jump_max_rad"] = round(
                        float(np.abs(raw_jump).max()), 4)
                    recovery_handoff["raw_target_jump_l2_rad"] = round(
                        float(np.linalg.norm(raw_jump)), 4)
                    recovery_handoff["first_locomotion_target_rad"] = [
                        round(float(value), 3) for value in target]
                if low_frame_locomotion_blend_duration_s > 0.0:
                    alpha = np.clip(
                        (step - locomotion_switch_step) * sim.control_dt
                        / low_frame_locomotion_blend_duration_s,
                        0.0, 1.0)
                    blended_target = (
                        (1.0 - alpha) * recovery_start_target + alpha * target)
                    target = (
                        blended_target if low_frame_unclipped_expert_target
                        else np.clip(blended_target, sim.lower, sim.upper))
                if step == locomotion_switch_step + 1:
                    applied_jump = np.asarray(target) - recovery_start_target
                    recovery_handoff["applied_target_jump_max_rad"] = round(
                        float(np.abs(applied_jump).max()), 4)
            if (obstacle == "low_frame" and low_frame_openloop_recovery and
                    recovery_start_target is not None):
                alpha = np.clip(
                    (step - crossed_step) * sim.control_dt
                    / low_frame_recovery_duration_s,
                    0.0, 1.0)
                target = np.clip(
                    (1.0 - alpha) * recovery_start_target
                    + alpha * recovery_stand_target,
                    sim.lower,
                    sim.upper,
                )
            if obstacle == "low_frame" and low_frame_dynamics_trace:
                pre_q = sim.joint_pos().copy()
                pre_dq = sim.joint_vel().copy()
                first_torque = np.clip(
                    sim.kp * (target - pre_q) - sim.kd * pre_dq,
                    -sim.torque_limits, sim.torque_limits)
            if obstacle in {"low_frame", "slalom", "beam"}:
                substep_contacts = []
                sim.step_control(
                    target,
                    on_substep=lambda current: substep_contacts.append(
                        current.contact_snapshot()),
                )
                contacts = aggregate_contact_snapshots(
                    substep_contacts, float(sim.model.opt.timestep))
            else:
                sim.step_control(target)
                contacts = sim.contact_snapshot()
            last_target = np.asarray(target).copy()
            if frame_callback is not None:
                frame_callback(sim, step + 1)
            if obstacle == "low_frame" and low_frame_dynamics_trace:
                dynamics_trace.append({
                    "time_s": round((step + 1) * sim.control_dt, 4),
                    "base_xyz": sim.data.qpos[:3].tolist(),
                    "base_quat_wxyz": sim.data.qpos[3:7].tolist(),
                    "joint_pos_rad": sim.joint_pos().tolist(),
                    "joint_vel_rad_s": sim.joint_vel().tolist(),
                    "pd_target_rad": np.asarray(target).tolist(),
                    "first_substep_torque_nm": first_torque.tolist(),
                })
            max_contact_force_n = max(max_contact_force_n, contacts["max_obstacle_force_n"])
            contact_steps += int(bool(contacts["obstacle_bodies"]))
            if contacts["obstacle_bodies"] and len(contact_events) < 20:
                contact_events.append(
                    {
                        "time_s": round((step + 1) * sim.control_dt, 3),
                        "base_xy": [round(x, 3), round(y, 3)],
                        "base_z": round(z, 3),
                        "bodies": sorted(contacts["obstacle_bodies"]),
                        "frame_parts": sorted(contacts["obstacle_geoms"]),
                        "contacting_robot_top_z": round(
                            contacts["max_contacting_robot_top_z"], 4),
                        "force_n": round(float(contacts["max_obstacle_force_n"]), 3),
                    }
                )

            pre_x, pre_y = x, y
            x, y, z = (float(value) for value in sim.data.qpos[:3])
            if step % 25 == 0:
                trajectory.append(
                    [round((step + 1) * sim.control_dt, 2), round(x, 3), round(y, 3), round(z, 3)]
                )
            if (obstacle == "low_frame" and
                    contacts["max_obstacle_force_n"] > violent_contact_force_threshold_n):
                attempt.mark("violent_collision")
                break
            if sim.fell:
                attempt.mark("fell")
                break

            if obstacle == "slalom":
                if contacts["obstacle_bodies"]:
                    attempt.mark("pole_contact")
                zone_radius = 0.5 * float(
                    rules["obstacles"]["slalom"]["required_zone_diameter_m"])
                in_zone = {
                    name: math.hypot(x - centre[0], y - centre[1]) <= zone_radius
                    for name, centre in SLALOM_ZONES.items()
                }
                if in_zone["start"]:
                    attempt.reach_zone("start")
                projected = slalom_projected_passes((pre_x, pre_y), (x, y))
                for pole_index, side, offset in projected:
                    if pole_index == 5 and in_zone["corner"]:
                        attempt.reach_zone("corner")
                    attempt.pass_slalom_pole(pole_index, side, offset)
                if in_zone["corner"]:
                    attempt.reach_zone("corner")
                if in_zone["finish"]:
                    attempt.reach_zone("finish")
                min_pole_clearance_m = min(
                    min_pole_clearance_m, sim.pole_clearance(SLALOM_POLES, pole_radius)
                )
                hull = convex_hull(sim.stance_points())
                if any(point_in_polygon(pole, hull) for pole in SLALOM_POLES):
                    attempt.mark("pole_overflight")
                reached = len(attempt.slalom_events) == 9
                if reached and math.hypot(x - finish_zone[0], y - finish_zone[1]) <= zone_radius:
                    attempt.mark("crossed")
                    completion_time = (step + 1) * sim.control_dt
                    break
            elif obstacle == "beam":
                half_width = 0.5 * float(rules["obstacles"]["beam"]["width_m"])
                per_foot = {}
                for point in contacts["surface_contact_points"]:
                    foot = point["foot"]
                    contact_x, contact_y, _ = point["xyz"]
                    margin = half_width - abs(contact_y)
                    bridge_min_edge_margin_m = min(bridge_min_edge_margin_m, margin)
                    bridge_min_edge_margin_by_foot[foot] = min(
                        bridge_min_edge_margin_by_foot.get(foot, float("inf")), margin)
                    if foot not in per_foot or margin < per_foot[foot]["margin_m"]:
                        per_foot[foot] = {
                            "x_m": round(contact_x, 4),
                            "y_m": round(contact_y, 4),
                            "margin_m": round(margin, 4),
                        }
                foot_centres = sim.stance_points()[1:]
                for foot, centre in zip(("FL", "FR", "RL", "RR"), foot_centres):
                    if foot not in contacts["surface_feet"]:
                        continue
                    centre_margin = half_width - abs(float(centre[1]))
                    bridge_min_support_center_margin_m = min(
                        bridge_min_support_center_margin_m, centre_margin)
                    bridge_min_support_center_margin_by_foot[foot] = min(
                        bridge_min_support_center_margin_by_foot.get(
                            foot, float("inf")), centre_margin)
                    if foot in per_foot:
                        per_foot[foot]["foot_center_y_m"] = round(
                            float(centre[1]), 4)
                        per_foot[foot]["foot_center_margin_m"] = round(
                            centre_margin, 4)
                if step % 5 == 0 and per_foot:
                    bridge_foot_contact_trace.append({
                        "time_s": round((step + 1) * sim.control_dt, 3),
                        "base_x_m": round(x, 4),
                        "feet": per_foot,
                    })
                if x >= beam_entry_x and len(contacts["surface_feet"]) >= 2:
                    bridge_active = True
                    attempt.mark("entered_short_side")
                if bridge_active and 0.25 <= x <= beam_active_end_x:
                    max_centreline_error_m = max(max_centreline_error_m, abs(y))
                    attempt.record_outside_ground_contacts(
                        contacts["max_simultaneous_floor_feet"])
                    if attempt.outside_ground_failure:
                        bridge_first_outside_ground_x_m = x
                        break
                if x >= beam_length + 0.25:
                    attempt.mark("exited_short_side")
                    beam_exit_confirmed = True
                finish_extra = (beam_finish_extra_m if beam_finish_extra_m is not None
                                else (0.45 if beam_modes_enabled else 0.25))
                if x >= beam_length + finish_extra:
                    attempt.mark("crossed")
                    completion_time = (step + 1) * sim.control_dt
                    break
            else:
                if contacts["obstacle_bodies"]:
                    attempt.record_frame_contact(contacts["obstacle_impulse_ns"])
                spec = rules["obstacles"]["low_frame"]
                frame_length = float(spec["length_m"])
                geom_lowers, geom_uppers = sim.robot_collision_aabb(
                    per_geom=True)
                overlaps, bypassed, lateral_extent = low_frame_gate_overlap(
                    geom_lowers, geom_uppers, frame_length,
                    float(spec["inner_width_m"]))
                if overlaps:
                    max_gate_lateral_extent_m = max(
                        max_gate_lateral_extent_m, lateral_extent)
                    if bypassed:
                        outside_gate = True
                        attempt.mark("outside_gate")
                        break
                    entered_gate = True
                if (crossed_step is None and
                        float(np.min(geom_lowers[:, 0])) > frame_length + 0.05):
                    if not entered_gate:
                        outside_gate = True
                        attempt.mark("outside_gate")
                        break
                    crossed_step = step
                    recovery_start_target = np.asarray(target).copy()
                if (crossed_step is not None and low_frame_locomotion_recovery
                        and not locomotion_recovery_switched
                        and (step - crossed_step) * sim.control_dt
                        >= low_frame_locomotion_switch_delay_s):
                    locomotion_switch_step = step
                    recovery_stable_steps = 0
                    max_recovery_stable_steps = 0
                    recovery_start_target = np.asarray(target).copy()
                    recovery_handoff = {
                        "switch_time_s": round((step + 1) * sim.control_dt, 3),
                        "base_z_at_switch_m": round(z, 4),
                        "command_at_switch": [
                            round(float(value), 4) for value in command],
                        "base_qvel_at_switch": [
                            round(float(value), 4) for value in sim.data.qvel[:6]],
                        "last_expert_target_rad": [
                            round(float(value), 3)
                            for value in recovery_start_target],
                        "joint_position_at_switch_rad": [
                            round(float(value), 3)
                            for value in sim.joint_pos()],
                    }
                    recovery_config = load_config(DEFAULT_CONFIG)
                    apply_policy_profile(recovery_config, "config")
                    sim.switch_policy(
                        recovery_body_path or (
                            LOCOMOTION_CKPT / "body_059950.jit"),
                        recovery_adaptation_path or (
                            LOCOMOTION_CKPT
                            / "adaptation_module_059950.jit"),
                        recovery_config["policy_dof_pos"],
                        float(recovery_config["clip_actions"]),
                    )
                    locomotion_recovery_switched = True
                if crossed_step is not None:
                    w, qx, qy, qz = (float(v) for v in sim.data.qpos[3:7])
                    roll = math.atan2(
                        2.0 * (w * qx + qy * qz),
                        1.0 - 2.0 * (qx * qx + qy * qy),
                    )
                    pitch = math.asin(np.clip(
                        2.0 * (w * qy - qz * qx), -1.0, 1.0))
                    stable = (
                        z > 0.16 and max(abs(roll), abs(pitch)) < 0.60
                        and (not low_frame_locomotion_recovery or
                             (locomotion_recovery_switched and
                              step > locomotion_switch_step)))
                    recovery_stable_steps = (
                        recovery_stable_steps + 1 if stable else 0)
                    max_recovery_stable_steps = max(
                        max_recovery_stable_steps, recovery_stable_steps)
                    if ((step - crossed_step) % 5 == 0 and
                            len(recovery_trace) < 30):
                        recovery_trace.append([
                            round((step - crossed_step) * sim.control_dt, 3),
                            round(z, 4), round(math.degrees(roll), 2),
                            round(math.degrees(pitch), 2),
                        ])
                    if (recovery_stable_steps >= int(round(1.0 / sim.control_dt))
                            and (not low_frame_locomotion_recovery
                                 or locomotion_recovery_switched)):
                        attempt.mark("crossed")
                        completion_time = (step + 1) * sim.control_dt
                        break

        attempt.elapsed_s = completion_time
        result = attempt.finalize()
        summary = sim.summary()
        return {
            "success": result.passed and not outside_gate,
            "reason": "outside_gate" if outside_gate else result.reason,
            "controller": controller,
            "scene_xml_sha256": sim.scene_xml_sha256,
            "spawn": {
                "x_m": round(spawn_x, 5),
                "y_m": round(spawn_y, 5),
                "yaw_deg": round(math.degrees(spawn_yaw), 4),
            },
            "completion_time_s": completion_time,
            "final_base_xyz": [round(float(value), 4) for value in sim.data.qpos[:3]],
            "fell": bool(summary["fell"]),
            "max_tilt_deg": round(float(summary["max_tilt_deg"]), 3),
            "mean_abs_action": round(float(summary["mean_abs_action"]), 5),
            "action_saturation_fraction": round(float(summary["action_saturation_fraction"]), 5),
            "frame_or_obstacle_contact_steps": contact_steps,
            "max_obstacle_contact_force_n": round(max_contact_force_n, 3),
            "contact_force_sample_dt_s": (
                float(sim.model.opt.timestep) if obstacle in {"low_frame", "slalom", "beam"}
                else sim.control_dt),
            "violent_contact_force_threshold_n": (
                violent_contact_force_threshold_n),
            "recovery_stable_steps": recovery_stable_steps,
            "max_recovery_stable_steps": max_recovery_stable_steps,
            "recovery_handoff": recovery_handoff,
            "recovery_trace": recovery_trace,
            "crossed_frame": crossed_step is not None,
            "entered_gate": entered_gate,
            "outside_gate": outside_gate,
            "max_gate_lateral_extent_m": round(max_gate_lateral_extent_m, 4),
            "openloop_recovery": bool(low_frame_openloop_recovery),
            "recovery_blend_duration_s": low_frame_recovery_duration_s,
            "locomotion_recovery": bool(low_frame_locomotion_recovery),
            "locomotion_recovery_switched": locomotion_recovery_switched,
            "locomotion_blend_duration_s": low_frame_locomotion_blend_duration_s,
            "locomotion_switch_delay_s": low_frame_locomotion_switch_delay_s,
            "low_frame_nominal_init": bool(low_frame_nominal_init),
            "low_frame_straight_command": bool(low_frame_straight_command),
            "low_frame_zero_damping": bool(low_frame_zero_damping),
            "low_frame_unclipped_expert_target": bool(low_frame_unclipped_expert_target),
            "low_frame_training_payload": bool(low_frame_training_payload),
            "low_frame_isaac_dynamics": bool(low_frame_isaac_dynamics),
            "low_frame_isaac_reset": bool(low_frame_isaac_reset),
            "low_frame_damping_scale": low_frame_damping_scale,
            "low_frame_crouch_residual_scale": low_frame_crouch_residual_scale,
            "model_total_mass_kg": round(float(sim.model.body_mass.sum()), 6),
            "model_robot_body_count": int(sim.model.nbody - 1),
            "expert_target_clipped_steps": expert_target_clipped_steps,
            "expert_target_clipped_joints": expert_target_clipped_joints,
            "expert_target_clip_by_joint": expert_target_clip_by_joint.tolist(),
            "expert_target_max_overrun_rad": round(expert_target_max_overrun_rad, 4),
            "min_pole_clearance_m": (
                round(min_pole_clearance_m, 4) if math.isfinite(min_pole_clearance_m) else None
            ),
            "max_centreline_error_m": round(max_centreline_error_m, 4),
            "bridge_min_edge_margin_m": (
                round(bridge_min_edge_margin_m, 4)
                if math.isfinite(bridge_min_edge_margin_m) else None),
            "bridge_min_edge_margin_by_foot_m": {
                foot: round(margin, 4)
                for foot, margin in bridge_min_edge_margin_by_foot.items()},
            "bridge_min_support_center_margin_m": (
                round(bridge_min_support_center_margin_m, 4)
                if math.isfinite(bridge_min_support_center_margin_m) else None),
            "bridge_min_support_center_margin_by_foot_m": {
                foot: round(margin, 4)
                for foot, margin in bridge_min_support_center_margin_by_foot.items()},
            "bridge_foot_contact_trace": bridge_foot_contact_trace,
            "bridge_first_outside_ground_x_m": (
                round(bridge_first_outside_ground_x_m, 4)
                if bridge_first_outside_ground_x_m is not None else None),
            "bridge_failure_stage": (
                None if result.passed else
                "approach" if x < beam_entry_x else
                "entry" if x < beam_entry_stage_end_x else
                "middle" if x < beam_active_end_x else "exit"
            ) if obstacle == "beam" else None,
            "beam_modes_enabled": beam_modes_enabled,
            "beam_mode_at_end": beam_mode if beam_modes_enabled else None,
            "beam_exit_confirmed": beam_exit_confirmed if beam_modes_enabled else None,
            "beam_switch_events": beam_switch_events,
            "beam_switch_blend_duration_s": (beam_switch_blend_duration_s
                                             if beam_modes_enabled else None),
            "contact_events": contact_events,
            "rule_metrics": {
                **result.metrics,
                "entered_gate": entered_gate,
                "outside_gate": outside_gate,
                "max_gate_lateral_extent_m": round(
                    max_gate_lateral_extent_m, 4),
            },
            "trajectory": trajectory,
            "dynamics_trace": dynamics_trace,
            "initial_policy_obs": initial_policy_obs,
            "initial_estimator_obs": initial_estimator_obs,
        }
    finally:
        sim.close()


def wilson95(successes: int, trials: int) -> List[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    rate = successes / trials
    denom = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denom
    radius = z * math.sqrt(
        rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)
    ) / denom
    return [round(max(0.0, centre - radius), 4),
            round(min(1.0, centre + radius), 4)]


def aggregate(obstacle: str, records: Iterable[Dict]) -> Dict:
    records = list(records)
    successful = [record for record in records if record["success"]]
    reasons: Dict[str, int] = {}
    for record in records:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    rate = len(successful) / max(1, len(records))
    times = [record["completion_time_s"] for record in successful]
    clearances = [
        record["min_pole_clearance_m"]
        for record in records
        if record.get("min_pole_clearance_m") is not None
    ]
    return {
        "trials": len(records),
        "successes": len(successful),
        "success_rate": round(rate, 4),
        "wilson95": wilson95(len(successful), len(records)),
        "classification": _classify(obstacle, rate),
        "failure_reasons": reasons,
        "mean_completion_time_s": round(float(np.mean(times)), 3) if times else None,
        "p50_completion_time_s": round(float(np.percentile(times, 50)), 3) if times else None,
        "p95_completion_time_s": round(float(np.percentile(times, 95)), 3) if times else None,
        "min_pole_clearance_m": round(float(np.min(clearances)), 4) if clearances else None,
        "max_centreline_error_m": round(
            max((record["max_centreline_error_m"] for record in records), default=0.0), 4
        ),
        "max_obstacle_contact_force_n": round(
            max((record["max_obstacle_contact_force_n"] for record in records), default=0.0), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obstacle", choices=("slalom", "beam", "low_frame", "all"), default="all"
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--controller", choices=("pursuit", "waypoint"), default="pursuit")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--slalom-speed", type=float, default=None)
    parser.add_argument("--beam-width", type=float, default=None,
                        help="Bridge course width [m]; default is rule size.")
    parser.add_argument("--beam-length", type=float, default=None,
                        help="Bridge course length [m]; default is rule size.")
    parser.add_argument("--beam-height", type=float, default=None,
                        help="Bridge course height [m]; default is rule size.")
    parser.add_argument("--beam-speed", type=float, default=None,
                        help="Bridge pursuit forward speed [m/s].")
    parser.add_argument(
        "--slalom-pole6-entry-x", type=float, default=None,
        help="X coordinate of the (nominally 2.78, 1.35) m waypoint before pole 6.",
    )
    parser.add_argument("--slalom-lookahead", type=float, default=None)
    parser.add_argument("--slalom-curvature-slow", type=float, default=None)
    parser.add_argument("--slalom-k-cross", type=float, default=None)
    parser.add_argument("--slalom-k-yaw", type=float, default=None)
    parser.add_argument("--slalom-k-slow", type=float, default=None)
    parser.add_argument("--slalom-heading-slow", type=float, default=None)
    parser.add_argument("--body", type=Path, default=None,
                        help="Optional TorchScript body override.")
    parser.add_argument("--adaptation", type=Path, default=None,
                        help="Optional TorchScript adaptation-module override.")
    parser.add_argument(
        "--low-frame-stage", type=int, choices=(1, 2, 3, 4), default=3,
        help=("Exact H/W/L curriculum geometry: 1=.36/1.10/.40, "
              "2=.32/.90/.80, 3=.29/.80/.80, 4=.2755/.76/.84 m."),
    )
    parser.add_argument(
        "--low-frame-unclipped-expert-target", action="store_true",
        help="Send the expert's action centre plus residual directly to joint PD, as in Isaac.",
    )
    parser.add_argument(
        "--low-frame-zero-damping", action="store_true",
        help="Use the training URDF's zero joint damping for low-frame dynamics.",
    )
    parser.add_argument(
        "--low-frame-training-payload", action="store_true",
        help="Add the training URDF's fixed payload and IMU links to the MuJoCo scene.",
    )
    parser.add_argument(
        "--low-frame-isaac-dynamics", action="store_true",
        help="Match Isaac link armature, zero DOF armature, and RR calf inertia.",
    )
    parser.add_argument(
        "--low-frame-isaac-reset", action="store_true",
        help="Prime history after one zero-action control step, matching Isaac reset.",
    )
    parser.add_argument(
        "--low-frame-crouch-residual-scale", type=float, default=0.0,
        help="Optional Isaac-style smooth thigh-crouch residual diagnostic.",
    )
    parser.add_argument(
        "--low-frame-dynamics-trace", action="store_true",
        help="Record per-step base pose, joint state, PD target and first torque.",
    )
    parser.add_argument(
        "--low-frame-damping-scale", type=float, default=1.0,
        help="Scale MuJoCo joint damping for open-loop system-ID ablation.",
    )
    parser.add_argument(
        "--low-frame-straight-command", action="store_true",
        help="Use constant vx and zero vy/yaw before clearance, matching the Isaac low-frame evaluator.",
    )
    parser.add_argument(
        "--low-frame-nominal-init", action="store_true",
        help="Start the low-frame rollout immediately from the 0.21 m crouched pose, without a 1 s stand settling step.",
    )
    parser.add_argument(
        "--low-frame-speed", type=float, default=None,
        help="Override the low-frame pursuit forward-speed cap [m/s].",
    )
    parser.add_argument(
        "--low-frame-crouch-center", action="store_true",
        help="Use the Stage-1 expert's thigh=1.40/calf=-2.15 action centre.",
    )
    parser.add_argument("--low-frame-center-thigh", type=float, default=1.40)
    parser.add_argument("--low-frame-center-calf", type=float, default=-2.15)
    parser.add_argument(
        "--low-frame-openloop-recovery", action="store_true",
        help="After whole-body clearance, blend to the normal stand pose for 1 s.",
    )
    parser.add_argument(
        "--low-frame-recovery-duration", type=float, default=1.0,
        help="Open-loop post-clear stand-pose blend duration [s].",
    )
    parser.add_argument(
        "--low-frame-locomotion-recovery", action="store_true",
        help=("After whole-body clearance, hot-switch to the 059950 general "
              "locomotion policy with reset policy/estimator histories."),
    )
    parser.add_argument(
        "--low-frame-locomotion-blend-duration", type=float, default=0.0,
        help="Blend the post-clear locomotion target from the final expert target for this many seconds.",
    )
    parser.add_argument(
        "--low-frame-locomotion-switch-delay", type=float, default=0.0,
        help="Hold the low-frame expert after whole-body clearance before switching policies [s].",
    )
    parser.add_argument(
        "--recovery-body", type=Path, default=None,
        help="Optional recovery locomotion TorchScript body override.",
    )
    parser.add_argument(
        "--recovery-adaptation", type=Path, default=None,
        help="Optional recovery locomotion adaptation-module override.",
    )
    parser.add_argument(
        "--fixed-spawn", action="store_true",
        help="Disable the baseline's small repeatable spawn perturbations.",
    )
    parser.add_argument(
        "--violent-contact-force-threshold", type=float, default=250.0,
        help="Provisional frame impact threshold [N]; requires physical calibration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluations/robocon_2027/zero_training_baselines.json",
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    slalom_overrides = {
        "speed": args.slalom_speed,
        "lookahead_m": args.slalom_lookahead,
        "curvature_slow": args.slalom_curvature_slow,
        "k_cross": args.slalom_k_cross,
        "k_yaw": args.slalom_k_yaw,
        "k_slow": args.slalom_k_slow,
        "heading_slow": args.slalom_heading_slow,
    }
    if any(value is not None for value in slalom_overrides.values()) or args.slalom_pole6_entry_x is not None:
        if args.obstacle != "slalom" or args.controller != "pursuit":
            parser.error("slalom overrides require --obstacle slalom --controller pursuit")
        for name, value in slalom_overrides.items():
            if value is not None:
                if not math.isfinite(value) or value < 0 or (
                    name in {"speed", "k_cross", "k_yaw"} and value == 0):
                    parser.error(f"invalid slalom {name}: {value}")
                PURSUIT["slalom"][name] = value
        if args.slalom_pole6_entry_x is not None:
            entry_x = args.slalom_pole6_entry_x
            if not math.isfinite(entry_x) or not 2.3 <= entry_x <= 3.05:
                parser.error("--slalom-pole6-entry-x must be within [2.3, 3.05] m")
            points = list(SLALOM_WAYPOINTS)
            entry_index = points.index((2.78, 1.35))
            points[entry_index] = (entry_x, 1.35)
            PATH_BY_OBSTACLE["slalom"] = tuple(points)
    beam_overrides = (args.beam_width, args.beam_length, args.beam_height,
                      args.beam_speed)
    if any(value is not None for value in beam_overrides):
        if args.obstacle != "beam" or args.controller != "pursuit":
            parser.error("beam overrides require --obstacle beam --controller pursuit")
        if any(value is not None and (not math.isfinite(value) or value <= 0)
               for value in beam_overrides):
            parser.error("beam dimensions and speed must be finite and positive")
        if args.beam_length is not None and args.beam_length <= 0.8:
            parser.error("--beam-length must exceed 0.8 m")
        if args.beam_speed is not None:
            PURSUIT["beam"]["speed"] = args.beam_speed
    if args.low_frame_damping_scale < 0.0:
        parser.error("--low-frame-damping-scale must be nonnegative")
    if args.low_frame_crouch_residual_scale < 0.0:
        parser.error("--low-frame-crouch-residual-scale must be nonnegative")
    if args.low_frame_speed is not None and args.low_frame_speed <= 0.0:
        parser.error("--low-frame-speed must be positive")
    if args.low_frame_nominal_init and not args.low_frame_crouch_center:
        parser.error("--low-frame-nominal-init requires --low-frame-crouch-center")
    if args.violent_contact_force_threshold <= 0.0:
        parser.error("--violent-contact-force-threshold must be positive")
    if args.low_frame_recovery_duration <= 0.0:
        parser.error("--low-frame-recovery-duration must be positive")
    if args.low_frame_locomotion_blend_duration < 0.0:
        parser.error("--low-frame-locomotion-blend-duration must be nonnegative")
    if args.low_frame_locomotion_switch_delay < 0.0:
        parser.error("--low-frame-locomotion-switch-delay must be nonnegative")

    if args.low_frame_speed is not None:
        PURSUIT["low_frame"]["speed"] = args.low_frame_speed
    rules = load_rules()
    beam_spec = rules["obstacles"]["beam"]
    for key, value in (("width_m", args.beam_width),
                       ("length_m", args.beam_length),
                       ("height_m", args.beam_height)):
        if value is not None:
            beam_spec[key] = value
    if args.beam_length is not None:
        length = args.beam_length
        PATH_BY_OBSTACLE["beam"] = (
            (-0.60, 0.0), (0.0, 0.0),
            (0.20 * length, 0.0), (0.50 * length, 0.0),
            (0.80 * length, 0.0), (length, 0.0),
            (length + 0.45, 0.0),
        )
    low_frame_stages = {
        1: (0.36, 1.10, 0.40),
        2: (0.32, 0.90, 0.80),
        3: (0.29, 0.80, 0.80),
        4: (0.2755, 0.76, 0.84),
    }
    height, width, length = low_frame_stages[args.low_frame_stage]
    rules["obstacles"]["low_frame"].update(
        inner_height_m=height, inner_width_m=width, length_m=length)
    rng = np.random.default_rng(args.seed)
    obstacles = ("slalom", "beam", "low_frame") if args.obstacle == "all" else (args.obstacle,)
    payload = {
        "rulebook": rules["rulebook"],
        "kind": ("fine_tuned_policy_evaluation" if args.body or args.adaptation
                 else "zero_training_baseline"),
        "provenance": git_provenance(),
        "argv": sys.argv,
        "versions": {"mujoco": mujoco.__version__, "torch": torch.__version__},
        "input_sha256": {
            "rules_yaml": sha256_file(
                PROJECT_ROOT / "robodog_gym/envs/robocon_2027/config/rules.yaml"),
            "policy_config_yaml": sha256_file(DEFAULT_CONFIG),
            "base_xml": sha256_file(BASE_XML),
        },
        "controller": args.controller,
        "seed": args.seed,
        "low_frame_speed_mps": PURSUIT["low_frame"]["speed"],
        "pursuit_parameters": {name: dict(values) for name, values in PURSUIT.items()},
        "slalom_contact_sample_dt_s": 0.005,
        "slalom_waypoints_m": [list(point) for point in PATH_BY_OBSTACLE["slalom"]],
        "beam_waypoints_m": [list(point) for point in PATH_BY_OBSTACLE["beam"]],
        "beam_geometry_m": {key: beam_spec[key] for key in
                            ("width_m", "length_m", "height_m")},
        "low_frame_nominal_init": bool(args.low_frame_nominal_init),
        "low_frame_straight_command": bool(args.low_frame_straight_command),
        "low_frame_zero_damping": bool(args.low_frame_zero_damping),
        "low_frame_unclipped_expert_target": bool(args.low_frame_unclipped_expert_target),
        "low_frame_training_payload": bool(args.low_frame_training_payload),
        "low_frame_isaac_dynamics": bool(args.low_frame_isaac_dynamics),
        "low_frame_isaac_reset": bool(args.low_frame_isaac_reset),
        "low_frame_crouch_residual_scale": args.low_frame_crouch_residual_scale,
        "low_frame_damping_scale": args.low_frame_damping_scale,
        "obstacles": {},
        "notes": {
            "slalom_layout": "six 100 mm posts, 900 mm centre spacing, L shaped",
            "slalom_overflight_detector": "post centre inside the convex hull of base+feet",
            "slalom_pass_detector": "ordered forward projection crossings with alternating sides",
            "min_pole_clearance_m": "foot-centre XY proxy only; negative does not prove 3D collision",
            "low_frame_geometry": "full-width roof, four corner posts, and per-geom lateral gate check",
            "low_frame_collision": "contact peak sampled each physics substep; violent/displacement threshold awaits referee/physical calibration",
        },
    }
    defaults = {"slalom": 60.0, "beam": 20.0, "low_frame": 16.0}
    for obstacle in obstacles:
        body_path = args.body or MODELS[obstacle][0]
        adaptation_path = args.adaptation or MODELS[obstacle][1]
        payload["obstacles"][obstacle] = {
            "body_path": str(body_path),
            "body_sha256": sha256_file(body_path),
            "adaptation_path": str(adaptation_path),
            "adaptation_sha256": sha256_file(adaptation_path),
        }
        records = []
        max_seconds = args.max_seconds or defaults[obstacle]
        for trial in range(args.trials):
            # Small repeatable pose offsets expose brittle planner/policy
            # combinations without turning this smoke test into domain randomisation.
            spawn_perturbation = (
                (0.0, 0.0, 0.0)
                if args.fixed_spawn else (
                    float(rng.uniform(-0.03, 0.03)),
                    float(rng.uniform(-0.03, 0.03)),
                    float(rng.uniform(math.radians(-2.0), math.radians(2.0))),
                )
            )
            record = run_trial(
                obstacle, rules, max_seconds, args.controller,
                spawn_perturbation, args.body, args.adaptation,
                args.low_frame_crouch_center,
                args.violent_contact_force_threshold,
                args.low_frame_center_thigh,
                args.low_frame_center_calf,
                args.low_frame_openloop_recovery,
                args.low_frame_recovery_duration,
                args.low_frame_locomotion_recovery,
                args.low_frame_locomotion_blend_duration,
                args.low_frame_locomotion_switch_delay,
                args.low_frame_nominal_init,
                args.low_frame_straight_command,
                args.low_frame_zero_damping,
                args.low_frame_unclipped_expert_target,
                args.low_frame_training_payload,
                args.low_frame_isaac_dynamics,
                args.low_frame_isaac_reset,
                args.low_frame_crouch_residual_scale,
                args.low_frame_dynamics_trace,
                args.low_frame_damping_scale,
                args.recovery_body,
                args.recovery_adaptation)
            record["trial"] = trial
            records.append(record)
            print(
                f"[{obstacle}] trial={trial + 1}/{args.trials} "
                f"success={record['success']} reason={record['reason']} "
                f"time={record['completion_time_s']} xyz={record['final_base_xyz']}",
                flush=True,
            )
        payload["obstacles"][obstacle].update({
            "aggregate": aggregate(obstacle, records),
            "records": records,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(
        {key: value["aggregate"] for key, value in payload["obstacles"].items()},
        ensure_ascii=False,
        indent=2,
    ))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
