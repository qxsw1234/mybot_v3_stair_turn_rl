#!/usr/bin/env python3
"""Record one reproducible MuJoCo obstacle trial using the rule evaluator.

The MP4 is a visual aid, not a statistical evaluation. The adjacent JSON gives
its exact checkpoint, spawn, scene and rule verdict. Set MUJOCO_GL=egl on a
headless desktop.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import yaml

from eval_robocon_2027_baselines import (
    PATH_BY_OBSTACLE, PURSUIT, SLALOM_WAYPOINTS, SPAWNS,
    LOCOMOTION_CKPT, run_trial,
)
from robodog_gym.envs.robocon_2027 import load_rules

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SLALOM = ROOT / 'robodog_gym/envs/robocon_2027/config/slalom_controller_p1_20261015.yaml'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class Recorder:
    def __init__(self, path, obstacle, width=640, height=480, fps=10):
        self.path = Path(path)
        self.obstacle = obstacle
        self.width, self.height, self.fps = width, height, fps
        self.renderer = None
        self.process = None
        self.frames = 0

    def __call__(self, sim, step):
        stride = max(1, int(round(1.0 / (self.fps * sim.control_dt))))
        if step % stride:
            return
        if self.renderer is None:
            self.renderer = mujoco.Renderer(sim.model, height=self.height, width=self.width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            if self.obstacle == 'slalom':
                camera.lookat[:] = (1.1, 0.85, 0.1)
                camera.distance = 6.0
                camera.azimuth = 90
                camera.elevation = -68
            else:
                camera.lookat[:] = (1.5, 0.0, 0.15)
                camera.distance = 5.2
                camera.azimuth = 135
                camera.elevation = -38
            self.camera = camera
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.process = subprocess.Popen([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'rawvideo', '-pixel_format', 'rgb24',
                '-video_size', f'{self.width}x{self.height}',
                '-framerate', str(self.fps), '-i', 'pipe:0',
                '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(self.path),
            ], stdin=subprocess.PIPE)
        self.renderer.update_scene(sim.data, camera=self.camera)
        frame = np.ascontiguousarray(self.renderer.render())
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            status = self.process.wait(timeout=120)
            if status:
                raise RuntimeError(f'ffmpeg exited {status}')
        if self.renderer is not None:
            self.renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--obstacle', required=True, choices=['slalom', 'beam'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--trial', type=int, default=0, help='Zero-based trial in seeded evaluator series')
    ap.add_argument('--body', type=Path)
    ap.add_argument('--adaptation', type=Path)
    ap.add_argument('--beam-expert-body', type=Path)
    ap.add_argument('--beam-expert-adaptation', type=Path)
    ap.add_argument('--beam-switch-blend-duration', type=float, default=0.3)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.trial < 0:
        ap.error('--trial must be nonnegative')
    if (args.beam_expert_body is None) != (args.beam_expert_adaptation is None):
        ap.error('provide both beam expert JIT files')
    if args.beam_expert_body and args.obstacle != 'beam':
        ap.error('beam expert switching requires --obstacle beam')
    rules = load_rules()
    if args.obstacle == 'slalom':
        frozen = yaml.safe_load(FROZEN_SLALOM.read_text())
        PURSUIT['slalom']['speed'] = float(frozen['params']['speed_mps'])
        points = list(SLALOM_WAYPOINTS)
        points[points.index((2.78, 1.35))] = (float(frozen['params']['pole6_entry_x_m']), 1.35)
        PATH_BY_OBSTACLE['slalom'] = tuple(points)
        max_seconds = float(frozen['assessment']['max_seconds'])
    else:
        max_seconds = 60.0
    rng = np.random.default_rng(args.seed)
    for _ in range(args.trial + 1):
        perturb = (
            float(rng.uniform(-0.03, 0.03)),
            float(rng.uniform(-0.03, 0.03)),
            float(rng.uniform(math.radians(-2), math.radians(2))),
        )
    body = args.body or LOCOMOTION_CKPT / 'body_059950.jit'
    adaptation = args.adaptation or LOCOMOTION_CKPT / 'adaptation_module_059950.jit'
    recorder = Recorder(args.output, args.obstacle)
    try:
        result = run_trial(
            args.obstacle, rules, max_seconds, 'pursuit', perturb,
            body_path=body, adaptation_path=adaptation,
            beam_expert_body_path=args.beam_expert_body,
            beam_expert_adaptation_path=args.beam_expert_adaptation,
            beam_switch_blend_duration_s=args.beam_switch_blend_duration,
            frame_callback=recorder,
        )
    finally:
        recorder.close()
    summary = {
        'kind': 'single_visual_demo_not_statistical_evidence',
        'obstacle': args.obstacle,
        'seed': args.seed,
        'trial_zero_based': args.trial,
        'nominal_spawn': SPAWNS[args.obstacle],
        'body_path': str(body), 'body_sha256': digest(body),
        'adaptation_path': str(adaptation), 'adaptation_sha256': digest(adaptation),
        'beam_expert_body_path': str(args.beam_expert_body) if args.beam_expert_body else None,
        'beam_expert_body_sha256': digest(args.beam_expert_body) if args.beam_expert_body else None,
        'beam_expert_adaptation_path': str(args.beam_expert_adaptation) if args.beam_expert_adaptation else None,
        'beam_expert_adaptation_sha256': digest(args.beam_expert_adaptation) if args.beam_expert_adaptation else None,
        'beam_switch_events': result['beam_switch_events'],
        'scene_xml_sha256': result['scene_xml_sha256'],
        'video_path': str(args.output.resolve()),
        'video_sha256': digest(args.output),
        'video_frames': recorder.frames, 'video_fps': recorder.fps,
        'verdict': {key: result[key] for key in
                    ('success', 'reason', 'completion_time_s', 'spawn',
                     'frame_or_obstacle_contact_steps', 'max_obstacle_contact_force_n')},
    }
    sidecar = args.output.with_suffix('.json')
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
