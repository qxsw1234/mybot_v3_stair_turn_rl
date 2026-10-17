#!/usr/bin/env python3
"""Keyboard inspection of the 2027 rule-size slalom or low-frame scene.

This viewer loads research checkpoints and is not a scored evaluation.
"""

import argparse
import math
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from eval_robocon_2027_baselines import (
    LOCOMOTION_CKPT, PATH_BY_OBSTACLE, PURSUIT, SPAWNS, Robocon2027Sim,
    pursuit_command, resample_path, stand,
)
from robodog_gym.envs.robocon_2027 import load_rules
from sim2sim_mujoco import KeyboardController

ROOT = Path(__file__).resolve().parents[1]
LOW_FRAME_DIR = (ROOT / 'runs/mybot_v3_low_bar_expert_resume_064242'
                 / '2026-10-12_03-07-21.669914/checkpoints')
SLALOM_CONFIG = (ROOT / 'robodog_gym/envs/robocon_2027/config'
                 / 'slalom_controller_p1_20261015.yaml')
MODELS = {
    'slalom': (LOCOMOTION_CKPT / 'body_059950.jit',
               LOCOMOTION_CKPT / 'adaptation_module_059950.jit'),
    'low_frame': (LOW_FRAME_DIR / 'body_latest.jit',
                  LOW_FRAME_DIR / 'adaptation_module_latest.jit'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('obstacle', choices=('slalom', 'low_frame'))
    parser.add_argument('--duration', type=float, default=600.0)
    parser.add_argument('--vx', type=float, default=0.22)
    parser.add_argument('--vy', type=float, default=0.20)
    parser.add_argument('--yaw-rate', type=float, default=0.40)
    parser.add_argument('--body', type=Path)
    parser.add_argument('--adaptation', type=Path)
    parser.add_argument('--scene-request-file', type=Path,
                        help='internal scene switch request for the 2027 live launcher')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--headless-command', type=float, nargs=3,
                        default=(0.0, 0.0, 0.0), metavar=('VX', 'VY', 'YAW'))
    parser.add_argument('--autopilot', action='store_true',
                        help='follow the current frozen slalom path or drive straight under the low frame')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    if not args.headless and not os.environ.get('DISPLAY'):
        parser.error('MuJoCo viewer needs desktop DISPLAY')
    body = args.body or MODELS[args.obstacle][0]
    adaptation = args.adaptation or MODELS[args.obstacle][1]
    for path in (body, adaptation):
        if not path.is_file():
            parser.error(f'missing model: {path}')

    rules = load_rules()
    sim = Robocon2027Sim(
        args.obstacle, rules, body, adaptation,
        low_frame_crouch_center=(args.obstacle == 'low_frame'))
    path = None
    params = None
    if args.obstacle == 'slalom':
        frozen = yaml.safe_load(SLALOM_CONFIG.read_text())
        params = dict(PURSUIT['slalom'])
        params['speed'] = float(frozen['params']['speed_mps'])
        points = list(PATH_BY_OBSTACLE['slalom'])
        points[points.index((2.78, 1.35))] = (
            float(frozen['params']['pole6_entry_x_m']), 1.35)
        path = resample_path(points)
    autopilot = args.autopilot
    controller = None
    next_scene = None
    contact_announced = False
    fall_announced = False
    path_index = 0

    def reset():
        nonlocal contact_announced, fall_announced, path_index
        sim.teleport(*SPAWNS[args.obstacle])
        if args.obstacle == 'low_frame':
            sim.data.qpos[2] = 0.21
            mujoco.mj_forward(sim.model, sim.data)
        stand(sim)
        contact_announced = False
        fall_announced = False
        path_index = 0
        print(f'[2027 {args.obstacle}] 已回到入口', flush=True)

    reset()
    print(f'[2027 {args.obstacle}] 使用模型 {body}', flush=True)
    print('[2027] W/S 前后，A/D 横移，Q/E 转向；P 自动路线开关，R 重置；关窗或 Ctrl+C 退出。',
          flush=True)
    print('[2027] 键盘画面仅用于观察；真实规则结果以批量评测器为准。', flush=True)
    if args.obstacle == 'low_frame':
        print('[2027 low_frame] 当前 064341 是 Stage 1 研究模型；默认 MuJoCo 动力学仍有明显差距。',
              flush=True)

    def advance(step, manual_command):
        nonlocal path_index, contact_announced, fall_announced
        command = np.asarray(manual_command, dtype=np.float32)
        if autopilot and not np.any(command):
            if args.obstacle == 'slalom':
                command, path_index = pursuit_command(sim, path, path_index, params)
            else:
                command = np.array([
                    PURSUIT['low_frame']['speed']
                    if float(sim.data.qpos[0]) < 1.50 else 0.0,
                    0.0, 0.0], dtype=np.float32)
        action = sim.infer(command)
        target = sim.target_from_action(action, float(sim.config['action_scale']))
        target = np.clip(target, sim.lower, sim.upper)
        sim.step_control(target)
        snapshot = sim.contact_snapshot()
        if args.obstacle == 'slalom':
            if snapshot['obstacle_geoms'] and not contact_announced:
                print('[2027 slalom] 检测到碰桩；此提示按控制步采样，不是正式判定。',
                      flush=True)
                contact_announced = True
        elif snapshot['obstacle_geoms'] and not contact_announced:
            print('[2027 low_frame] 检测到框架接触；轻触可能被规则允许。', flush=True)
            contact_announced = True
        if sim.fell and not fall_announced:
            print('[2027] 检测到跌倒，按 R 重置。', flush=True)
            fall_announced = True
        if step % 100 == 0:
            print('[2027 {}] t={:.1f}s x={:.2f} y={:.2f} autopilot={} fell={}'.format(
                args.obstacle, float(sim.data.time), float(sim.data.qpos[0]),
                float(sim.data.qpos[1]), autopilot, sim.fell), flush=True)

    total_steps = max(1, int(args.duration / sim.control_dt))
    try:
        if args.headless:
            for step in range(total_steps):
                advance(step, args.headless_command)
        else:
            controller = KeyboardController(0.0, args.vx, args.vy, args.yaw_rate)
            controller.keys['p'] = False
            if args.scene_request_file:
                controller.keys.update({'4': False, '5': False, '6': False})
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.lookat[:] = ((1.4, 0.4, 0.2)
                                        if args.obstacle == 'slalom' else (0.4, 0.0, 0.15))
                viewer.cam.distance = 5.5 if args.obstacle == 'slalom' else 4.2
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -38.0
                for step in range(total_steps):
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    if args.scene_request_file:
                        for key, scene in (('4', 'slalom'), ('5', 'beam'),
                                           ('6', 'low_frame')):
                            if controller.consume(key):
                                next_scene = scene
                                break
                        if next_scene is not None:
                            break
                    if controller.consume('r'):
                        reset()
                    if controller.consume('p'):
                        autopilot = not autopilot
                        print(f'[2027] 自动路线={autopilot}', flush=True)
                    advance(step, controller.command())
                    viewer.sync()
                    remaining = sim.control_dt - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if controller is not None:
            controller.close()
    if next_scene is not None:
        args.scene_request_file.write_text(next_scene + '\n')
        print(f'[2027 {args.obstacle}] 请求切换到 {next_scene}', flush=True)
        return
    print(f'[2027 {args.obstacle}] 已退出；最终位置 '
          f'x={float(sim.data.qpos[0]):.2f}, y={float(sim.data.qpos[1]):.2f}', flush=True)


if __name__ == '__main__':
    main()
