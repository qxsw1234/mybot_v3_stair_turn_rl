#!/usr/bin/env python3
"""Keyboard-driven 2027 rule-size bridge with optional internal policy switching.

This is a MuJoCo inspection tool, not a scored evaluation or a motor command.
Run it in the graphical desktop session. Use the evaluation scripts for rates.
"""

import argparse
import json
import os
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from eval_robocon_2027_baselines import (
    DEFAULT_CONFIG, LOCOMOTION_CKPT, SPAWNS, Robocon2027Sim, stand,
)
from robodog_gym.envs.robocon_2027 import load_rules
from sim2sim_mujoco import KeyboardController, load_config

ROOT = Path(__file__).resolve().parents[1]
EXPERT_DIR = (ROOT / 'runs/mybot_v3_beam_expert_resume_060049'
              / '2026-10-16_16-36-23.211375/checkpoints')
BASE_BODY = LOCOMOTION_CKPT / 'body_059950.jit'
BASE_ADAPTATION = LOCOMOTION_CKPT / 'adaptation_module_059950.jit'
EXPERT_BODY = EXPERT_DIR / 'body_latest.jit'
EXPERT_ADAPTATION = EXPERT_DIR / 'adaptation_module_latest.jit'


def wanted_auto_mode(x, bridge_length):
    return 'beam_expert' if -0.15 <= x < bridge_length + 0.30 else 'locomotion'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=600.0)
    parser.add_argument('--mode', choices=('auto', 'locomotion', 'beam_expert'),
                        default='auto')
    parser.add_argument('--vx', type=float, default=0.24,
                        help='forward/back keyboard command in m/s')
    parser.add_argument('--vy', type=float, default=0.15)
    parser.add_argument('--yaw-rate', type=float, default=0.40)
    parser.add_argument('--blend-duration', type=float, default=0.30)
    parser.add_argument('--disturbance-force', type=float, default=0.0)
    parser.add_argument('--body', type=Path, default=BASE_BODY)
    parser.add_argument('--adaptation', type=Path, default=BASE_ADAPTATION)
    parser.add_argument('--expert-body', type=Path, default=EXPERT_BODY)
    parser.add_argument('--expert-adaptation', type=Path, default=EXPERT_ADAPTATION)
    parser.add_argument('--scene-request-file', type=Path,
                        help='internal scene switch request for the 2027 live launcher')
    parser.add_argument('--headless', action='store_true',
                        help='run a short fixed-command smoke test without a viewer')
    parser.add_argument('--headless-command', type=float, nargs=3,
                        default=(0.24, 0.0, 0.0), metavar=('VX', 'VY', 'YAW'))
    args = parser.parse_args()
    if args.duration <= 0 or args.blend_duration < 0:
        parser.error('--duration must be positive and --blend-duration nonnegative')
    if not args.headless and not os.environ.get('DISPLAY'):
        parser.error('MuJoCo viewer needs a graphical desktop; set DISPLAY=:0 on desktop')
    for path in (args.body, args.adaptation, args.expert_body, args.expert_adaptation):
        if not path.is_file():
            parser.error(f'missing JIT model: {path}')

    rules = load_rules()
    bridge_length = float(rules['obstacles']['beam']['length_m'])
    normal_cfg = load_config(DEFAULT_CONFIG)
    sim = Robocon2027Sim('beam', rules, args.body, args.adaptation)
    models = {
        'locomotion': (args.body, args.adaptation),
        'beam_expert': (args.expert_body, args.expert_adaptation),
    }
    selected = args.mode
    active = 'locomotion'
    switches = []
    last_target = sim.policy_dof_pos.copy()
    blend_start = None
    blend_target = None
    outside_warned = False
    fall_warned = False
    controller = None
    next_scene = None

    def switch(to_mode, step):
        nonlocal active, blend_start, blend_target
        if to_mode == active:
            return
        sim.switch_policy(
            *models[to_mode], normal_cfg['policy_dof_pos'],
            float(normal_cfg['clip_actions']))
        blend_start = step
        blend_target = last_target.copy()
        active = to_mode
        event = {'time_s': round(float(sim.data.time), 2),
                 'x_m': round(float(sim.data.qpos[0]), 3), 'mode': active}
        switches.append(event)
        print(f"[bridge] 切换 {active}，t={event['time_s']} s，x={event['x_m']} m",
              flush=True)

    def reset():
        nonlocal active, last_target, blend_start, blend_target, outside_warned, fall_warned
        if active != 'locomotion':
            sim.switch_policy(
                *models['locomotion'], normal_cfg['policy_dof_pos'],
                float(normal_cfg['clip_actions']))
        active = 'locomotion'
        sim.teleport(*SPAWNS['beam'])
        last_target = stand(sim)
        blend_start = None
        blend_target = None
        outside_warned = False
        fall_warned = False
        print('[bridge] 已回到规则桥入口；当前策略 locomotion', flush=True)

    reset()
    total_steps = max(1, int(args.duration / sim.control_dt))
    print('[bridge] 规则桥 0.40 × 3.00 × 0.10 m；这是手动观察，不计入正式通过率。')
    print(f'[bridge] 普通行走: {args.body}; 桥专家: {args.expert_body}')
    print('[bridge] W/S 前后，A/D 横移，Q/E 转向；1 普通行走，2 桥专家，3 自动切换，R 重置。')
    print('[bridge] 自动切换位置：x≥-0.15 m 进入专家，x≥3.30 m 切回行走；Ctrl+C 或关窗退出。',
          flush=True)

    def control_step(step, command, disturbance):
        nonlocal last_target, blend_start, outside_warned, fall_warned
        x = float(sim.data.qpos[0])
        desired_mode = (wanted_auto_mode(x, bridge_length)
                        if selected == 'auto' else selected)
        switch(desired_mode, step)
        action = sim.infer(np.asarray(command, dtype=np.float32))
        target = np.clip(
            sim.target_from_action(action, float(sim.config['action_scale'])),
            sim.lower, sim.upper)
        if blend_start is not None:
            elapsed = (step - blend_start + 1) * sim.control_dt
            alpha = min(1.0, elapsed / args.blend_duration) if args.blend_duration else 1.0
            target = np.clip((1.0 - alpha) * blend_target + alpha * target,
                             sim.lower, sim.upper)
            if alpha >= 1.0:
                blend_start = None
        sim.step_control(target, disturbance)
        last_target = np.asarray(target).copy()
        snapshot = sim.contact_snapshot()
        if (0.0 <= float(sim.data.qpos[0]) <= bridge_length
                and len(snapshot['floor_feet']) >= 2 and not outside_warned):
            print('[bridge] 注意：桥段检测到至少两足接触桥外地面；此回合可能失分。',
                  flush=True)
            outside_warned = True
        if sim.fell and not fall_warned:
            print('[bridge] 检测到跌倒；按 R 回到入口。', flush=True)
            fall_warned = True

    try:
        if args.headless:
            for step in range(total_steps):
                control_step(step, args.headless_command, np.zeros(3))
        else:
            controller = KeyboardController(args.disturbance_force, args.vx,
                                            args.vy, args.yaw_rate)
            if args.scene_request_file:
                controller.keys.update({'4': False, '5': False, '6': False})
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.lookat[:] = (1.5, 0.0, 0.15)
                viewer.cam.distance = 5.2
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
                    if controller.consume('1'):
                        selected = 'locomotion'
                        print('[bridge] 选择普通行走模式', flush=True)
                    if controller.consume('2'):
                        selected = 'beam_expert'
                        print('[bridge] 选择桥专家模式', flush=True)
                    if controller.consume('3'):
                        selected = 'auto'
                        print('[bridge] 选择自动切换模式', flush=True)
                    control_step(step, controller.command(), controller.disturbance())
                    viewer.sync()
                    if step % 100 == 0:
                        print('[bridge] t={:.1f}s x={:.2f} y={:.2f} mode={} select={} fell={}'.format(
                            float(sim.data.time), float(sim.data.qpos[0]),
                            float(sim.data.qpos[1]), active, selected, sim.fell),
                            flush=True)
                    remaining = sim.control_dt - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if controller is not None:
            controller.close()
    if next_scene is not None:
        args.scene_request_file.write_text(next_scene + '\n')
        print(f'[bridge] 请求切换到 {next_scene}', flush=True)
        return
    print('[bridge] summary=' + json.dumps({
        'mode': active, 'selection': selected, 'switches': switches,
        'x_m': float(sim.data.qpos[0]), 'y_m': float(sim.data.qpos[1]),
        'fell': bool(sim.fell), 'outside_ground_warning': outside_warned,
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
