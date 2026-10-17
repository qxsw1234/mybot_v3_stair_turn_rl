#!/usr/bin/env python3
"""Drive all eight 2027 obstacles in one continuous MuJoCo preview map."""
import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from eval_robocon_2027_baselines import DEFAULT_CONFIG, LOCOMOTION_CKPT, Robocon2027Sim
from robocon_2027_full_map import SPAWNS, STATIONS, build_full_map
from sim2sim_mujoco import KeyboardController, load_config
from sim2sim_mujoco_robocon_map import AutoStandController, RoboconMapSim, settle

ROOT = Path(__file__).resolve().parents[1]
BASE = (LOCOMOTION_CKPT/'body_059950.jit', LOCOMOTION_CKPT/'adaptation_module_059950.jit')
BEAM_DIR = ROOT/'runs/mybot_v3_beam_expert_resume_060049/2026-10-16_16-36-23.211375/checkpoints'
LOW_DIR = ROOT/'runs/mybot_v3_low_bar_expert_resume_064242/2026-10-12_03-07-21.669914/checkpoints'
MODELS = {
    'walk': BASE,
    'beam': (BEAM_DIR/'body_latest.jit', BEAM_DIR/'adaptation_module_latest.jit'),
    'low_frame': (LOW_DIR/'body_latest.jit', LOW_DIR/'adaptation_module_latest.jit'),
}
CROUCH = [0.10, 1.40, -2.15, -0.10, 1.40, -2.15]*2
SPAWN_KEYS = dict(zip('12345678', (station[0] for station in STATIONS)))


class FullMapSim(RoboconMapSim):
    switch_policy = Robocon2027Sim.switch_policy

    def _configure_terrain(self):
        ids = super()._configure_terrain()
        # Height scans should see walkable surfaces, not the roof or vertical obstacles.
        ignore = ('low_frame_', 'wall', 'tire_', 'pit_guard_', 'marker_')
        return [i for i in ids if not (mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').replace(
            'rc27_full_', '', 1).startswith(ignore)]


def desired_mode(x):
    if 1.15 <= x < 3.18:
        return 'low_frame'
    if 3.73 <= x < 7.35:
        return 'beam'
    return 'walk'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=3600.0)
    parser.add_argument('--spawn', choices=tuple(SPAWNS), default='start')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--vx', type=float, default=0.24)
    parser.add_argument('--vy', type=float, default=0.15)
    parser.add_argument('--yaw-rate', type=float, default=0.4)
    parser.add_argument('--overview', action='store_true')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('--duration must be positive')
    if not args.headless and not os.environ.get('DISPLAY'):
        parser.error('graphical desktop DISPLAY is required')
    for pair in MODELS.values():
        for path in pair:
            if not path.is_file():
                parser.error(f'missing policy: {path}')
    xml, manifest = build_full_map()
    config = load_config(DEFAULT_CONFIG)
    config.update(obstacle_ahead_x=2.0, obstacle_ahead_y=0.0,
                  obstacle_clearance_m=0.29, obstacle_thickness_m=0.05,
                  obstacle_heading_rad=0.0, obstacle_ahead_valid=1.0)
    sim = FullMapSim(config, *BASE, ROOT, xml)
    stand_controller = AutoStandController(sim.policy_dof_pos.copy(),
                                           sim.control_dt, 0.02, 0.20, 0.30)
    normal_centre = list(config['policy_dof_pos'])
    normal_clip = float(config['clip_actions'])
    controller = None
    mode = 'walk'
    spawn_name = args.spawn
    follow = not args.overview
    last_target = sim.policy_dof_pos.copy()
    blend_start = None
    blend_from = last_target.copy()
    switches = []
    warned = False

    def switch(to_mode, step):
        nonlocal mode, blend_start, blend_from
        if to_mode == mode:
            return
        centre, clip = ((CROUCH, 4.5) if to_mode == 'low_frame'
                        else (normal_centre, normal_clip))
        sim.switch_policy(*MODELS[to_mode], centre, clip)
        stand_controller.nominal = sim.policy_dof_pos.copy()
        blend_start = step
        blend_from = last_target.copy()
        mode = to_mode
        event = {'t_s': round(float(sim.data.time), 2),
                 'x_m': round(float(sim.data.qpos[0]), 2), 'mode': mode}
        switches.append(event)
        print('[full-map] 自动切换 '+json.dumps(event, ensure_ascii=False), flush=True)

    def reset(name):
        nonlocal mode, last_target, blend_start, warned, spawn_name
        if mode != 'walk':
            sim.switch_policy(*MODELS['walk'], normal_centre, normal_clip)
        mode = 'walk'
        spawn_name = name
        sim.teleport(*SPAWNS[name])
        settle(sim, 0.30)
        last_target = np.clip(sim.policy_dof_pos, sim.lower, sim.upper).copy()
        stand_controller.nominal = last_target.copy()
        stand_controller.reset()
        blend_start = None
        warned = False
        print(f'[full-map] 起点 {name} x={sim.data.qpos[0]:.2f} m；同一张地图', flush=True)

    def step_once(step, command, disturbance):
        nonlocal last_target, blend_start, warned
        x = float(sim.data.qpos[0])
        switch(desired_mode(x), step)
        stand_controller.update_mode(command)
        if stand_controller.standing:
            target = np.clip(sim.policy_dof_pos, sim.lower, sim.upper)
        else:
            action = sim.infer(np.asarray(command, dtype=np.float32))
            target = np.clip(sim.target_from_action(
                action, float(sim.config['action_scale'])), sim.lower, sim.upper)
        if blend_start is not None:
            elapsed = (step - blend_start + 1)*sim.control_dt
            blend_seconds = 0.55 if mode == 'low_frame' else 0.30
            alpha = min(1.0, elapsed/blend_seconds)
            target = np.clip((1-alpha)*blend_from+alpha*target, sim.lower, sim.upper)
            if alpha >= 1.0:
                blend_start = None
        target = np.clip(stand_controller.blend_target(target), sim.lower, sim.upper)
        sim.step_control(target, disturbance)
        last_target = np.asarray(target).copy()
        if sim.fell and not warned:
            print('[full-map] 检测到跌倒；按 R 回到当前入口，0 回到整场起点。', flush=True)
            warned = True

    reset(spawn_name)
    print('[full-map] 单张连续地图：限高 → 单木桥 → 台阶跨沟 → 高墙 → 梅花桩 → 轮胎阵 → 碎木坑 → 绕桩。')
    print('[full-map] W/S 前后，A/D 横移，Q/E 转向；0 整场起点，1-8 快速定位同图障碍，R 重置。')
    print('[full-map] V 切换跟随/鸟瞰镜头；模型自动切换：行走、限高、单木桥。其余项目是几何预览。', flush=True)
    print('[full-map] 地图文件 '+str(xml), flush=True)
    total_steps = max(1, int(args.duration/sim.control_dt))
    try:
        if args.headless:
            command = (args.vx, 0.0, 0.0)
            for step in range(total_steps):
                step_once(step, command, np.zeros(3, dtype=np.float32))
        else:
            controller = KeyboardController(0.0, args.vx, args.vy, args.yaw_rate)
            controller.keys.update({key: False for key in '0678v'})
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                def camera():
                    viewer.cam.lookat[:] = ((float(sim.data.qpos[0])+1.0,
                                              float(sim.data.qpos[1]), 0.2)
                                             if follow else (18.0, 0.0, 0.0))
                    viewer.cam.distance = 5.0 if follow else 46.0
                    viewer.cam.azimuth = 135.0 if follow else 90.0
                    viewer.cam.elevation = -35.0 if follow else -78.0
                camera()
                for step in range(total_steps):
                    if not viewer.is_running():
                        break
                    start = time.monotonic()
                    if controller.consume('0'):
                        reset('start')
                    for key, name in SPAWN_KEYS.items():
                        if controller.consume(key):
                            reset(name)
                            break
                    if controller.consume('r'):
                        reset(spawn_name)
                    if controller.consume('v'):
                        follow = not follow
                        camera()
                    step_once(step, controller.command(), controller.disturbance())
                    if follow and step % 4 == 0:
                        viewer.cam.lookat[:] = (float(sim.data.qpos[0])+1.0,
                                                float(sim.data.qpos[1]), 0.2)
                    viewer.sync()
                    if step % 100 == 0:
                        print('[full-map] t={:.1f}s x={:.2f} y={:.2f} mode={} fell={}'.format(
                            float(sim.data.time), float(sim.data.qpos[0]),
                            float(sim.data.qpos[1]), mode, sim.fell), flush=True)
                    remaining = sim.control_dt-(time.monotonic()-start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if controller:
            controller.close()
    print('[full-map] summary='+json.dumps({
        'x_m': float(sim.data.qpos[0]), 'y_m': float(sim.data.qpos[1]),
        'mode': mode, 'fell': bool(sim.fell), 'switches': switches,
        'map_stations': len(manifest['stations'])}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
