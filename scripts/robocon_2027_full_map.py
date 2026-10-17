#!/usr/bin/env python3
"""Build one continuous eight-obstacle MuJoCo preview course from 2027 rules.

Dimensions explicitly present in rules.yaml are copied. Unspecified layouts,
materials and composite collision shapes are reproducible previews, not
competition geometry or scoring evidence.
"""

import json
import math
import random
import re
from pathlib import Path

from robodog_gym.envs.robocon_2027 import load_rules

ROOT = Path(__file__).resolve().parents[1]
BASE_XML = ROOT / 'deploy_cpp/robot/mybot_v3/xml/mybot_v3.xml'
OUTPUT_DIR = ROOT / 'tmp/robocon_2027'
OUTPUT_XML = OUTPUT_DIR / 'full_map_preview_20261017.xml'
OUTPUT_MANIFEST = OUTPUT_DIR / 'full_map_preview_20261017.json'

# Free-order competition is deliberately arranged into one connected showcase
# path. Every station remains in the same MuJoCo model and physical state.
STATIONS = (
    ('low_frame', 2.0, 0.0, '规则尺寸框架'),
    ('beam', 4.2, 0.0, '规则尺寸单木桥'),
    ('gully', 8.2, 0.0, '规则尺寸台阶/平台/沟，组合布局预览'),
    ('wall', 15.7, 0.0, '规则尺寸墙体'),
    ('plum_posts', 17.0, 0.0, '方块尺寸/间距规则，数量布局预览'),
    ('tire_array', 23.0, 0.0, '2×4 环形碰撞近似'),
    ('brick_wood_pit', 27.5, 0.0, '刚性块材料预览'),
    ('slalom', 32.0, 0.0, '规则尺寸六杆 L 路线'),
)
SPAWNS = {
    'start': (0.0, 0.0, 0.0),
    'low_frame': (1.25, 0.0, 0.0),
    'beam': (3.48, 0.0, 0.0),
    'gully': (7.45, 0.0, 0.0),
    'wall': (14.85, 0.0, 0.0),
    'plum_posts': (16.30, 0.0, 0.0),
    'tire_array': (22.25, 0.0, 0.0),
    'brick_wood_pit': (26.75, 0.0, 0.0),
    'slalom': (31.30, 0.0, 0.0),
}
QUALITY = {
    'low_frame': 'rule_size; frame-displacement threshold uncalibrated',
    'beam': 'rule_size; integrated entry state unvalidated',
    'gully': 'rule_size subparts; platform/slope sequence provisional',
    'wall': 'rule_size box; no validated wall skill',
    'plum_posts': 'rule_size blocks/gaps; block count/layout provisional',
    'tire_array': 'rule_size outer/inner diameter; 16-box ring approximation',
    'brick_wood_pit': 'rule_size envelope; rigid bark proxy, material uncalibrated',
    'slalom': 'rule_size L-shaped poles; zone/overflight scoring not integrated',
}
FRICTION = '1 0.005 0.0001'


def _v(values):
    return ' '.join(f'{float(v):.6f}' for v in values)


def _box(name, center, half, color, *, collision=True, quat=None):
    extra = f' quat="{_v(quat)}"' if quat else ''
    return (f'<geom name="rc27_full_{name}" type="box" pos="{_v(center)}" '
            f'size="{_v(half)}"{extra} friction="{FRICTION}" '
            f'contype="{1 if collision else 0}" '
            f'conaffinity="{1 if collision else 0}" rgba="{color}" />')


def _cylinder(name, center, radius, halfheight, color, *, collision=True):
    return (f'<geom name="rc27_full_{name}" type="cylinder" pos="{_v(center)}" '
            f'size="{radius:.6f} {halfheight:.6f}" friction="{FRICTION}" '
            f'contype="{1 if collision else 0}" '
            f'conaffinity="{1 if collision else 0}" rgba="{color}" />')


def _tile(geoms, name, left, right, half_width=3.0, color='0.54 0.59 0.62 1'):
    if right <= left:
        raise ValueError(f'empty ground tile {name}')
    geoms.append(_box(name, ((left + right) / 2, 0, -0.025),
                      ((right - left) / 2, half_width, 0.025), color))


def build_full_map(output_xml=OUTPUT_XML, output_manifest=OUTPUT_MANIFEST):
    rules = load_rules()
    specs = rules['obstacles']
    geoms = []
    # Collision ground has one real hole below the 360 mm gully gap.
    gap_left = STATIONS[2][1] + 3 * specs['gully']['stair_run_m'] + 0.80
    gap_right = gap_left + specs['gully']['gap_width_m']
    _tile(geoms, 'ground_before_gap', -2.5, gap_left)
    _tile(geoms, 'ground_after_gap', gap_right, 38.0)
    # Colored floor stripes are visual guides and never carry contact.
    palette = ('0.20 0.55 0.90 0.55', '0.15 0.70 0.32 0.55',
               '0.95 0.60 0.12 0.55', '0.78 0.25 0.25 0.55',
               '0.58 0.35 0.90 0.55', '0.30 0.65 0.68 0.55',
               '0.67 0.46 0.23 0.55', '0.90 0.30 0.62 0.55')
    for index, (_, x, _, _) in enumerate(STATIONS):
        geoms.append(_box(f'marker_{index+1}', (x - 0.55, 0, 0.004),
                          (0.025, 0.95, 0.004), palette[index], collision=False))

    # 1. Low frame: full roof and four corner posts, matching rule dimensions.
    x = STATIONS[0][1]
    frame = specs['low_frame']
    length = float(frame['length_m'])
    half_inner = float(frame['inner_width_m']) / 2
    clearance = float(frame['inner_height_m'])
    post_half = 0.025
    geoms.append(_box('low_frame_roof', (x + length/2, 0, clearance + post_half),
                      (length/2, half_inner + 2*post_half, post_half),
                      '0.95 0.38 0.20 1'))
    for end_x, side in ((x, -1), (x, 1), (x+length, -1), (x+length, 1)):
        geoms.append(_box(f'low_frame_post_{len(geoms)}',
                          (end_x, side*(half_inner+post_half), clearance/2),
                          (post_half, post_half, clearance/2), '0.92 0.28 0.18 1'))

    # 2. Rule-size elevated bridge.
    x = STATIONS[1][1]
    beam = specs['beam']
    geoms.append(_box('beam', (x + beam['length_m']/2, 0, beam['height_m']/2),
                      (beam['length_m']/2, beam['width_m']/2, beam['height_m']/2),
                      '0.13 0.70 0.28 1'))

    # 3. Three risers, 800 mm platform, true 360 mm empty gap, landing,
    #    three descending levels and a separate 20-degree ramp preview.
    x = STATIONS[2][1]
    gully = specs['gully']
    step_run = float(gully['stair_run_m'])
    step_rise = float(gully['stair_rise_m'])
    half_width = float(gully['stair_width_m'])/2
    for i in range(3):
        height = (i+1)*step_rise
        geoms.append(_box(f'gully_up_{i+1}',
                          (x+(i+0.5)*step_run, 0, height/2),
                          (step_run/2, half_width, height/2),
                          '0.86 0.58 0.20 1'))
    x += 3*step_run
    platform_length = float(gully['platform_length_m'])
    platform_height = float(gully['platform_height_m'])
    platform_half_width = float(gully['platform_width_m'])/2
    geoms.append(_box('gully_takeoff_platform',
                      (x+platform_length/2, 0, platform_height/2),
                      (platform_length/2, platform_half_width, platform_height/2),
                      '0.95 0.70 0.24 1'))
    x += platform_length + float(gully['gap_width_m'])
    geoms.append(_box('gully_landing_platform',
                      (x+platform_length/2, 0, platform_height/2),
                      (platform_length/2, platform_half_width, platform_height/2),
                      '0.95 0.70 0.24 1'))
    x += platform_length
    for i in range(2):
        height = (2-i)*step_rise
        geoms.append(_box(f'gully_down_{i+1}',
                          (x+(i+0.5)*step_run, 0, height/2),
                          (step_run/2, half_width, height/2),
                          '0.80 0.48 0.16 1'))
    slope_length = platform_height / math.tan(math.radians(gully['slope_angle_deg']))
    theta = math.radians(gully['slope_angle_deg'])
    slope_x = 13.15
    geoms.append(_box('gully_slope20_preview',
                      (slope_x+slope_length/2, 1.25, platform_height/2),
                      (math.hypot(slope_length, platform_height)/2, 0.40, 0.025),
                      '0.88 0.56 0.14 1',
                      quat=(math.cos(theta/2), 0, -math.sin(theta/2), 0)))

    # 4. 300 mm high wall, 800 mm wide and 80 mm thick.
    wall = specs['wall']
    x = STATIONS[3][1]
    geoms.append(_box('wall', (x, 0, wall['height_m']/2),
                      (wall['thickness_m']/2, wall['width_m']/2, wall['height_m']/2),
                      '0.78 0.24 0.23 1'))

    # 5. Two rows of alternating 200 mm blocks between 800 mm platforms.
    x = STATIONS[4][1]
    posts = specs['plum_posts']
    block_x, block_y, block_h = map(float, posts['block_size_m'])
    platform_x, platform_y, platform_h = map(float, posts['platform_size_m'])
    geoms.append(_box('plum_entry', (x+platform_x/2, 0, platform_h/2),
                      (platform_x/2, platform_y/2, platform_h/2), '0.54 0.36 0.78 1'))
    for row, y in enumerate((-(block_y + posts['inter_row_gap_m'])/2,
                              +(block_y + posts['inter_row_gap_m'])/2)):
        for col in range(4):
            center_x = x + platform_x + 0.20 + col*(block_x+posts['same_row_gap_m']) \
                + row*(block_x+posts['same_row_gap_m'])/2
            geoms.append(_box(f'plum_block_r{row+1}_c{col+1}',
                              (center_x, y, block_h/2),
                              (block_x/2, block_y/2, block_h/2),
                              '0.70 0.47 0.93 1'))
    geoms.append(_box('plum_exit', (x+3.9, 0, platform_h/2),
                      (platform_x/2, platform_y/2, platform_h/2),
                      '0.54 0.36 0.78 1'))

    # 6. Eight 16-segment box rings; the nominal 330 mm holes are physically
    #    empty. The rectangular segments are a reproducible mesh proxy.
    tire = specs['tire_array']
    center_radius = (tire['approximate_outer_diameter_m']
                     + tire['approximate_inner_diameter_m']) / 4
    radial_half = (tire['approximate_outer_diameter_m']
                   - tire['approximate_inner_diameter_m']) / 4
    tire_height = float(tire['nominal_tread_width_m'])
    x0 = STATIONS[5][1]
    for col in range(int(tire['columns'])):
        for row, y0 in enumerate((-0.34, 0.34)):
            cx = x0 + 0.33 + 0.65*col
            for seg in range(16):
                angle = 2*math.pi*seg/16
                tangent = angle + math.pi/2
                geoms.append(_box(
                    f'tire_c{col+1}_r{row+1}_s{seg+1}',
                    (cx+center_radius*math.cos(angle),
                     y0+center_radius*math.sin(angle), tire_height/2),
                    (0.5*(2*math.pi*center_radius/16)+0.005,
                     radial_half, tire_height/2),
                    '0.20 0.22 0.25 1',
                    quat=(math.cos(tangent/2), 0, 0, math.sin(tangent/2))))

    # 7. Rule-size envelope and guarded rigid-block proxy. Real bark sinkage
    #    and friction have not been measured, so this is visual-only material.
    pit = specs['brick_wood_pit']
    x = STATIONS[6][1]
    pit_len = float(pit['long_side_length_m'])
    pit_half_width = float(pit['short_side_width_m'])/2
    guard_h = float(pit['guard_height_m'])
    for side in (-1, 1):
        geoms.append(_box(f'pit_guard_{side}',
                          (x+pit_len/2, side*(pit_half_width+0.02), guard_h/2),
                          (pit_len/2, 0.02, guard_h/2), '0.50 0.31 0.18 1'))
    rng = random.Random(20271017)
    for ix in range(16):
        for iy in range(4):
            bx = x + (ix+0.5)*pit_len/16
            by = -pit_half_width + (iy+0.5)*(2*pit_half_width)/4
            top = rng.uniform(0.01, 0.02)
            geoms.append(_box(f'pit_block_{ix}_{iy}',
                              (bx, by, top/2),
                              (pit_len/32, pit_half_width/4, top/2),
                              '0.56 0.35 0.20 1' if (ix+iy)%2 else '0.63 0.42 0.26 1'))

    # 8. Exact pole count/diameter/spacing, same L arrangement as the
    #    single-obstacle 2027 evaluator. Colored zones are non-colliding.
    slalom = specs['slalom']
    x = STATIONS[7][1]
    pole_radius = float(slalom['pole_diameter_m'])/2
    pole_halfheight = float(slalom['pole_height_m'])/2
    spacing = float(slalom['pole_center_spacing_m'])
    for index, (px, py) in enumerate(
            [(x+i*spacing, 0.0) for i in range(4)]
            + [(x+3*spacing, i*spacing) for i in (1, 2)], start=1):
        geoms.append(_cylinder(f'slalom_pole_{index}',
                               (px, py, pole_halfheight),
                               pole_radius, pole_halfheight, '0.95 0.25 0.35 1'))
    zone_radius = float(slalom['required_zone_diameter_m'])/2
    for name, px, py in (('start', x-0.40, 0),
                         ('corner', x+3*spacing+0.30, -0.30),
                         ('finish', x+3*spacing, 2*spacing+0.40)):
        geoms.append(_cylinder(f'slalom_zone_{name}',
                               (px, py, 0.003), zone_radius, 0.003,
                               '0.10 0.45 0.85 0.50', collision=False))

    text = BASE_XML.read_text()
    text, removed = re.subn(
        r'^\s*<geom name="sim_(?:stair|three_stair)_[^"]*"[^>]*/>\s*\n',
        '', text, flags=re.M)
    if removed < 9:
        raise RuntimeError(f'base XML stair placeholders missing: {removed}')
    mesh_dir = (BASE_XML.parent / '../meshes').resolve()
    text, count = re.subn(r'meshdir="[^"]*"',
                          f'meshdir="{mesh_dir}"', text, count=1)
    if count != 1:
        raise RuntimeError('base XML meshdir missing')
    text = text.replace('<size njmax="500" nconmax="100" />',
                        '<size njmax="1000" nconmax="350" />', 1)
    floor_line = next(line for line in text.splitlines() if '<geom name="floor"' in line)
    visual_floor = floor_line.replace('contype="1"', 'contype="0"') \
        .replace('conaffinity="1"', 'conaffinity="0"')
    visual_floor = re.sub(r'rgba="[^"]*"', 'rgba="0.7 0.8 0.9 0.0"', visual_floor)
    text = text.replace(floor_line, visual_floor, 1)
    anchor = text.index('\n', text.index('<geom name="floor"'))+1
    text = text[:anchor] + '\n'.join('        '+geom for geom in geoms)+'\n' + text[anchor:]
    output_xml = Path(output_xml)
    output_manifest = Path(output_manifest)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    output_xml.write_text(text)
    manifest = {
        'kind': '2027_eight_obstacle_continuous_preview_not_rule_evaluation',
        'source_rules': str(ROOT/'robodog_gym/envs/robocon_2027/config/rules.yaml'),
        'xml': str(output_xml),
        'stations': [{'id': name, 'x_m': x, 'y_m': y, 'label': label,
                      'geometry_quality': QUALITY[name]}
                     for name, x, y, label in STATIONS],
        'spawns': {key: list(value) for key, value in SPAWNS.items()},
        'gap_interval_m': [gap_left, gap_right],
        'number_of_new_geoms': len(geoms),
        'notes': ['one physical MuJoCo scene; fixed display order; official obstacle order is free',
                  'only slalom/beam/low-frame have trained or research policy integrations',
                  'geometry/material proxies for gully/plum/tire/pit are not scoring evidence'],
    }
    output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return output_xml, manifest


if __name__ == '__main__':
    path, manifest = build_full_map()
    print(path)
    print(json.dumps({'stations': len(manifest['stations']),
                      'geoms': manifest['number_of_new_geoms'],
                      'gap_interval_m': manifest['gap_interval_m']},
                     ensure_ascii=False))
