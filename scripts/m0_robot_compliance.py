#!/usr/bin/env python3
"""M0 robot compliance audit for mybot_v3 (RoboCon 2026 仿生足式障碍赛).

Checks vs rules:
  1) total mass <= 35 kg
  2) natural standing envelope <= 800 x 600 x 600 mm
  3) foot ground-contact circle diameter <= 80 mm
  4) crouch clearance: can the whole robot fit under a 300 mm bar (限高杆)?

Method: MuJoCo (mesh-accurate) kinematic evaluation + mirror-symmetric pose
search with all four feet on the ground.
Run:  /home/ldl/anaconda3/envs/robodog_gym/bin/python scripts/m0_robot_compliance.py
"""
import itertools
import struct
import numpy as np
import mujoco

XML = "/home/ldl/mybot_v3_stair_turn_rl/resources/robots/mybot_v3/xml/mybot_v3.xml"
STL_FOOT = "/home/ldl/mybot_v3_stair_turn_rl/resources/robots/mybot_v3/meshes/FL_foot.STL"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

print("=" * 72)
print("MYBOT V3 合规审计 (MuJoCo kinematic)")
print("=" * 72)

# ---------- 1. mass ----------
total_mass = float(m.body_mass.sum())
print(f"\n[1] 总质量: {total_mass:.2f} kg   (限制 <= 35 kg)  ->  {'PASS' if total_mass <= 35 else 'FAIL'}")

# ---------- helpers ----------
def geom_world_verts(gid):
    t = m.geom_type[gid]
    xpos = d.geom_xpos[gid]
    xmat = d.geom_xmat[gid].reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n]
        s = m.geom_size[gid][:3].copy()
        s[s == 0] = 1.0
        return v * s @ xmat.T + xpos
    else:
        size = m.geom_size[gid]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = size[0]
            pts = np.array([[r,0,0],[-r,0,0],[0,r,0],[0,-r,0],[0,0,r],[0,0,-r]])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = size[0], size[1]
            pts = np.array([[r,0,h],[-r,0,h],[0,r,h],[0,-r,h],[0,0,h+r],
                            [r,0,-h],[-r,0,-h],[0,r,-h],[0,-r,-h],[0,0,-h-r]])
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = size
            pts = np.array(list(itertools.product([-sx, sx], [-sy, sy], [-sz, sz])))
        else:
            pts = np.array([[0, 0, 0]], dtype=float)
        return pts @ xmat.T + xpos

ROBOT_GIDS = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]

def feet_ids():
    ids = []
    for side in ("FL", "FR", "RL", "RR"):
        bid = m.body(f"{side}_foot").id
        ids += [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    return ids

FOOT_GIDS = feet_ids()

def all_robot_points():
    return np.vstack([geom_world_verts(g) for g in ROBOT_GIDS])

def place_base_on_feet():
    pts = np.vstack([geom_world_verts(g) for g in FOOT_GIDS])
    minz = pts[:, 2].min()
    d.qpos[2] -= minz
    mujoco.mj_forward(m, d)

def set_mirror_pose(tf, cf, tb=None, cb=None, hip=0.1):
    """mirror-symmetric pose: FL/RL=+hip, FR/RR=-hip; front uses (tf,cf), rear (tb,cb)."""
    if tb is None:
        tb = tf
    if cb is None:
        cb = cf
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    jid = lambda nm: m.joint(nm).id  # noqa: E731
    for side, h, t, c in (("FL", hip, tf, cf), ("FR", -hip, tf, cf),
                          ("RL", hip, tb, cb), ("RR", -hip, tb, cb)):
        d.qpos[m.jnt_qposadr[jid(f"{side}_hip_joint")]] = h
        d.qpos[m.jnt_qposadr[jid(f"{side}_thigh_joint")]] = t
        d.qpos[m.jnt_qposadr[jid(f"{side}_calf_joint")]] = c
    mujoco.mj_forward(m, d)

# ---------- 2. standing envelope at default pose ----------
set_mirror_pose(0.8, -1.5)
d.qpos[2] = 0.34
mujoco.mj_forward(m, d)
pts = all_robot_points()
ext = pts.max(0) - pts.min(0)
print(f"\n[2] 自然站立(默认姿态 thigh=0.8,calf=-1.5, base_z=0.34) 外廓 (长x宽x高): "
      f"{ext[0]*1000:.0f} x {ext[1]*1000:.0f} x {ext[2]*1000:.0f} mm   (限制 800x600x600)")

set_mirror_pose(0.8, -1.5)
place_base_on_feet()
pts2 = all_robot_points()
ext2 = pts2.max(0) - pts2.min(0)
print(f"    足底贴地站立: 外廓 {ext2[0]*1000:.0f} x {ext2[1]*1000:.0f} x {ext2[2]*1000:.0f} mm, "
      f"机身最高点 {pts2[:,2].max()*1000:.0f} mm, base_z={d.qpos[2]*1000:.0f} mm")

# ---------- 3. foot size ----------
def stl_bbox(path):
    with open(path, "rb") as f:
        head = f.read(84)
        ntri = struct.unpack("<I", head[80:84])[0]
        data = np.frombuffer(f.read(ntri * 50), dtype=np.uint8).reshape(ntri, 50)
        tri = data[:, 12:48].copy().view("<f4").reshape(ntri, 3, 3)
    v = tri.reshape(-1, 3)
    return v.min(0), v.max(0)

lo, hi = stl_bbox(STL_FOOT)
size = (hi - lo) * 1000
print(f"\n[3] 足端 FL_foot.STL 包围盒: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm")
circ_d = float(np.hypot(size[0], size[1]))
print(f"    水平外接圆直径 ≈ {circ_d:.1f} mm   (限制 <= 80 mm)  ->  {'PASS' if circ_d <= 80 else 'FAIL'}")

# ---------- 4. crouch clearance search (mirror-symmetric pose, exact eval) ----------
print("\n[4] 蹲姿可达性扫描: 四腿镜像对称折叠, 四足全部贴地")
print("    髋符号约定: FL/RL=+h, FR/RR=-h (与训练默认一致), 目标: 全身最高点 < 300 mm")

GRID_T = np.arange(-1.0, 4.15, 0.05)
GRID_C = np.arange(-2.68, -0.92, 0.05)

def all_leg_geom_ids():
    ids = []
    for p in ("FL", "FR", "RL", "RR"):
        bids = [m.body(f"{p}_{part}").id for part in ("hip", "thigh", "calf", "foot")]
        ids += [g for g in range(m.ngeom) if m.geom_bodyid[g] in bids]
    return ids

LEG_GIDS = all_leg_geom_ids()
BASE_GIDS = [g for g in ROBOT_GIDS if g not in set(LEG_GIDS)]

d.qpos[:] = 0
d.qpos[3] = 1.0
mujoco.mj_forward(m, d)
base_pts = np.vstack([geom_world_verts(g) for g in BASE_GIDS])
BASE_MIN, BASE_TOP = base_pts[:, 2].min(), base_pts[:, 2].max()
print(f"    (机身 z 范围相对原点: {BASE_MIN*1000:.1f} ~ {BASE_TOP*1000:.1f} mm; 腿几何数={len(LEG_GIDS)})")

LAST_DBG = {}
FOOT_BY_SIDE = {s: [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{s}_foot").id]
                for s in ("FL", "FR", "RL", "RR")}
FOOT_SET = set(FOOT_GIDS)

def eval_config(t, c, hip, exact=False):
    """(ok, top, base_z). Exact: place so the robot's lowest point touches z=0;
    ok requires: top<300mm, all four feet touching (<=10mm), no part below floor."""
    set_mirror_pose(t, c, hip=hip)
    if not exact:
        leg = np.vstack([geom_world_verts(g) for g in LEG_GIDS])
        fz = [geom_world_verts(g)[:, 2].min() for g in FOOT_GIDS]
        bz = -min(fz)
        top_est = max(BASE_TOP, leg[:, 2].max()) + bz
        nf_min = min(BASE_MIN, leg[:, 2].min()) + bz
        return (top_est < 0.30 + 1e-9 and nf_min > -0.002), top_est, bz
    allv = all_robot_points()
    d.qpos[2] -= allv[:, 2].min()
    mujoco.mj_forward(m, d)
    allv = all_robot_points()
    top = float(allv[:, 2].max())
    feet_mins = {s: float(min(geom_world_verts(g)[:, 2].min() for g in FOOT_BY_SIDE[s])) for s in FOOT_BY_SIDE}
    nonfoot = np.vstack([geom_world_verts(g) for g in ROBOT_GIDS if g not in FOOT_SET])
    ok = top < 0.30 and max(feet_mins.values()) <= 0.010 and nonfoot[:, 2].min() >= -0.001
    LAST_DBG.update(feet=feet_mins, nonfoot_min=float(nonfoot[:, 2].min()), top=top, bz=float(d.qpos[2]))
    return ok, top, d.qpos[2]

# stage 1: coarse scan at hip=0.1 (training default)
cands = []
for t in GRID_T:
    for c in GRID_C:
        ok, top_est, bz = eval_config(t, c, 0.1, exact=False)
        if ok:
            cands.append((top_est, t, c))
cands.sort()
print(f"    预筛(代理指标)通过组合数: {len(cands)} / {len(GRID_T)*len(GRID_C)}"
      + (f", 最优 top≈{cands[0][0]*1000:.1f} mm" if cands else ""))

# stage 2: exact vertex evaluation for the best 120
results = []
for i, (top_est, t, c) in enumerate(cands[:120]):
    ok, zmax, bz = eval_config(round(t, 3), round(c, 3), 0.1, exact=True)
    if i < 6:
        feet = {k: round(v * 1000, 1) for k, v in LAST_DBG['feet'].items()}
        print(f"      [dbg{i}] t={t:.2f} c={c:.2f} est={top_est*1000:.0f} ok={ok} "
              f"feet={feet} nonfoot={LAST_DBG['nonfoot_min']*1000:.1f} top={LAST_DBG['top']*1000:.1f}")
    if ok:
        results.append((zmax, t, c, 0.1, bz))
results.sort(key=lambda r: r[0])
print(f"    精确评估有效姿态数: {len(results)} (最佳 120 个中)")
if results:
    print("    最优 5 个 (精确):")
    for zmax, t, c, hip, bz in results[:5]:
        print(f"      top={zmax*1000:6.1f} mm  thigh={t:5.2f}  calf={c:5.2f}  hip={hip:5.2f}  base_z={bz*1000:6.1f} mm")

# stage 3: refine around best over thigh/calf/hip
final_top = float("nan")
if results:
    _, t0, c0, hip0, _ = results[0]
    cur = results[0]
    for it in range(2):
        _, t0, c0, hip0, _ = cur
        for hip in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
            for t in np.arange(max(-1.0, t0 - 0.10), min(4.15, t0 + 0.10) + 1e-9, 0.02):
                for c in np.arange(max(-2.68, c0 - 0.10), min(-0.92, c0 + 0.10) + 1e-9, 0.02):
                    ok, zmax, bz = eval_config(round(t, 3), round(c, 3), hip, exact=True)
                    if ok and zmax < cur[0]:
                        cur = (zmax, round(t, 3), round(c, 3), hip, bz)
    zmax, t, c, hip, bz = cur
    final_top = zmax
    print(f"\n    精调最优: top={zmax*1000:.1f} mm  thigh={t:.3f}  calf={c:.3f}  hip={hip}  base_z={bz*1000:.1f} mm")
    print(f"    -> 最终: {'PASS' if zmax < 0.30 else 'FAIL'} (限高杆 300mm, 余量 {(0.30-zmax)*1000:.0f} mm)")
    import os
    os.makedirs("/home/ldl/mybot_v3_stair_turn_rl/logs", exist_ok=True)
    with open("/home/ldl/mybot_v3_stair_turn_rl/logs/m0_crouch_pose.txt", "w") as f:
        f.write(f"thigh={t}\ncalf={c}\nhip=+/-{hip} (FL/RL=+{hip}, FR/RR=-{hip})\n"
                f"top={zmax}\nbase_z={bz}\n")
    print("\n    可行域抽样 (hip=0.1, 全身最高点 mm; -- 表示不可行或>300):")
    for t in np.arange(1.2, 4.0, 0.2):
        row = []
        for c in np.arange(-2.6, -0.95, 0.15):
            ok, zmax2, bz2 = eval_config(round(t, 3), round(c, 3), 0.1, exact=True)
            row.append(f"{zmax2*1000:4.0f}" if ok else "  --")
        print(f"      thigh={t:4.2f}: " + " ".join(row))
else:
    print("    !! 未找到满足约束的蹲姿")

# ---------- summary ----------
print("\n" + "=" * 72)
print("SUMMARY")
print(f"  mass          : {total_mass:.2f} kg  (<=35)          {'PASS' if total_mass<=35 else 'FAIL'}")
print(f"  envelope      : {ext2[0]*1000:.0f}x{ext2[1]*1000:.0f}x{ext2[2]*1000:.0f} mm (<=800x600x600) "
      f"{'PASS' if ext2[0]<=0.800 and ext2[1]<=0.600 and ext2[2]<=0.600 else 'CHECK'}")
print(f"  foot circle   : {circ_d:.1f} mm  (<=80)            {'PASS' if circ_d<=80 else 'FAIL'}")
print(f"  crouch <300mm : min top {final_top*1000:.0f} mm          {'PASS' if final_top<0.30 else 'FAIL'}")
print("=" * 72)
