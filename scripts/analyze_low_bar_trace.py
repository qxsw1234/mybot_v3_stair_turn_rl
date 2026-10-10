#!/usr/bin/env python3
"""Diagnose Low Bar bar contacts from an ``eval_low_bar_isaac.py`` trace.

Answers the M1.3 questions with measurements instead of impressions:

1. Where is the robot when the bar contact happens?
2. Does it lower its body before the bar, and by how much?
3. Does the whole robot (by rigid-body frame height) ever fit under the bar?
4. How close are the actions to saturation while crossing?

Two modelling details matter and are handled here:

* ``env.step()`` resets terminated environments itself, so on the terminating
  step the recorded pose is already the spawn pose again.  Only steps flagged
  ``active`` describe the attempt, and the pose that produced the contact is the
  last active frame, one step before the termination.
* The bar bottom sits ``clearance_m`` above the ground, so ``max_body_z`` above
  the clearance means the robot's body is above the bar's lower edge.
"""

import argparse
import json
import sys

import numpy as np


def _quantiles(values, points=(5, 25, 50, 75, 95)):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(array, p)), 4) for p in points}


def _trajectory(x_rel, base_z, body_z, force, first_contact, mask, length=12):
    """The last active frames of each terminating environment."""
    rows = []
    for env in np.flatnonzero(mask & (first_contact >= 0)):
        stop = int(first_contact[env])
        start = max(stop - length, 0)
        rows.append({
            "env": int(env),
            "termination_step": stop,
            "x_rel_m": [round(float(v), 3) for v in x_rel[start:stop, env]],
            "base_z_m": [round(float(v), 3) for v in base_z[start:stop, env]],
            "max_body_z_m": [round(float(v), 3)
                             for v in body_z[start:stop, env]],
            "bar_force_n": [round(float(v), 1) for v in force[start:stop, env]],
            "_start_step": start,
        })
    return rows


# A rigid body near the bar plane and above the bar's lower edge is the body
# that produced the bar contact; the net contact force tensor cannot separate
# ground contacts from bar contacts per body, so geometry is used instead.
BAR_PLANE_TOLERANCE_M = 0.35
# Half-thickness of the cross-bar collision box along x (low_bar.urdf is a
# 0.05 x 1.0 x 0.05 box, so the bar spans +-0.025 m about its origin).
BAR_HALF_THICKNESS_M = 0.025


def _bar_box_distance(x_rel, z, clearance):
    """Distance in the x-z plane from a body to the cross-bar box.

    The box spans ``|x| <= BAR_HALF_THICKNESS_M`` and
    ``clearance <= z <= clearance + 0.05``.  The two posts share the same rigid
    body but sit at |y| = 0.475 m, which a centred robot never reaches, so the
    cross-bar box is the right target here.
    """
    dx = np.maximum(np.abs(x_rel) - BAR_HALF_THICKNESS_M, 0.0)
    dz = np.maximum(np.maximum(clearance - z, z - (clearance + 0.05)), 0.0)
    return np.sqrt(dx * dx + dz * dz)


def _contact_snapshots(payload):
    """Attribute the first bar contact to a robot body using live geometry."""
    body_names = [str(name) for name in payload["body_names"]]
    snapshots = json.loads(str(payload.get("contact_snapshots_json", "[]")))
    rows = []
    counts = {}
    for snapshot in snapshots:
        clearance = np.asarray(snapshot["clearance"], dtype=np.float64)
        for row, env in enumerate(snapshot["env"]):
            x_rel = np.asarray(snapshot["x_rel"][row], dtype=np.float64)
            z = np.asarray(snapshot["body_z"][row], dtype=np.float64)
            distance = _bar_box_distance(x_rel, z, float(clearance[row]))
            body = int(np.argmin(distance))
            name = body_names[body] if body < len(body_names) else f"body{body}"
            counts[name] = counts.get(name, 0) + 1
            rows.append({
                "env": int(env),
                "step": int(snapshot["step"]),
                "base_x_rel_m": round(float(snapshot["base_x_rel"][row]), 3),
                "clearance_m": round(float(clearance[row]), 4),
                "bar_force_n": round(float(snapshot["bar_force_n"][row]), 2),
                "closest_body": name,
                "closest_x_rel_m": round(float(x_rel[body]), 3),
                "closest_z_m": round(float(z[body]), 3),
                "closest_distance_m": round(float(distance[body]), 3),
                "closest_z_minus_clearance_m":
                    round(float(z[body] - clearance[row]), 3),
                "body_positions": {
                    body_names[i]: [round(float(x_rel[i]), 3), round(float(z[i]), 3)]
                    for i in range(min(len(x_rel), len(body_names)))
                },
            })
    return {
        "n_contact_snapshots": len(rows),
        "closest_body_counts": dict(
            sorted(counts.items(), key=lambda item: -item[1])),
        "events": rows,
    }


def _bar_contact_bodies(payload, first_contact):
    """Body geometry at each bar-contact termination.

    The bar's own net contact force says *that* the bar was touched; the poses
    of the robot's bodies one frame earlier say *which* part touched it.
    """
    body_names = [str(name) for name in payload["body_names"]]
    events = json.loads(str(payload["termination_events_json"]))
    clearance = payload["clearance_m"]
    rows = []
    for event in events:
        previous_x = event.get("prev_body_x_rel")
        previous_z = event.get("prev_body_z")
        if previous_x is None:
            continue
        for row, env in enumerate(event["env"]):
            if int(event["step"]) != int(first_contact[env]):
                continue
            x = np.asarray(previous_x[row], dtype=np.float64)
            z = np.asarray(previous_z[row], dtype=np.float64)
            near_plane = np.abs(x) <= BAR_PLANE_TOLERANCE_M
            above_edge = z - float(clearance[env])
            candidate = int(np.argmax(np.where(near_plane, z, -np.inf))) \
                if near_plane.any() else None
            rows.append({
                "env": int(env),
                "step": int(event["step"]),
                "clearance_m": round(float(clearance[env]), 4),
                "base_x_rel_m": round(float(x[0]), 3),
                "nearest_to_bar": body_names[candidate] if candidate is not None
                else None,
                "nearest_x_rel_m": round(float(x[candidate]), 3)
                if candidate is not None else None,
                "nearest_z_m": round(float(z[candidate]), 3)
                if candidate is not None else None,
                "nearest_z_over_edge_m": round(float(above_edge[candidate]), 3)
                if candidate is not None else None,
                "bodies_near_plane": {
                    body_names[i]: [round(float(x[i]), 3), round(float(z[i]), 3)]
                    for i in range(len(x)) if near_plane[i]
                },
            })
    counts = {}
    for row in rows:
        name = row["nearest_to_bar"]
        counts[name] = counts.get(name, 0) + 1
    return {
        "n_bar_contact_terminations": len(rows),
        "nearest_body_counts": dict(
            sorted(counts.items(), key=lambda item: -(item[1] or 0))),
        "z_over_bar_edge_m": _quantiles(
            [row["nearest_z_over_edge_m"] for row in rows
             if row["nearest_z_over_edge_m"] is not None]),
        "events": rows,
    }


def summarize(payload, window, approach, force_threshold, saturation):
    trace = payload["trace"]                                  # steps, envs, 6
    bar_x = payload["bar_x"]
    clearance = payload["clearance_m"]
    records = json.loads(str(payload["records_json"]))
    fields = {name: index for index, name in enumerate(
        str(x) for x in payload["fields"])}

    base_x = trace[:, :, fields["base_x"]]
    base_z = trace[:, :, fields["base_z"]]
    body_z = trace[:, :, fields["max_body_z"]]
    force = trace[:, :, fields["bar_force_n"]]
    action = trace[:, :, fields["action_absmax"]]
    active = trace[:, :, fields["active"]] > 0.5

    x_rel = base_x - bar_x[None, :]
    margin = clearance[None, :] - body_z
    crossing = (x_rel >= window[0]) & (x_rel <= window[1]) & active
    nearing = (x_rel >= approach[0]) & (x_rel <= approach[1]) & active

    collided = np.array([r["collision"] for r in records], dtype=bool)
    success = np.array([r["success"] for r in records], dtype=bool)

    max_body_z = np.max(np.where(active, body_z, -np.inf), axis=0)
    min_margin = np.min(np.where(active, margin, np.inf), axis=0)
    min_base_z = np.min(np.where(active, base_z, np.inf), axis=0)
    max_x_rel = np.max(np.where(active, x_rel, -np.inf), axis=0)
    approach_base_z = np.mean(np.where(nearing, base_z, np.nan), axis=0)
    crouch = approach_base_z - min_base_z
    crossing_values = action[crossing] if crossing.any() else np.array([])

    hits = (force > force_threshold) & active
    contact_steps = np.where(hits, np.arange(len(force))[:, None], len(force))
    first_contact = contact_steps.min(axis=0)
    first_contact = np.where(first_contact == len(force), -1, first_contact)
    # The last frame that still describes the attempt.
    last_active_frame = np.maximum(first_contact - 1, 0)
    touched = first_contact >= 0

    def at(values):
        return values[last_active_frame[touched], np.flatnonzero(touched)] \
            if touched.any() else np.array([])

    def subset(mask):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return {"n": 0}
        cross_values = action[:, mask][crossing[:, mask]]
        limit = clip_actions if clip_actions is not None else saturation
        return {
            "n": int(mask.sum()),
            "mean_clearance_m": round(float(np.mean(clearance[mask])), 4),
            "approach_base_z_m": _quantiles(approach_base_z[mask]),
            "min_base_z_m": _quantiles(min_base_z[mask]),
            "crouch_vs_approach_m": _quantiles(crouch[mask]),
            "max_body_z_m": _quantiles(max_body_z[mask]),
            "min_bar_margin_m": _quantiles(min_margin[mask]),
            "fits_under_bar_fraction": round(
                float(np.mean(min_margin[mask] > 0.0)), 4),
            "max_x_rel_reached_m": _quantiles(max_x_rel[mask]),
            "crossing_action_absmax_median": round(
                float(np.median(cross_values)), 4) if len(cross_values) else None,
            "crossing_action_absmax_p95": round(
                float(np.percentile(cross_values, 95)), 4)
            if len(cross_values) else None,
            # clip_actions is 100 for this project, so this fraction is expected
            # to be 0: the actor output is not clipped and the policy is not
            # amplitude-limited.  Kept so the claim is measured, not assumed.
            "crossing_action_clipping_fraction": round(
                float(np.mean(np.abs(cross_values) > limit)), 4)
            if len(cross_values) else None,
            "crossing_steps_observed": int(crossing[:, mask].sum()),
        }

    clip_actions = float(payload["clip_actions"]) if "clip_actions" in payload \
        else None
    return {
        "n_envs": int(len(bar_x)),
        "collision_rate": round(float(collided.mean()), 4),
        "success_rate": round(float(success.mean()), 4),
        "envs_with_any_bar_contact": int(touched.sum()),
        "contact_frame_is_last_active": True,
        "contact_x_rel_m": _quantiles(at(x_rel)),
        "contact_base_z_m": _quantiles(at(base_z)),
        "contact_max_body_z_m": _quantiles(at(body_z)),
        "contact_force_n": _quantiles(at(force)),
        "contact_t_s": _quantiles(
            first_contact[touched] * float(payload["dt_s"])),
        "all_envs": subset(np.ones(len(bar_x), dtype=bool)),
        "collided": subset(collided),
        "clean": subset(~collided),
        "window_m": list(window),
        "approach_window_m": list(approach),
        "force_threshold_n": force_threshold,
        "clip_actions": clip_actions,
        "action_scale": float(payload["action_scale"])
        if "action_scale" in payload else None,
        "saturation_threshold": saturation,
        "trajectory": _trajectory(x_rel, base_z, body_z, force,
                                  first_contact, collided)[:4],
        "bar_contact_bodies": _bar_contact_bodies(payload, first_contact),
        "contact_snapshots": _contact_snapshots(payload),
        "_arrays": {
            "x_rel": x_rel, "base_z": base_z, "body_z": body_z,
            "force": force, "active": active, "first_contact": first_contact,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="npz written by eval_low_bar_isaac.py")
    parser.add_argument("--window", type=float, nargs=2, default=[-0.25, 0.35],
                        help="Base x relative to the bar plane, in metres.")
    parser.add_argument("--approach", type=float, nargs=2, default=[-1.00, -0.70],
                        help="Approach window used as the walking stance height.")
    parser.add_argument("--force-threshold", type=float, default=1.0)
    parser.add_argument("--saturation", type=float, default=0.95)
    parser.add_argument("--out")
    parser.add_argument("--show-trajectories", action="store_true")
    args = parser.parse_args()

    data = np.load(args.trace, allow_pickle=False)
    results = {}
    for key in data.files:
        if not key.endswith("::trace"):
            continue
        prefix = key[: -len("::trace")]
        payload = {name.split("::", 1)[1]: data[name]
                   for name in data.files if name.startswith(prefix + "::")}
        payload["trace"] = data[key]
        results[prefix] = summarize(payload, tuple(args.window),
                                    tuple(args.approach),
                                    args.force_threshold, args.saturation)

    for prefix, block in results.items():
        print(f"\n===== {prefix} =====")
        print(f"  envs={block['n_envs']} collision={block['collision_rate']} "
              f"success={block['success_rate']} "
              f"any_bar_contact={block['envs_with_any_bar_contact']}")
        print(f"  contact x_rel [m]      {block['contact_x_rel_m']}")
        print(f"  contact t [s]          {block['contact_t_s']}")
        print(f"  contact base z [m]     {block['contact_base_z_m']}")
        print(f"  contact max body z [m] {block['contact_max_body_z_m']}")
        print(f"  contact bar force [N]  {block['contact_force_n']}")
        bodies = block["bar_contact_bodies"]
        print(f"  bar-contact terminations observed: "
              f"{bodies['n_bar_contact_terminations']}")
        print(f"      nearest body to the bar plane: "
              f"{bodies['nearest_body_counts']}")
        print(f"      that body's z above the bar edge: "
              f"{bodies['z_over_bar_edge_m']}")
        for row in bodies["events"][:8]:
            print(f"      env {row['env']} step {row['step']} "
                  f"clearance {row['clearance_m']} "
                  f"near={row['nearest_to_bar']} "
                  f"x_rel={row['nearest_x_rel_m']} z={row['nearest_z_m']} "
                  f"z_over_edge={row['nearest_z_over_edge_m']}")
            print(f"          bodies near plane: {row['bodies_near_plane']}")
        snaps = block["contact_snapshots"]
        print(f"  first-contact snapshots: {snaps['n_contact_snapshots']} "
              f"closest body {snaps['closest_body_counts']}")
        for row in snaps["events"][:8]:
            print(f"      env {row['env']} step {row['step']} "
                  f"clearance {row['clearance_m']} base_x_rel "
                  f"{row['base_x_rel_m']} force {row['bar_force_n']}N -> "
                  f"{row['closest_body']} "
                  f"x_rel={row['closest_x_rel_m']} z={row['closest_z_m']} "
                  f"gap={row['closest_distance_m']} "
                  f"(z-clearance={row['closest_z_minus_clearance_m']})")
            if args.show_trajectories:
                print(f"          {row['body_positions']}")
        for name in ("all_envs", "collided", "clean"):
            group = block[name]
            if group.get("n", 0) == 0:
                print(f"  {name}: none")
                continue
            print(f"  {name} (n={group['n']}, "
                  f"clearance {group['mean_clearance_m']} m)")
            print(f"      approach base z [m]   {group['approach_base_z_m']}")
            print(f"      min base z [m]        {group['min_base_z_m']}")
            print(f"      crouch vs approach[m] {group['crouch_vs_approach_m']}")
            print(f"      max body z [m]        {group['max_body_z_m']}")
            print(f"      min bar margin [m]    {group['min_bar_margin_m']}")
            print(f"      fits under bar        "
                  f"{group['fits_under_bar_fraction']}")
            print(f"      max x_rel reached [m] {group['max_x_rel_reached_m']}")
            print(f"      crossing action |max|: median "
                  f"{group['crossing_action_absmax_median']} "
                  f"p95 {group['crossing_action_absmax_p95']} "
                  f"clipped@"
                  f"{block['clip_actions']} "
                  f"{group['crossing_action_clipping_fraction']} "
                  f"(scale {block['action_scale']}, "
                  f"steps {group['crossing_steps_observed']})")
        if args.show_trajectories:
            for row in block["trajectory"]:
                print(f"  env {row['env']} stops at step "
                      f"{row['termination_step']} (from step {row['_start_step']})")
                print(f"      x_rel    {row['x_rel_m']}")
                print(f"      base_z   {row['base_z_m']}")
                print(f"      body_z   {row['max_body_z_m']}")
                print(f"      force    {row['bar_force_n']}")

    if args.out:
        serializable = {prefix: {k: v for k, v in block.items()
                                 if k != "_arrays"}
                        for prefix, block in results.items()}
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
