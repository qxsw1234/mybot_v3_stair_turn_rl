# Mybot V3 stair and turning controller

This is an isolated ELMAP based reinforcement learning project for Mybot V3.
The policy receives proprioception history and a 77 point local terrain height
map. Its training curriculum covers:

- ascending stairs;
- descending stairs;
- forward and reverse motion;
- lateral motion;
- left and right turns, including turning in place;
- stopping and standing still.

## Training design

- 4096 Isaac Gym environments by default.
- 12 terrain difficulty levels.
- Terrain mix: 34% ascending stairs, 34% descending stairs, 10% discrete
  obstacles, 8% smooth flat, 6% rough flat, and 8% slopes.
- Stair height curriculum starts near 8 cm and reaches about 20 cm.
- Velocity commands begin conservatively and can expand to `vx` -0.70 to
  0.90 m/s, `vy` -0.30 to 0.30 m/s, and yaw -1.0 to 1.0 rad/s.
- Moderate friction, payload, center of mass, motor, sensor height map, gravity,
  and control delay randomization for later sim to real transfer.
- Checkpoints every 250 iterations.
- Headless training disables camera recording and per-step GPU-to-NumPy
  telemetry copies; deployment and data collection keep telemetry enabled by
  default.
- GPU physics avoids blocking CPU-style result fetches between control
  substeps, and curriculum plots are written with checkpoints instead of every
  ten iterations.
- New runs start from random weights. Continuation runs can load an explicit
  local checkpoint with `--resume-run` and `--checkpoint`.
- Terrain difficulty advances from linear and angular velocity tracking error,
  so commanded turns do not cause false terrain demotion.

## Start training

```bash
cd /home/ldl/mybot_v3_stair_turn_rl
./scripts/start_mybot_v3_training.sh
```

The default run is 60,000 iterations with seed 42. Optional arguments are
forwarded to the Python entry point, for example:

```bash
./scripts/start_mybot_v3_training.sh --num-envs 2048 --iterations 30000 --seed 7
```

Useful status commands:

```bash
cat logs/training.pid
tail -f logs/training_latest.log
nvidia-smi
```

Runs and checkpoints are stored under:

```text
runs/mybot_v3_stair_turn_from_scratch/<timestamp>/
```

The main configuration entry point is
`scripts/train_mybot_v3_stair_turn.py`.

## Continue from a checkpoint

```bash
./scripts/start_mybot_v3_training.sh \
  --resume-run /absolute/path/to/run \
  --checkpoint 3000 \
  --iterations 57000
```

The continuation uses the checkpoint number as its first global iteration and
stores output in a separate `velocity_curriculum_resume_*` run directory.

## View the trained policy on stairs

The interactive stair viewer loads one checkpoint and fixes the environment to
one ascending-stair difficulty level.  It does not modify the `.pt` weights;
the first run also generates indexed JIT files beside the checkpoint.

```bash
source /home/ldl/anaconda3/etc/profile.d/conda.sh
conda activate robodog_gym
cd /home/ldl/mybot_v3_stair_turn_rl

python -u scripts/view_stairs.py --iteration 37500 --level 8
```

`--level 4`, `--level 8`, and `--level 11` correspond to approximately 10.4,
15.8, and 19.9 cm steps.  Use `--iteration 39250` to inspect the latest saved
checkpoint.  In the viewer, `W/S` moves forward/backward, `A/D` moves laterally,
`Q/E` turns, and the arrow keys apply pushes.
