#!/bin/bash
source /home/ldl/anaconda3/etc/profile.d/conda.sh
conda activate robodog_gym
cd /home/ldl/mybot_v3_stair_turn_rl
export DISPLAY=:0
export ML_LOGGER_USER=training
exec python scripts/launch_phase5.py
