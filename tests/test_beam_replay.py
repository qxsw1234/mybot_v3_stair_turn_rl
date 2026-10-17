"""Guard the bridge replay split and reward masking used by PPO."""
import unittest
from types import SimpleNamespace

import isaacgym  # Must precede torch in this project.
import numpy as np
import torch

from robodog_gym.envs.base.legged_robot import LeggedRobot
from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.rewards.corl_rewards import CoRLRewards
from robodog_gym.utils.terrain import Terrain


class BridgeReplayTest(unittest.TestCase):
    def test_fixed_terrain_columns_have_exact_geometry(self):
        cfg = Cfg.terrain()
        cfg.mesh_type = "heightfield"
        cfg.curriculum = False
        cfg.selected = False
        cfg.num_rows = 1
        cfg.num_cols = 20
        cfg.terrain_width = 8.0
        cfg.terrain_length = 8.0
        cfg.horizontal_scale = 0.05
        cfg.border_size = 3.0
        cfg.sim2sim_bridge = True
        cfg.bridge_replay = True
        cfg.bridge_train_cols = 15
        cfg.bridge_width = 0.8
        cfg.bridge_length = 1.5
        cfg.bridge_height = 0.03
        cfg.bridge_start_offset = 0.70
        terrain = Terrain(cfg, 512)
        maxima = []
        for column in range(20):
            x0 = cfg.border
            y0 = cfg.border + column * cfg.length_per_env_pixels
            tile = terrain.height_field_raw[
                x0:x0 + cfg.width_per_env_pixels,
                y0:y0 + cfg.length_per_env_pixels]
            maxima.append(round(float(tile.max() * cfg.vertical_scale), 3))
        self.assertEqual(maxima[:15], [0.03] * 15)
        self.assertEqual(maxima[15:17], [0.0, 0.0])
        self.assertEqual(maxima[17:], [0.64, 0.80, 0.96])
        np.testing.assert_allclose(cfg.env_origins[0, :, 2], 0.0)

    def test_bridge_rewards_do_not_touch_replay_columns(self):
        n = 20
        cfg = SimpleNamespace(terrain=SimpleNamespace(
            bridge_replay=True, bridge_train_cols=15,
            bridge_start_offset=0.70, bridge_length=1.5,
            bridge_width=0.8, bridge_edge_target_margin=0.08))
        env = SimpleNamespace(
            cfg=cfg, num_envs=n, device="cpu",
            terrain_types=torch.arange(n),
            base_pos=torch.zeros(n, 3),
            env_origins=torch.zeros(n, 3),
            root_states=torch.zeros(n, 13),
            foot_positions=torch.zeros(n, 4, 3),
            contact_forces=torch.zeros(n, 4, 3),
            feet_indices=torch.arange(4),
        )
        env._bridge_env_mask = lambda: LeggedRobot._bridge_env_mask(env)
        env.base_pos[:, 0] = 1.0
        env.root_states[:, 7] = 0.3
        env.foot_positions[:, :, 0] = 1.0
        env.foot_positions[:, :, 1] = 0.38
        env.contact_forces[:, :, 2] = 2.0
        rewards = CoRLRewards(env)
        progress = rewards._reward_bridge_progress()
        edge = rewards._reward_bridge_foot_edge()
        self.assertTrue(torch.all(progress[:15] > 0))
        self.assertTrue(torch.all(edge[:15] > 0))
        self.assertTrue(torch.all(progress[15:] == 0))
        self.assertTrue(torch.all(edge[15:] == 0))


if __name__ == "__main__":
    unittest.main()
