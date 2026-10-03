# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice

from robodog_gym.envs.base.legged_robot_config import Cfg


def straight_stairs_terrain(terrain, step_width, step_height,
                            start_offset=0.55, stair_width=2.0,
                            num_steps=8, top_length=0.90):
    """Build the same forward, finite-width staircase used by MuJoCo.

    The robot is spawned at the heightfield centre.  The first riser is
    ``start_offset`` metres in front of that point and every subsequent tread
    rises by ``step_height``.  Keeping this generator in the shared terrain
    module makes the train/evaluation geometry explicit instead of relying on
    Isaac Gym's centred pyramid stairs (where a robot spawned at the centre
    starts on the *top* platform).
    """
    horizontal_scale = float(terrain.horizontal_scale)
    vertical_scale = float(terrain.vertical_scale)
    tread_pixels = max(1, int(round(step_width / horizontal_scale)))
    start_x = terrain.width // 2 + int(round(start_offset / horizontal_scale))
    half_width = max(1, int(round(0.5 * stair_width / horizontal_scale)))
    centre_y = terrain.length // 2
    start_y = max(0, centre_y - half_width)
    stop_y = min(terrain.length, centre_y + half_width)

    for step_index in range(int(num_steps)):
        tread_start = start_x + step_index * tread_pixels
        tread_stop = min(terrain.width, tread_start + tread_pixels)
        if tread_start >= terrain.width:
            break
        height_units = int(round(
            (step_index + 1) * float(step_height) / vertical_scale))
        terrain.height_field_raw[
            tread_start:tread_stop, start_y:stop_y] = height_units

    top_start = start_x + int(num_steps) * tread_pixels
    top_stop = min(
        terrain.width,
        top_start + max(1, int(round(top_length / horizontal_scale))))
    if top_start < terrain.width:
        top_height_units = int(round(
            int(num_steps) * float(step_height) / vertical_scale))
        terrain.height_field_raw[
            top_start:top_stop, start_y:stop_y] = top_height_units
    return terrain


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) + len(self.eval_rows)
        self.tot_cols = max(len(self.train_cols), len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)

    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        if self.eval_cfg is None:
            return self.cfg.row_indices, self.cfg.col_indices, [], []
        else:
            self._load_cfg(self.eval_cfg)
            self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
            self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
            self.eval_cfg.x_offset = self.cfg.tot_rows
            self.eval_cfg.rows_offset = self.cfg.num_rows
            return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if cfg.curriculum:
            print("Creating curriculum terrains")
            self.curriculum(cfg)
        elif cfg.selected:
            print("Creating selected terrains")
            self.selected_terrain(cfg)
        else:
            print("Creating randomized terrains")
            self.randomized_terrain(cfg)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(
                    cfg, choice, difficulty, cfg.proportions, terrain_level=i)
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.length_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs['terrain_kwargs'])
            self.add_terrain_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions,
                     terrain_level=None):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.length_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        slope = difficulty * 0.4
        # step_height = 0.05 + 0.18 * difficulty
        step_height = 0.05 + difficulty*(cfg.max_step_height-0.05)
        configured_step_heights = getattr(
            cfg, 'sim2sim_stair_step_heights', None)
        if configured_step_heights is not None and terrain_level is not None:
            step_height = float(configured_step_heights[
                min(int(terrain_level), len(configured_step_heights) - 1)])
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        # print("The terrain proportions are: ", proportions)
        # print("The current choice is: ", choice)
              
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                slope *= -1
            # print("Adding smooth slope terrain")
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            # print("Adding rough slope terrain")
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            stairs_up = choice < proportions[2]
            # print("Adding stairs terrain")
            if (getattr(cfg, 'sim2sim_straight_stairs', False)
                    and stairs_up):
                straight_stairs_terrain(
                    terrain,
                    step_width=float(getattr(
                        cfg, 'sim2sim_stair_tread_depth', 0.30)),
                    step_height=step_height,
                    start_offset=float(getattr(
                        cfg, 'sim2sim_stair_start_offset', 0.55)),
                    stair_width=float(getattr(
                        cfg, 'sim2sim_stair_width', 2.0)),
                    num_steps=int(getattr(
                        cfg, 'sim2sim_stair_num_steps', 8)),
                    top_length=float(getattr(
                        cfg, 'sim2sim_stair_top_length', 0.90)),
                )
            else:
                if stairs_up:
                    step_height *= -1
                terrain_utils.pyramid_stairs_terrain(
                    terrain, step_width=0.31, step_height=step_height,
                    platform_size=3.)
        elif choice < proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            # print("Adding discrete obstacles terrain")
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            # print("Adding stepping stones terrain")
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            pass
        elif choice < proportions[7]:
            # print("Adding smooth flat terrain")
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.0, # flat terrain
                                                 max_height=0.0, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[8]:
            # print("Adding rough flat terrain")
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.width_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.width_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.length_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.length_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_width + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_length
        x1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale)
        if getattr(cfg, 'sim2sim_straight_stairs', False):
            # The matched staircase has a flat approach at the map centre.
            # Sampling that exact patch keeps the base on the ground; the old
            # 2 m max window included the first risers and spawned it in air.
            centre_x = terrain.width // 2
            centre_y = terrain.length // 2
            footprint = max(1, int(round(0.20 / terrain.horizontal_scale)))
            env_origin_z = np.max(terrain.height_field_raw[
                centre_x - footprint:centre_x + footprint + 1,
                centre_y - footprint:centre_y + footprint + 1,
            ]) * terrain.vertical_scale
        else:
            env_origin_z = np.max(
                terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
