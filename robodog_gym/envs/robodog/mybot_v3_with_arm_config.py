from typing import Union

from params_proto import Meta

from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.mybot_v3_config import config_mybot_v3


def config_mybot_v3_with_arm(Cnfg: Union[Cfg, Meta]):
    config_mybot_v3(Cnfg)

    _ = Cnfg.asset
    _.file = '{MINI_GYM_ROOT_DIR}/resources/robots/mybot_v3_with_arm/urdf/mybot_v3_with_arm.urdf'
