from __future__ import annotations

# global
import sys
import logging
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# local
import time
import config
from ur_pilot.utils import set_logging_level
from ur_pilot.bota_sensor import BotaFtSensor
from ur_pilot.config_mdl import Config, read_toml

# typing
from numpy import typing as npt


DURATION_ = 3.0
PRINT_FT_ = True
PRINT_ACC_ = True


def main() -> None:

    config_fp = Path(config.__file__).parent.joinpath(config.RUNNING_CONFIG_FILE)
    config_dict = read_toml(config_fp)
    cfg = Config(**config_dict)
    sensor_cfg = cfg.robot.ft_sensor.dict()

    with BotaFtSensor(**sensor_cfg) as sensor:

        sensor.clear_ft_offset()

        t_start_ = time.time()
        while t_start_ + DURATION_ > time.time():
        
            if PRINT_FT_:
                print(f"Fx: {sensor.Fx:11.5f}")
                print(f"Fy: {sensor.Fy:11.5f}")
                print(f"Fz: {sensor.Fz:11.5f}")
                print(f"Tx: {sensor.Tx:11.5f}")
                print(f"Ty: {sensor.Ty:11.5f}")
                print(f"Tz: {sensor.Tz:11.5f}\n")

            if PRINT_ACC_:
                print(f"Ax: {sensor.Ax:11.5f}")
                print(f"Ay: {sensor.Ay:11.5f}")
                print(f"Az: {sensor.Az:11.5f}")
                print(f"Rx: {sensor.Rx:11.5f}")
                print(f"Ry: {sensor.Ry:11.5f}")
                print(f"Rz: {sensor.Rz:11.5f}\n")
    
            print(" ")
            time.sleep(sensor.time_step)



if __name__ == '__main__':
    print("Bota force-torque sensor example!")
    set_logging_level(logging.INFO)
    main()
