from __future__ import annotations

# global
import logging
import numpy as np
from pathlib import Path

# local
import time
import math
import config
from ur_pilot.utils import set_logging_level
from ur_pilot.bota_sensor import BotaFtSensor
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.monitor.signal_monitor import SignalMonitor


DURATION_ = math.inf
SUB_STEPS_ = 40
PRINT_FT_ = True
PRINT_ACC_ = True


def main() -> None:

    config_fp = Path(config.__file__).parent.joinpath(config.RUNNING_CONFIG_FILE)
    config_dict = read_toml(config_fp)
    cfg = Config(**config_dict)
    sensor_cfg = cfg.robot.ft_sensor.dict()

    ax_labels = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Tx [Nm]', 'Ty [Nm]', 'Tz [Nm]']

    with BotaFtSensor(**sensor_cfg) as sensor:

        sig_mtr = SignalMonitor(ax_labels, round(1/sensor.time_step), 10.0)

        t_start_ = time.time()
        t_next_ = t_start_
        while t_start_ + DURATION_ > time.time():
            
            signal = np.reshape(sensor.FT, [6, 1])
            for i in range(SUB_STEPS_ - 1):
                t_left_ = t_next_ - time.time()
                if t_left_ > 0.0:
                    time.sleep(sensor.time_step)
                t_next_ += sensor.time_step
                signal = np.hstack([signal, np.reshape(sensor.IMU, [6, 1])])
            sig_mtr.add(signal)
            

if __name__ == '__main__':
    print("Bota force-torque sensor example!")
    set_logging_level(logging.INFO)
    main()
