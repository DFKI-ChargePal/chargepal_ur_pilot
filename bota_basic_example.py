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
from ur_pilot.ft_sensor.bota_sensor import BotaFtSensor
from ur_pilot.config_mdl import Config, read_toml
from ur_pilot.monitor.signal_monitor import SignalMonitor

LOGGER = logging.getLogger(__name__)

DURATION_ = math.inf
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

        sub_steps = int(sig_mtr.display_rate / sensor.time_step)
        LOGGER.info(f'Perform {sub_steps} sub-steps before updating the Monitor.')

        t_start_ = time.time()
        t_next_ = t_start_
        while t_start_ + DURATION_ > time.time():
            
            signal = np.reshape(sensor.FT, [6, 1])
            for i in range(sub_steps - 1):
                signal = np.hstack([signal, np.reshape(sensor.FT, [6, 1])])
                time.sleep(sensor.time_step)

            t_next_ += sig_mtr.display_rate
            t_left_ = t_next_ - time.time()
            if t_left_ > 0.0:
                print("Sleep zzz")
                time.sleep(t_left_)
            sig_mtr.add(signal)


if __name__ == '__main__':
    print("Bota force-torque sensor example!")
    set_logging_level(logging.INFO)
    main()
