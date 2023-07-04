from __future__ import annotations

# global
import time
import numpy as np

# local
from ur_pilot.config_mdl import FTSensor
from ur_pilot.end_effector.bota_sensor import BotaFtSensor

# typing 
from numpy import typing as npt


def record_from_sensor(sensor_cfg: FTSensor, duration: float, rec_type: str) -> npt.NDArray[np.float64]:

    rec_imu, rec_ft = False, False
    if rec_type.capitalize() == 'FT':
        rec_ft = True
    elif rec_type.capitalize() == 'IMU':
        rec_imu = True
    else:
        rec_ft = True
        rec_imu = True

    data_rec: npt.NDArray[np.float64] | None = None

    with BotaFtSensor(**sensor_cfg.dict()) as sensor:

        t_start = time.time()
        while t_start + duration > time.time():

            ft_readings = np.reshape(sensor.FT_raw, [6, 1]) if rec_ft else np.array([], dtype=np.float64)
            imu_readings = np.reshape(sensor.IMU, [6, 1]) if rec_imu else np.array([], dtype=np.float64)
            sensor_readings = np.vstack([ft_readings, imu_readings])
            if data_rec is None:
                data_rec = sensor_readings
            else:
                data_rec = np.hstack([data_rec, sensor_readings])
            time.sleep(sensor.time_step)

    if data_rec is not None:
        return data_rec
    else:
        return np.array([], dtype=np.float64)
