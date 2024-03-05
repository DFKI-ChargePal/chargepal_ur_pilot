from __future__ import annotations

import numpy as np
# global
import spatialmath as sm
from spatialmath.base import q2r

# typing
from typing import Sequence
from numpy import typing as npt


class ToolModel:

    def __init__(self,
                 mass: float,
                 com: npt.ArrayLike,
                 gravity: npt.ArrayLike = (0.0, 0.0, -9.80665),
                 tip_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 sense_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                 ) -> None:
        self.mass = mass
        self.com = np.reshape(com, 3)
        self.gravity = np.reshape(gravity, 3)
        self.f_inertia = self.mass * self.gravity
        t_frame = np.reshape(tip_frame, 6)
        self.T_mounting2tip = sm.SE3.Rt(R=sm.SO3.EulerVec(t_frame[3:6]), t=t_frame[0:3])
        s_frame = np.reshape(sense_frame, 6)
        self.T_mounting2sense = sm.SE3.Rt(R=sm.SO3.EulerVec(s_frame[3:6]), t=s_frame[0:3])


class CameraModel:

    def __init__(self) -> None:
        self.T_flange2camera = sm.SE3()


class BotaSensONEModel:
    """ Class to get sensor parameters. Based on description file:
        https://gitlab.com/botasys/bota_driver/-/blob/master/rokubimini_description/urdf/BFT_SENS_ECAT_M8.urdf.xacro
    """
    def __init__(self,
                 sensor_wrench_mass: float = 0.081117,
                 sensor_mounting_mass: float = 0.14931,
                 sensor_imu_mass: float = 0.00841533584597687,
                 p_mounting2wrench: Sequence[float] = (0.0, 0.0, 0.035),
                 q_mounting2wrench: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
                 ) -> None:
        self.sensor_wrench_mass = sensor_wrench_mass
        self.sensor_mounting_mass = sensor_mounting_mass
        self.sensor_imu_mass = sensor_imu_mass
        self.sensor_mass = self.sensor_mounting_mass + self.sensor_wrench_mass + self.sensor_imu_mass
        self.T_mounting2wrench = sm.SE3.Rt(
            R=q2r(np.array(q_mounting2wrench), order='sxyz'),
            t=np.array(p_mounting2wrench)
        )
