from __future__ import annotations

# global
from rigmopy import Pose, Vector3d, Quaternion, Transformation

# typing
from typing import Sequence


class ToolModel:

    def __init__(self,
                 mass: float,
                 com: Sequence[float],
                 gravity: Sequence[float] = (0.0, 0.0, -9.80665),
                 p_mounting2tip: Sequence[float] = (0.0, 0.0, 0.0),
                 q_mounting2tip: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
                 ) -> None:
        self.mass = mass
        self.com = Vector3d().from_xyz(com)
        self.gravity = Vector3d().from_xyz(gravity)
        self.f_inertia = self.mass * self.gravity
        self.p_mounting2tip = Vector3d().from_xyz(p_mounting2tip)
        self.q_mounting2tip = Quaternion().from_wxyz(q_mounting2tip)
        self.T_mounting2tip = Transformation().from_pose(Pose().from_pq(self.p_mounting2tip, self.q_mounting2tip))


class CameraModel:

    def __init__(self) -> None:
        self.T_flange2camera = Transformation()


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
        self.p_mounting2wrench = Vector3d().from_xyz(p_mounting2wrench)
        self.q_mounting2wrench = Quaternion().from_wxyz(q_mounting2wrench)
        self.T_mounting2wrench = Transformation().from_pose(Pose().from_pq(self.p_mounting2wrench, self.q_mounting2wrench))
