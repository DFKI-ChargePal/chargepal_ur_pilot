from __future__ import annotations

# libs
import logging
import numpy as np
import spatialmath as sm
from enum import Enum, auto
from spatialmath.base import q2r
from contextlib import contextmanager
from ur_pilot.config_mdl import Config

# typing
from numpy import typing as npt
from typing import Iterator, Sequence, Any

LOGGER = logging.getLogger(__name__)


class ToolModel:

    def __init__(self, mass: float, com: npt.ArrayLike, gravity: npt.ArrayLike = (0.0, 0.0, -9.80665)):
        self.mass = mass
        self.com = np.reshape(com, 3)
        self.gravity = np.reshape(gravity, 3)
        self.f_inertia = self.mass * self.gravity


class TwistCouplingModel(ToolModel):

    def __init__(self,
                 mass: float,
                 com: npt.ArrayLike,
                 gravity: npt.ArrayLike = (0.0, 0.0, -9.80665),
                 link_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 safety_margin: npt.ArrayLike = (0.0, 0.0, 0.04, 0.0, 0.0, 0.0)
                 ):
        super().__init__(mass, com, gravity)
        l_frame = np.reshape(link_frame, 6)
        self.T_mounting2locked = sm.SE3.Rt(R=sm.SO3.EulerVec(l_frame[3:6]), t=l_frame[0:3])
        T_locked2unlocked = sm.SE3.EulerVec((0.0, 0.0, -np.pi/2))
        self.T_mounting2unlocked = self.T_mounting2locked * T_locked2unlocked
        s_margin = np.reshape(safety_margin, 6)
        T_unlocked2safety = sm.SE3.Rt(R=sm.SO3.EulerVec(s_margin[3:6]), t=s_margin[0:3])
        self.T_mounting2safety = self.T_mounting2unlocked * T_unlocked2safety


class PlugToolModel(ToolModel):

    def __init__(self,
                 mass: float,
                 com: npt.ArrayLike,
                 gravity: npt.ArrayLike = (0.0, 0.0, -9.80665),
                 lip_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 tip_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 sense_frame: npt.ArrayLike = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 safety_margin: npt.ArrayLike = (0.0, 0.0, 0.02, 0.0, 0.0, 0.0)
                 ) -> None:
        super().__init__(mass, com, gravity)
        l_frame = np.reshape(lip_frame, 6)
        self.T_mounting2lip = sm.SE3.Rt(R=sm.SO3.EulerVec(l_frame[3:6]), t=l_frame[0:3])
        t_frame = np.reshape(tip_frame, 6)
        self.T_mounting2tip = sm.SE3.Rt(R=sm.SO3.EulerVec(t_frame[3:6]), t=t_frame[0:3])
        s_frame = np.reshape(sense_frame, 6)
        self.T_mounting2sense = sm.SE3.Rt(R=sm.SO3.EulerVec(s_frame[3:6]), t=s_frame[0:3])
        s_margin = np.reshape(safety_margin, 6)
        T_tip2safety = sm.SE3.Rt(R=sm.SO3.EulerVec(s_margin[3:6]), t=s_margin[0:3])
        self.T_mounting2safety = self.T_mounting2tip * T_tip2safety


class PlugTypes(Enum):

    NONE = auto()
    TYPE2_FEMALE = auto()
    TYPE2_MALE = auto()
    CCS_FEMALE = auto()


class PlugModel:

    plug_types = PlugTypes

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.type = self.plug_types.NONE
        self.models: dict[PlugTypes, PlugToolModel] = {
            self.plug_types.TYPE2_FEMALE: PlugToolModel(**self.cfg.pilot.tm_type2_female.dict()),
            self.plug_types.TYPE2_MALE: PlugToolModel(**self.cfg.pilot.tm_type2_male.dict()),
            self.plug_types.CCS_FEMALE: PlugToolModel(**self.cfg.pilot.tm_ccs_female.dict()),
        }

    @property
    def mass(self) -> float:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            mass = self.models[self.type].mass
        return mass

    @property
    def com(self) -> npt.NDArray[np.float_]:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            com = self.models[self.type].com
        return com

    @property
    def gravity(self) -> npt.NDArray[np.float_]:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            gravity = self.models[self.type].gravity
        return gravity

    @property
    def f_inertia(self) -> npt.NDArray[np.float_]:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            f_inertia = self.models[self.type].f_inertia
        return f_inertia

    @property
    def T_mounting2lip(self) -> sm.SE3:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            T_mounting2lip = self.models[self.type].T_mounting2lip
        return T_mounting2lip

    @property
    def T_mounting2tip(self) -> sm.SE3:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            T_mounting2tip = self.models[self.type].T_mounting2tip
        return T_mounting2tip

    @property
    def T_mounting2sense(self) -> sm.SE3:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context is inactive.")
        else:
            T_mounting2sense = self.models[self.type].T_mounting2sense
        return T_mounting2sense

    @property
    def T_mounting2safety(self) -> sm.SE3:
        if self.type == self.plug_types.NONE:
            raise RuntimeError(f"Trying to access plug value while the context  is inactive.")
        else:
            T_mounting2safety = self.models[self.type].T_mounting2safety
        return T_mounting2safety

    def exit(self) -> None:
        self.type = self.plug_types.NONE

    @contextmanager
    def context(self, plug_type: str) -> Iterator[None]:
        name = plug_type.lower()
        try:
            if name == 'type2_male':
                LOGGER.debug(f"Enter workspace context using plug type: Type 2 male")
                yield
                LOGGER.debug(f"Exit workspace context using plug type: Type 2 male")
            elif name == 'type2_female':
                self.type = self.plug_types.TYPE2_FEMALE
                LOGGER.debug(f"Enter workspace context using plug type: Type 2 female")
                yield
                LOGGER.debug(f"Exit workspace context using plug type: Type 2 female")
            elif name == 'ccs_female':
                LOGGER.debug(f"Enter workspace context using plug type: CCS female")
                yield
                LOGGER.debug(f"Exit workspace context using plug type: CCS female")
            else:
                raise KeyError(f"No context with name '{name}' available")
        finally:
            self.exit()

    @contextmanager
    def type2_female(self) -> Iterator[None]:
        self.type = self.plug_types.TYPE2_FEMALE
        try:
            LOGGER.debug(f"Enter workspace context using plug type: Type 2 female")
            yield
        finally:
            self.exit()
            LOGGER.debug(f"Exit workspace context using plug type: Type 2 female")

    @contextmanager
    def type2_male(self) -> Iterator[None]:
        self.type = self.plug_types.TYPE2_MALE
        try:
            LOGGER.debug(f"Enter workspace context using plug type: Type 2 male")
            yield
        finally:
            self.exit()
            LOGGER.debug(f"Exit workspace context using plug type: Type 2 male")

    @contextmanager
    def ccs_female(self) -> Iterator[None]:
        self.type = self.plug_types.CCS_FEMALE
        try:
            LOGGER.debug(f"Enter workspace context using plug type: CCS female")
            yield
        finally:
            self.exit()
            LOGGER.debug(f"Exit workspace context using plug type: CCS female")


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
