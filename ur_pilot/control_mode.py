from __future__ import annotations

import abc
from enum import auto, Enum
from contextlib import contextmanager

from ur_pilot.robot import Robot

from typing import Iterator

from ur_pilot.config_mdl import Config


class ModeTypes(Enum):
    DISABLED = auto()
    FORCE = auto()
    SERVO = auto()
    MOTION = auto()
    HYBRID = auto()
    POSITION = auto()
    VELOCITY = auto()
    TEACH_IN = auto()


class ControlMode(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        """ Set up control mode """
        raise NotImplementedError("Must be implemented in subclass.")

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        """ Exit control mode """
        return ModeTypes.DISABLED


class PositionMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        return ModeTypes.POSITION


class VelocityMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        return ModeTypes.VELOCITY


class ServoMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        return ModeTypes.SERVO

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        robot.servo_stop()
        return ControlMode.exit(robot, cfg)


class ForceMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        robot.set_up_force_mode(cfg.pilot.force_mode.gain)
        return ModeTypes.FORCE

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        robot.stop_force_mode()
        return ControlMode.exit(robot, cfg)


class TeachMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        robot.set_up_teach_mode()
        return ModeTypes.TEACH_IN

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        robot.stop_teach_mode()
        return ControlMode.exit(robot, cfg)


class HybridMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        robot.set_up_hybrid_mode(
            error_scale=cfg.pilot.hybrid_mode.error_scale,
            force_limit=cfg.pilot.hybrid_mode.force_limit,
            Kp_6_force=cfg.pilot.hybrid_mode.Kp_force,
            Kd_6_force=cfg.pilot.hybrid_mode.Kd_force,
            Kp_6_motion=cfg.pilot.hybrid_mode.Kp_motion,
            Kd_6_motion=cfg.pilot.hybrid_mode.Kd_motion,
            ft_gain=cfg.pilot.force_mode.gain,
            ft_damping=cfg.pilot.force_mode.damping
        )
        return ModeTypes.HYBRID

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        robot.stop_hybrid_mode()
        return ControlMode.exit(robot, cfg)


class MotionMode(ControlMode):

    @staticmethod
    def enter(robot: Robot, cfg: Config) -> ModeTypes:
        robot.set_up_motion_mode(
            error_scale=cfg.pilot.motion_mode.error_scale,
            force_limit=cfg.pilot.motion_mode.force_limit,
            Kp_6=cfg.pilot.motion_mode.Kp,
            Kd_6=cfg.pilot.motion_mode.Kd,
            ft_gain=cfg.pilot.force_mode.gain,
            ft_damping=cfg.pilot.force_mode.damping
        )
        return ModeTypes.MOTION

    @staticmethod
    def exit(robot: Robot, cfg: Config) -> ModeTypes:
        robot.stop_motion_mode()
        return ControlMode.exit(robot, cfg)


class ControlContextManager:

    mode_types = ModeTypes

    def __init__(self, robot: Robot, cfg: Config):
        self.cfg = cfg
        self.robot = robot
        self.mode = self.mode_types.DISABLED

    def check_mode(self, expected: ModeTypes | list[ModeTypes]) -> None:
        if self.mode is ModeTypes.DISABLED:
            raise RuntimeError(f"Manager is disabled and not in any control mode. Running actions is not possible.")
        if isinstance(expected, list) and self.mode not in expected:
            raise RuntimeError(f"This action is not able to use one of the control modes '{expected}'")
        if isinstance(expected, ModeTypes) and self.mode is not expected:
            raise RuntimeError(f"This action is not able to use the control mode '{expected}'")

    def exit(self) -> None:
        if self.mode == ModeTypes.POSITION:
            self.mode = PositionMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.SERVO:
            self.mode = ServoMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.FORCE:
            self.mode = ForceMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.MOTION:
            self.mode = MotionMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.HYBRID:
            self.mode = HybridMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.VELOCITY:
            self.mode = VelocityMode.exit(self.robot, self.cfg)
        elif self.mode == ModeTypes.TEACH_IN:
            self.mode = TeachMode.exit(self.robot, self.cfg)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    @contextmanager
    def position_control(self) -> Iterator[None]:
        self.mode = PositionMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def velocity_control(self) -> Iterator[None]:
        self.mode = VelocityMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def servo_control(self) -> Iterator[None]:
        self.mode = ServoMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def force_control(self) -> Iterator[None]:
        self.mode = ForceMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def teach_in_control(self) -> Iterator[None]:
        self.mode = TeachMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def hybrid_control(self) -> Iterator[None]:
        self.mode = HybridMode.enter(self.robot, self.cfg)
        yield
        self.exit()

    @contextmanager
    def motion_control(self) -> Iterator[None]:
        self.mode = MotionMode.enter(self.robot, self.cfg)
        yield
        self.exit()
