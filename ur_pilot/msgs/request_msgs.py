""" File to define request message classes. """
from __future__ import annotations

# global
from dataclasses import dataclass
from rigmopy import Pose, Vector6d


@dataclass
class JointPosRequest:
     joint_pos: list[float]


@dataclass
class TCPPoseRequest:
    tcp_target: Pose


@dataclass
class MoveToPoseRequest:
     tcp_target: Pose
     controller_type: str
     t_limit: float = -1.0


@dataclass
class PlugInForceModeRequest:
     compliant_axes: list[int]
     wrench: Vector6d
     t_limit: float


@dataclass
class PlugInMotionModeRequest:
     tcp_target: Pose
     t_limit: float
     error_scale: float | None
     Kp: list[float] | None
     Kd: list[float] | None


@dataclass
class PlugOutRequest:
     compliant_axes: list[int]
     wrench: Vector6d
     moving_distance: float
     t_limit: float