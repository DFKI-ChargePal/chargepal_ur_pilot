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
    tcp_pose: Pose


@dataclass
class PlugInRequest:
     compliant_axes: list[int]
     wrench: Vector6d
     t_limit: float


@dataclass
class PlugOutRequest:
     compliant_axes: list[int]
     wrench: Vector6d
     moving_distance: float
     t_limit: float