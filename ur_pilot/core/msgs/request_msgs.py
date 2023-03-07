""" File to define request message classes. """
from __future__ import annotations

# global
import rigmopy as rp
from dataclasses import dataclass


@dataclass
class JointPosRequest:
     joint_pos: list[float]


@dataclass
class TCPPoseRequest:
    tcp_pose: rp.Pose


@dataclass
class PlugInRequest:
     compliant_axes: list[int]
     wrench: rp.Wrench
     t_limit: float


@dataclass
class PlugOutRequest:
     compliant_axes: list[int]
     wrench: rp.Wrench
     moving_distance: float
     t_limit: float