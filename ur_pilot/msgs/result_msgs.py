""" File to define result message classes. """
from __future__ import annotations

# global
from rigmopy import Pose
from dataclasses import dataclass

# typing
from typing import Sequence


@dataclass
class JointPosResult:
    joint_pos: Sequence[float]


@dataclass
class TCPPoseResult:
    tcp_pose: Pose


@dataclass
class PlugInResult:
    tcp_pose: Pose
    time_out: bool


@dataclass
class PlugOutResult:
    tcp_pose: Pose
    time_out: bool
