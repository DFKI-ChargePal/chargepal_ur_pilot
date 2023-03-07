""" File to define result message classes. """
from __future__ import annotations

# global
import rigmopy as rp
from dataclasses import dataclass

# typing
from typing import Sequence


@dataclass
class JointPosResult:
    joint_pos: Sequence[float]


@dataclass
class TCPPoseResult:
    tcp_pose: rp.Pose


@dataclass
class PlugOutResult:
    tcp_pose: rp.Pose
    time_out: bool
