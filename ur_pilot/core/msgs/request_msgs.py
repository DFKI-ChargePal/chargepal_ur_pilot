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
