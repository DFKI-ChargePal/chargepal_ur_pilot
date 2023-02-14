"""" This file contains functionalities to move the robot randomly. """
from __future__ import annotations

# global
import numpy as np
from dataclasses import dataclass

# local
from ur_pilot.core.robot import Robot


@dataclass
class Result:
     rnd_joint_pos: list[float]


def move_joints_random(rob: Robot) -> Result:
     # Move to home joint position
    rob.move_home()
    # Move to random joint positions near to the home configuration
    home_q = np.array(rob.home_joint_config, dtype=np.float32)
    tgt_joint_q = (home_q + (np.random.rand(6) * 2.0 - 1.0) * 0.075).tolist()
    rob.move_j(tgt_joint_q)
    # Move back to home joint positions
    rob.move_home()
    return Result(tgt_joint_q)
