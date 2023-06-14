"""" This file contains functionalities to move the robot randomly. """
# global
import numpy as np

# local
from ur_pilot.core import URPilot
from ur_pilot.msgs.result_msgs import JointPosResult


def move_joints_random(rob: URPilot) -> JointPosResult:
     # Move to home joint position
    rob.move_home()
    # Move to random joint positions near to the home configuration
    home_q = np.array(rob.home_joint_config, dtype=np.float32)
    tgt_joint_q = (home_q + (np.random.rand(6) * 2.0 - 1.0) * 0.075).tolist()
    rob.move_j(tgt_joint_q)
    # Move back to home joint positions
    rob.move_home()
    return JointPosResult(tgt_joint_q)
