""" This file contains function to move the robot to specific goal configurations. """
# global
import time
import numpy as np

# local
from ur_pilot.core import URPilot
from ur_pilot.msgs.result_msgs import JointPosResult, TCPPoseResult, MoveToPoseResult
from ur_pilot.msgs.request_msgs import JointPosRequest, TCPPoseRequest, MoveToPoseRequest


def move_to_joint_pos(rob: URPilot, req: JointPosRequest) -> JointPosResult:
    # Move to requested joint position
    rob.move_j(q=req.joint_pos)
    new_joint_pos = rob.get_joint_pos()
    return JointPosResult(new_joint_pos)


def move_to_tcp_pose(rob: URPilot, req: TCPPoseRequest) -> TCPPoseResult:
    # Move to requested TCP pose
    rob.move_l(req.tcp_target)
    res_pose = rob.get_tcp_pose()
    return TCPPoseResult(res_pose)


def move_to_pose(rob: URPilot, req: MoveToPoseRequest) -> MoveToPoseResult:
    # Check mode
    ctrl_type = req.controller_type.lower()
    if ctrl_type == 'position':
        rob.move_l(req.tcp_target)
        res = MoveToPoseResult(rob.get_tcp_pose())
    elif ctrl_type == 'velocity':
        raise NotImplementedError(f"Moving robot with control type'{ctrl_type}' is not implemented yet.")
    elif ctrl_type == 'motion':
        rob.set_up_motion_mode(
            error_scale=5000.0)
        # rob.set_up_motion_mode()
        time_out = False
        t_start = time.time()
        tgt_3pts = req.tcp_target.to_3pt_set()
        while True:
            rob.motion_mode(req.tcp_target)
            cur_3pts = rob.get_tcp_pose().to_3pt_set()
            error = np.mean(np.abs(tgt_3pts - cur_3pts))
            if error <= 0.005:
                break
            elif time.time() - t_start > req.t_limit:
                time_out = True
                break
        res = MoveToPoseResult(rob.get_tcp_pose(), time_out)
    else:
        raise ValueError(f"Unknown controller type: '{ctrl_type}'")
    return res
