""" This file contains function to move the robot to specific goal configurations. """

# local
from ur_pilot.core import URPilot
from ur_pilot.msgs.result_msgs import JointPosResult, TCPPoseResult
from ur_pilot.msgs.request_msgs import JointPosRequest, TCPPoseRequest


def move_to_joint_pos(rob: URPilot, req: JointPosRequest) -> JointPosResult:
    # Move to requested joint position
    rob.move_j(q=req.joint_pos)
    new_joint_pos = rob.get_joint_pos()
    return JointPosResult(new_joint_pos)


def move_to_tcp_pose(rob: URPilot, req: TCPPoseRequest) -> TCPPoseResult:
    # Move to requested TCP pose
    req_pose = req.tcp_pose.xyz + req.tcp_pose.axis_angle
    rob.move_l(req.tcp_pose)
    res_pose = rob.get_tcp_pose()
    return TCPPoseResult(res_pose)
