""" This file contains function to move the robot to specific goal configurations. """

# local
from ur_pilot.core.robot import Robot
from ur_pilot.core.msgs.result_msgs import JointPosResult, TCPPoseResult
from ur_pilot.core.msgs.request_msgs import JointPosRequest, TCPPoseRequest


def move_to_joint_pos(rob: Robot, req: JointPosRequest) -> JointPosResult:
    # Move to requested joint position
    rob.move_j(q=req.joint_pos)
    new_joint_pos = rob.get_joint_pos()
    return JointPosResult(new_joint_pos)


def move_to_tcp_pose(rob: Robot, req: TCPPoseRequest) -> TCPPoseResult:
    # Move to requested TCP pose
    req_pose = req.tcp_pose.pos.xyz + req.tcp_pose.ori.axis_angle
    rob.move_l(req.tcp_pose)
    res_pose = rob.get_tcp_pose()
    return TCPPoseResult(res_pose)
