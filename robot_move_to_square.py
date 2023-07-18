"""
Script to move the robot to different positions.
"""
# global
import numpy as np
from rigmopy import Pose

# local
from ur_pilot import URPilot
from ur_pilot.move_to import move_to_pose
from ur_pilot.msgs.request_msgs import MoveToPoseRequest


def move_to_square(ctrl_type: str, length: float) -> None:
    # Connect to robot arm
    ur10 = URPilot()
    # Move home
    ur10.move_home()
    ref_pose = ur10.get_tcp_pose()
    ref_p, ref_q = ref_pose.p, ref_pose.q
    # Move 10 cm in y direction
    new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, 0.0)), ref_q.wxyz)
    res = move_to_pose(ur10, MoveToPoseRequest(new_pose, ctrl_type, 15.0))
    if res.time_out:
        print(f"Warning: Target was not reached")
    # Move 10cm in z direction
    new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, length)), ref_q.wxyz)
    res = move_to_pose(ur10, MoveToPoseRequest(new_pose, ctrl_type, 15.0))
    if res.time_out:
        print(f"Warning: Target was not reached")
    # Move 10cm back in y direction
    new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, length)), ref_q.wxyz)
    res = move_to_pose(ur10, MoveToPoseRequest(new_pose, ctrl_type, 15.0))
    if res.time_out:
        print(f"Warning: Target was not reached")
    # Move 10cm back in z direction
    new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, 0.0)), ref_q.wxyz)
    res = move_to_pose(ur10, MoveToPoseRequest(new_pose, ctrl_type, 15.0))
    if res.time_out:
        print(f"Warning: Target was not reached")
    print(f"Error wrt. start position: {ref_p - ur10.get_tcp_pose().p}")

    # Disconnect to robot arm
    if ctrl_type == 'motion':
        ur10.stop_motion_mode()
    ur10.exit()


if __name__ == '__main__':
    move_to_square('motion', 0.25)
