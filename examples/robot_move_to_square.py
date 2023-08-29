"""
Script to move the robot to different positions.
"""
# global
import numpy as np
from rigmopy import Pose

# local
import ur_pilot


def move_to_square(ctrl_type: str, length: float) -> None:
    # Connect to pilot/robot

    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.position_control():
            pilot.move_home()

        # Use motion mode to moving to the corners
        with pilot.motion_control():
            ref_pose = pilot.robot.get_tcp_pose()
            ref_p, ref_q = ref_pose.p, ref_pose.q
            # Move 10 cm in y direction
            new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, 0.0)), ref_q.wxyz)
            success, _ = pilot.move_to_tcp_pose(new_pose, time_out=5.0)
            if not success:
                print(f"Warning: Target was not reached")
            # Move 10cm in z direction
            new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, length)), ref_q.wxyz)
            success, _ = pilot.move_to_tcp_pose(new_pose, time_out=5.0)
            if not success:
                print(f"Warning: Target was not reached")
            # Move 10cm back in y direction
            new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, length)), ref_q.wxyz)
            success, _ = pilot.move_to_tcp_pose(new_pose, time_out=5.0)
            if not success:
                print(f"Warning: Target was not reached")
            # Move 10cm back in z direction
            new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, 0.0)), ref_q.wxyz)
            success, _ = pilot.move_to_tcp_pose(new_pose, time_out=5.0)
            if not success:
                print(f"Warning: Target was not reached")
            print(f"Error wrt. start position: {ref_p - pilot.robot.get_tcp_pose().p}")


if __name__ == '__main__':
    move_to_square('motion', 0.10)
