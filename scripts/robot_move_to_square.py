"""
Script to move the robot to different positions.
"""
# global
import logging
import numpy as np
from rigmopy import Pose

# local
import ur_pilot

LOGGER = logging.getLogger(__name__)


def move_to_square(ctrl_type: str, length: float) -> None:
    ur_pilot.logger.set_logging_level(logging.INFO)
    # Connect to pilot/robot
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.context.position_control():
            pilot.robot.move_home()

        with pilot.context.motion_control():
            ref_pose = pilot.robot.tcp_pose
            ref_pos, ref_rot = ref_pose.t, ref_pose.R

            # T_tcp2target = sm.SE3.Tz(0.05)
            # T_base2target = T_base2tcp * T_tcp2target
            # pilot.move_to_tcp_pose(T_base2target)


        # # Use motion mode to moving to the corners
        # with pilot.context.motion_control():
        #     ref_pose = pilot.robot.tcp_pose
        #     ref_p, ref_q = ref_pose.p, ref_pose.q
        #     # Move 10 cm in y direction
        #     new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, 0.0)), ref_q.wxyz)
        #     success, _ = pilot.move_to_tcp_pose(new_pose, time_out=15.0)
        #     if not success:
        #         LOGGER.warning(f"Target was not reached")
        #     # Move 10cm in z direction
        #     new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, -length, length)), ref_q.wxyz)
        #     success, _ = pilot.move_to_tcp_pose(new_pose, time_out=15.0)
        #     if not success:
        #         LOGGER.warning(f"Warning: Target was not reached")
        #     # Move 10cm back in y direction
        #     new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, length)), ref_q.wxyz)
        #     success, _ = pilot.move_to_tcp_pose(new_pose, time_out=15.0)
        #     if not success:
        #         LOGGER.warning(f"Warning: Target was not reached")
        #     # Move 10cm back in z direction
        #     new_pose = Pose().from_xyz_wxyz(np.array(ref_p.xyz) + np.array((0.0, 0.0, 0.0)), ref_q.wxyz)
        #     success, _ = pilot.move_to_tcp_pose(new_pose, time_out=15.0)
        #     if not success:
        #         LOGGER.warning(f"Warning: Target was not reached")
        #     LOGGER.info(f"Final error wrt. start position: {ref_p - pilot.robot.get_tcp_pose().p}")


if __name__ == '__main__':
    move_to_square('motion', 0.10)
