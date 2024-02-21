"""
Script to move the robot to different positions.
"""
# libs
import logging
import ur_pilot
import spatialmath as sm


LOGGER = logging.getLogger(__name__)


def move_to() -> None:
    ur_pilot.logger.set_logging_level(logging.INFO)
    # Connect to pilot/robot
    with ur_pilot.connect() as pilot:
        # Move home
        # with pilot.context.position_control():
        with pilot.context.motion_control():
            pilot.robot.move_home()

            LOGGER.info(f"Start pose: {ur_pilot.utils.se3_to_str(pilot.robot.tcp_pose)}")
            # Move in z direction wrt tcp
            T_base2tcp = pilot.robot.tcp_pose
            T_tcp2target = sm.SE3.Tz(0.05)
            T_base2target = T_base2tcp * T_tcp2target
            pilot.move_to_tcp_pose(T_base2target)
            # Move in x direction wrt tcp
            T_base2tcp = pilot.robot.tcp_pose
            T_tcp2target = sm.SE3.Tx(-0.05)
            T_base2target = T_base2tcp * T_tcp2target
            pilot.move_to_tcp_pose(T_base2target)
            # Move in xz direction wrt tcp
            T_base2tcp = pilot.robot.tcp_pose
            T_tcp2target = sm.SE3.Trans(x=0.05, y=0.0, z=-0.05)
            T_base2target = T_base2tcp * T_tcp2target
            pilot.move_to_tcp_pose(T_base2target)
            # Print result
            LOGGER.info(f"  End pose: {ur_pilot.utils.se3_to_str(pilot.robot.tcp_pose)}")


if __name__ == '__main__':
    move_to()
