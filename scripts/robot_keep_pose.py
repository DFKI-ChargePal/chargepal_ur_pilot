""" Short demo to demonstrate motion mode. Goal is to keep the current pose """
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


def keep_pose(soft: bool, time_out: float) -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        
        with pilot.context.position_control():
            pilot.robot.move_home()

        with pilot.context.motion_control():
            init_pose = pilot.robot.tcp_pose
            new_pose = init_pose
            LOGGER.info(f"You can now start moving the end-effector")
            success, _ = pilot.move_to_tcp_pose(new_pose, time_out=time_out)
            LOGGER.info(f"Success: {success}")
            LOGGER.info(f"Shutting down the demo")


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    keep_pose(True, 10.0)
