""" Short demo to demonstrate motion mode. Goal is to keep the current pose """
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


def keep_pose(soft: bool, time_out: float) -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        if soft:
            pilot.cfg.robot.motion_mode.Kp = [1.0, 1.0, 1.0, 0.01, 0.01, 0.01]
        else:
            pilot.cfg.robot.motion_mode.Kp = [25.0, 25.0, 25.0, 1.0, 1.0, 1.0]
        
        with pilot.context.motion_control():
            init_pose = pilot.robot.get_tcp_pose()
            LOGGER.info(f"You can now start moving the end-effector")
            pilot.move_to_tcp_pose(init_pose, time_out=time_out)
            LOGGER.info(f"Shutting down the demo")


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    keep_pose(False, 30.0)
