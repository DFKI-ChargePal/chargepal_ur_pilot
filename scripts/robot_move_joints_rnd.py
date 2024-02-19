""" Script to move the robot arm randomly around the home position """
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    with ur_pilot.connect() as pilot:
        with pilot.context.position_control():
            rnd_joint_pos = pilot.move_joints_random()
            # Print result
            LOGGER.info(f"Target joint positions: {ur_pilot.utils.vec_to_str(rnd_joint_pos, 3)}")
