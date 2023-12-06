""" Script to move the robot arm randomly around the home position """
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    with ur_pilot.connect() as pilot:
        with pilot.position_control():
            rnd_joint_pos = pilot.move_joints_random()
            # Print result
            LOGGER.info("Target joint positions: " + " ".join(f"{q:.3f}" for q in rnd_joint_pos))
