""" 
Script to print the current robot state.
Contains: Joint positions
          TCP pose
"""
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    # Connect to API
    with ur_pilot.connect() as pilot:
        # Read joint and pose information
        joint_pos = pilot.robot.joint_pos
        tcp_pose = pilot.robot.tcp_pose
        # Print out
        LOGGER.info("      Joint positions: " + " ".join(f"{q:.3f}" for q in joint_pos))
        LOGGER.info("TCP pose w. axis ang.: " + " ".join(f"{p:.3f}" for p in tcp_pose.t.tolist() + tcp_pose.eulervec().tolist()))
