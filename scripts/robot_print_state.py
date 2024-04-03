""" 
Script to print the current robot state.
Contains: Joint positions
          TCP pose
"""
import config
import logging
import ur_pilot

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    ur_pilot.logger.set_logging_level(logging.INFO)
    # Connect to API
    with ur_pilot.connect(config_dir=config.config_data.robot_dir) as pilot:
        # Read joint and pose information
        joint_pos = pilot.robot.joint_pos
        tcp_pose = pilot.robot.tcp_pose
        # Print out
        LOGGER.info(f"        Joint positions: {ur_pilot.utils.vec_to_str(joint_pos)}")
        LOGGER.info(f"Transformation Base-TCP: {ur_pilot.utils.se3_to_str(tcp_pose)}")

        print(pilot.get_pose('flange').t, pilot.get_pose('flange').eulervec()) # rpy(unit="deg", order="zyx"))
