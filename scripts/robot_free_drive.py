""" Script to move the robot in free drive mode. """
import logging
import ur_pilot
import camera_kit as ck
from config import config_data


LOGGER = logging.getLogger(__name__)


def free_drive() -> None:
    # Use camera for user interaction
    cam = ck.camera_factory.create("realsense_tcp_cam")
    # Connect to API
    with ur_pilot.connect(config_data.robot_dir) as pilot:
        # Start free drive mode
        LOGGER.info("Hit key 'S' to print out the current state")
        with pilot.plug_model.context('type2_female'):
            with pilot.context.teach_in_control():
                while not ck.user.stop():
                    if ck.user.save():
                        # Read joint and pose information
                        joint_pos = pilot.robot.joint_pos
                        # tcp_pose = pilot.robot.tcp_pose
                        tcp_pose = pilot.get_pose(ur_pilot.EndEffectorFrames.PLUG_LIP)
                        # Print out
                        LOGGER.info(f"        Joint positions: {ur_pilot.utils.vec_to_str(joint_pos)}")
                        LOGGER.info(f"Transformation Base-TCP: {ur_pilot.utils.se3_to_str(tcp_pose)}")
                    cam.render()
    cam.end()


if __name__ == '__main__':
    free_drive()
