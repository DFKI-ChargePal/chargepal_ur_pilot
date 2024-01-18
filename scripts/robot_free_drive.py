""" Script to move the robot in free drive mode. """
import logging
import ur_pilot
import camera_kit as ck


LOGGER = logging.getLogger(__name__)


def free_drive() -> None:
    # Use camera for user interaction
    cam = ck.camera_factory.create("realsense_tcp_cam")
    # Connect to API
    with ur_pilot.connect() as pilot:
        # Start free drive mode
        LOGGER.info("Hit key 'S' to print out the current state")
        with pilot.teach_in_control():
            while not ck.user.stop():
                if ck.user.save():
                    # Read joint and pose information
                    joint_pos = pilot.robot.get_joint_pos()
                    tcp_pose = pilot.robot.get_tcp_pose()
                    # Print out
                    LOGGER.info("\n      Joint positions: " + " ".join(f"{q:.3f}" for q in joint_pos))
                    LOGGER.info("TCP pose w. axis ang.: " + " ".join(f"{p:.3f}" for p in tcp_pose.xyz + tcp_pose.axis_angle))

                cam.render()
    cam.end()


if __name__ == '__main__':
    free_drive()
