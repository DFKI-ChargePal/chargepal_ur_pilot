""" Script to move the robot in free drive mode. """
import ur_pilot
import camera_kit as ck


def free_drive() -> None:
    # Use camera for user interaction
    cam = ck.create("realsense_tcp_cam")
    # Connect to API
    with ur_pilot.connect() as pilot:
        # Start free drive mode
        with pilot.teach_in_control():
            while not ck.user.stop():
                cam.render()
    cam.end()


if __name__ == '__main__':
    free_drive()
