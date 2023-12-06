""" Script to move the robot in free drive mode. """
import ur_pilot
import chargepal_aruco as ca


def free_drive() -> None:
    # Use camera for user interaction
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    display = ca.Display('Monitor')
    # Connect to API
    with ur_pilot.connect() as pilot:
        # Start free drive mode
        with pilot.teach_in_control():
            while not (ca.EventObserver.state is ca.EventObserver.Type.QUIT):
                img = cam.get_color_frame()
                display.show(img)


if __name__ == '__main__':
    free_drive()
