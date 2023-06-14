
# global
import chargepal_aruco as ca

# local
from ur_pilot import URPilot


def free_drive() -> None:

    # Use camera for user interaction
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    display = ca.Display('Monitor')

    # Connect to robot
    ur10 = URPilot()
    # Enable free drive mode
    ur10.teach_mode()

    while True:
        img = cam.get_color_frame()
        display.show(img)
        if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
            break

    # Stop free drive mode
    ur10.stop_teach_mode()
    # Clean up
    ur10.exit()


if __name__ == '__main__':
    free_drive()
