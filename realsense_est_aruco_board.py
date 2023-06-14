""" Script to estimate the ChArUco board pose """

# global
import logging
import chargepal_aruco as ca


def estimate_board_pose() -> None:

    camera = ca.RealSenseCamera("tcp_cam_realsense")
    camera.load_coefficients()
    aru_board = ca.CharucoBoard("DICT_4X4_100", 19, (10, 7), 25)
    detector = ca.CharucoboardDetector(camera, aru_board, display=True)

    while ca.EventObserver.state is not ca.EventObserver.Type.QUIT:
        _, _poses = detector.find_board_pose()
        if detector.display and ca.EventObserver.state is ca.EventObserver.Type.PAUSE:
            print("Press any key to continue")
            ca.EventObserver.wait_for_user()
    print(f"Poses found: \n {_poses}")


if __name__ == '__main__':
    ca.set_logging_level(logging.INFO)
    estimate_board_pose()
