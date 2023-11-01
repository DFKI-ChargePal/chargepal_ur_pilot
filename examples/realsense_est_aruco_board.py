""" Script to estimate the ChArUco board pose """
from __future__ import annotations

# global
import logging
import numpy as np
import chargepal_aruco as ca

# typing
from numpy import typing as npt


def estimate_board_pose() -> None:

    camera = ca.RealSenseCamera("tcp_cam_realsense")
    camera.load_coefficients()
    aru_board = ca.CharucoBoard("DICT_4X4_100", 19, (10, 7), 25)
    detector = ca.CharucoboardDetector(camera, aru_board, display=True)

    pose: npt.NDArray[np.float64] | None = None
    while ca.EventObserver.state is not ca.EventObserver.Type.QUIT:
        _, pose = detector.find_pose()
        if detector.display and ca.EventObserver.state is ca.EventObserver.Type.PAUSE:
            print("Press any key to continue")
            ca.EventObserver.wait_for_user()
    if pose is not None:
        print(f"Pose found: \n {pose}")


if __name__ == '__main__':
    ca.set_logging_level(logging.INFO)
    estimate_board_pose()
