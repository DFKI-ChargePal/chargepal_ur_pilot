""" Script to estimate the ChArUco board pose """
from __future__ import annotations

# global
import logging
import cvpd as pd
import camera_kit as ck
from pathlib import Path


LOGGER = logging.getLogger(__name__)

_cfg_fp = Path(__file__).absolute().parent.joinpath('detector', 'charuco_calibration.yaml')


def estimate_board_pose() -> None:

    with ck.camera_manager('realsense_tcp_cam', logger_level=logging.INFO) as cam:
        cam.load_coefficients()
        pose_detector = pd.CharucoDetector(_cfg_fp)
        pose_detector.register_camera(cam)
        found, pose = False, None
        while not ck.user.stop():
            found, pose = pose_detector.find_pose(render=True)
        if found:
            LOGGER.info(f"Last found pose: \n {pose}")


if __name__ == '__main__':
    estimate_board_pose()
