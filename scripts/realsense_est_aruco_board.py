""" Script to estimate the ChArUco board pose """
from __future__ import annotations

# global
import logging
import cvpd as pd
import camera_kit as ck
from pathlib import Path
from time import perf_counter


LOGGER = logging.getLogger(__name__)

_log_freq = 1
_cfg_fp = Path(__file__).absolute().parent.parent.joinpath('detector/aruco_pattern_test_adj.yaml')
# _cfg_fp = Path(__file__).absolute().parent.parent.joinpath('detector/charuco_ads_socket_ty2.yaml')


def find_pose() -> None:

    with ck.camera_manager('realsense_tcp_cam', logger_level=logging.INFO) as cam:
        dtt = pd.factory.create(_cfg_fp)
        dtt.register_camera(cam)
        log_interval = 1.0 / _log_freq
        _t_start = perf_counter()
        while not ck.user.stop():
            found, T_cam2obj = dtt.find_pose(render=True)
            if perf_counter() - _t_start > log_interval and found:
                print(f"Transformation Camera - Object: {T_cam2obj.t.tolist()} {T_cam2obj.eulervec().tolist()}")
                _t_start = perf_counter()


if __name__ == '__main__':
    find_pose()
