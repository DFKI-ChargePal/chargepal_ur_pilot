""" Script for testing the complete 'plug-in process'.
    This means:
        1) Detecting the socket
        2) Engaging plug and socket
        3) Insert plug to socket
        4) Unlock robot-plug closure
        5) Decouple robot from plug
"""
from __future__ import annotations

# libs
import time
import ur_pilot
import logging
import argparse
import cvpd as pd
import camera_kit as ck
import spatialmath as sm
from pathlib import Path
from time import perf_counter

# configuration module
from config import config_data

# typing
from ur_pilot import Pilot
from argparse import Namespace
from camera_kit import CameraBase


LOGGER = logging.getLogger(__name__)

_t_now = perf_counter

# Starting point
_socket_obs_j_pos = [3.4035, -1.6286, 2.1541, -0.4767, 1.8076, -1.5921]
_T_marker2obs_close = sm.SE3().Trans([0.022, -0.022, -0.15])


def detect(pilot: Pilot, cam: CameraBase, detector_fp: Path) -> tuple[bool, sm.SE3]:
    """ Function to run a new detection. """
    # Perception setup
    dtt = pd.factory.create(detector_fp)
    dtt.register_camera(cam)
    found, T_base2target = False, sm.SE3()
    t_start = _t_now()
    while _t_now() - t_start <= config_data.detector_time_out and not found:
        time.sleep(1/3)
        found, T_cam2target = dtt.find_pose(render=True)
        if found:
            # Get transformation matrices
            T_flange2cam = pilot.cam_mdl.T_flange2camera
            T_base2flange = pilot.get_pose('flange')
            # Get searched transformation
            T_base2target = T_base2flange * T_flange2cam * T_cam2target
    return found, T_base2target


def main(opt: Namespace) -> None:
    """ Main function to start process. """
    ur_pilot.base_logger.set_logging_level(opt.logging_level)
    # The object 'data' include the configuration
    LOGGER.info(config_data)
    # Perception setup
    cam = ck.camera_factory.create(config_data.camera_name, opt.logging_level)
    cam.load_coefficients(config_data.camera_cc)
    cam.render()

    with ur_pilot.connect(config_data.robot_dir) as pilot:
        pilot.register_ee_cam(cam, config_data.camera_dir)
        with pilot.plug_model.context(config_data.robot_plug_type):
            with pilot.context.position_control():
                pilot.move_to_joint_pos(_socket_obs_j_pos)

            if config_data.detector_two_step_approach:
                found, T_base2marker = detect(pilot, cam, config_data.detector_configs['i'])
                if found:
                    with pilot.context.position_control():
                        pilot.set_tcp('plug_safety')
                        T_base2obs_close = T_base2marker * _T_marker2obs_close
                        pilot.move_to_tcp_pose(T_base2obs_close)
            found, T_base2socket = detect(pilot, cam, config_data.detector_configs['ii'])
            if found:
                with pilot.context.position_control():
                    pilot.set_tcp('plug_safety')
                    pilot.move_to_tcp_pose(T_base2socket)

                with pilot.context.force_control():
                    success_eng, lin_ang_err = pilot.try2_engage_with_socket(T_base2socket)
                    LOGGER.debug(f"Final error after engaging between plug and socket: "
                                 f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
                    if success_eng:
                        success_in, lin_ang_err = pilot.try2_insert_plug(T_base2socket)
                        LOGGER.debug(f"Final error after inserting plug to socket: "
                                     f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
                        if success_in:
                            success_ul, lin_ang_err = pilot.try2_unlock_plug(T_base2socket)
                            LOGGER.debug(f"Final error after unlocking robot from plug: "
                                         f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")





if __name__ == '__main__':
    # Input parsing
    parser = argparse.ArgumentParser(description="Script to show the 'detect-engage-insert-unlock-decouple' process")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    main(args)
