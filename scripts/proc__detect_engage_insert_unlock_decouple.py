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
_retreat_j_pos = [3.4035, -1.6286, 2.1541, -0.4767, 1.8076, -1.5921]
_socket_obs_j_pos = [3.4518, -2.0271, 2.1351, -0.0672, 1.8260, -1.5651]
_T_marker2obs_close = sm.SE3().Rt(R=sm.SO3.EulerVec((0.0, 0.15, 0.0)), t=(0.1, -0.025, -0.1))


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

            # with pilot.context.force_control():
            #     T_base2socket = sm.SE3.Rt(R=sm.SO3.EulerVec((0.005, 1.568, -0.010)), t=(0.5799, 0.3177, 0.2683))
            #     success_up, _ = pilot.try2_unlock_plug(T_base2socket)
            #     LOGGER.debug(f"Unlocking successfully: {success_up}")

            #     success_dp, lin_ang_err = pilot.try2_decouple_to_plug()
            #     LOGGER.debug(f"Decoupling successfully: {success_dp}")
            #     LOGGER.debug(f"Final error after decoupling robot from plug: "
            #                  f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")


            if config_data.detector_two_step_approach:
                for _ in range(6):
                    found, T_base2marker = detect(pilot, cam, config_data.detector_configs['i'])
                    if found:
                        with pilot.context.position_control():
                            pilot.set_tcp('plug_safety')
                            T_base2obs_close = T_base2marker * _T_marker2obs_close
                            pilot.move_to_tcp_pose(T_base2obs_close)
            found, T_base2socket = detect(pilot, cam, config_data.detector_configs['ii'])
            # if found:
            #     with pilot.context.position_control():
            #         pilot.set_tcp('plug_safety')
            #         pilot.move_to_tcp_pose(T_base2socket)

            #     success_eng, success_in, success_in, success_ul, success_dc = False, False, False, False, False
            #     with pilot.context.force_control():
            #         success_eng, lin_ang_err = pilot.try2_engage_with_socket(T_base2socket)
            #         LOGGER.debug(f"Final error after engaging between plug and socket: "
            #                      f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
            #         if success_eng:
            #             success_in, lin_ang_err = pilot.try2_insert_plug(T_base2socket)
            #             LOGGER.debug(f"Final error after inserting plug to socket: "
            #                          f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
            #         if success_in:
            #             success_ul, lin_ang_err = pilot.try2_unlock_plug(T_base2socket)
            #             LOGGER.debug(f"Final error after unlocking robot from plug: "
            #                          f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

            #         if success_ul:
            #             success_dc, lin_ang_err = pilot.try2_decouple_to_plug()
            #             LOGGER.debug(f"Final error after decoupling robot from plug: "
            #                          f"(Linera error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

            # with pilot.context.position_control():
            #     if success_dp:
            #         pilot.move_to_joint_pos(_retreat_j_pos)


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
