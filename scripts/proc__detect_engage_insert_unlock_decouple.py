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
import ur_pilot
import logging
import argparse
import camera_kit as ck
import spatialmath as sm
# configuration module
from config import config_data

# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)


# Reference positions
_retreat_j_pos = [3.4035, -1.6286, 2.1541, -0.4767, 1.8076, -1.5921]
_socket_obs_j_pos = [3.4518, -2.0271, 2.1351, -0.0672, 1.8260, -1.5651]
_T_marker2obs_close = sm.SE3().Rt(R=sm.SO3.EulerVec((0.0, 0.15, 0.0)), t=(0.1, -0.025, -0.1))


def main(opt: Namespace) -> None:
    """ Main function to start process. """
    ur_pilot.base_logger.set_logging_level(opt.logging_level)
    # The object 'config_data' include the configuration
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
                for _ in range(2):
                    found, T_base2marker = pilot.find_target_pose(config_data.detector_configs['i'],
                                                                  time_out=config_data.detector_time_out,
                                                                  render=True)
                    if found:
                        with pilot.context.position_control():
                            pilot.set_tcp('plug_safety')
                            T_base2obs_close = T_base2marker * _T_marker2obs_close
                            pilot.move_to_tcp_pose(T_base2obs_close)
            found, T_base2socket = pilot.find_target_pose(config_data.detector_configs['ii'],
                                                          time_out=config_data.detector_time_out,
                                                          render=True)
            if found:
                with pilot.context.position_control():
                    pilot.try2_approach_to_socket(T_base2socket)
  
                success_ep, success_ip, success_up, success_dp = False, False, False, False
                with pilot.context.force_control():
                    success_ep, lin_ang_err = pilot.try2_engage_with_socket(T_base2socket)
                    LOGGER.info(f"Engaging plug to socket successfully: {success_ep}")
                    LOGGER.debug(f"Final error after engaging between plug and socket: "
                                 f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
                    if success_ep:
                        success_ip, lin_ang_err = pilot.try2_insert_plug(T_base2socket)
                        LOGGER.info(f"Inserting plug to socket successfully: {success_ip}")
                        LOGGER.debug(f"Final error after inserting plug to socket: "
                                     f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")
                    if success_ip:
                        success_up, lin_ang_err = pilot.try2_unlock_plug(T_base2socket)
                        LOGGER.info(f"Unlock robot from plug successfully: {success_up}")
                        LOGGER.debug(f"Final error after unlocking robot from plug: "
                                     f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

                    if success_up:
                        success_dp, lin_ang_err = pilot.try2_decouple_to_plug()
                        LOGGER.info(f"Decoupling from plug successfully: {success_dp}")
                        LOGGER.debug(f"Final error after decoupling robot from plug: "
                                     f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

            with pilot.context.position_control():
                if success_dp:
                    pilot.move_to_joint_pos(_retreat_j_pos)


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
