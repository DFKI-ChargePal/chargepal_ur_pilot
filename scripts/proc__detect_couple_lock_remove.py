""" Script for testing the complete 'unplugging process'.
    This means:
        1) Detecting the socket
        2) Couple robot with plug
        3) Lock robot-plug closure
        4) Remove plug from socket
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
                            pilot.set_tcp(ur_pilot.EndEffectorFrames.PLUG_SAFETY)
                            T_base2obs_close = T_base2marker * _T_marker2obs_close
                            pilot.move_to_tcp_pose(T_base2obs_close)
            found, T_base2socket = pilot.find_target_pose(config_data.detector_configs['ii'],
                                                          time_out=config_data.detector_time_out,
                                                          render=True)
            if found:
                with pilot.context.position_control():
                    pilot.try2_approach_to_plug(T_base2socket)

                sus_cp, sus_lp, sus_rp = False, False, False
                with pilot.context.force_control():
                    sus_cp, lin_ang_err = pilot.try2_couple_to_plug(T_base2socket)
                    LOGGER.info(f"Coupling robot and plug successfully: {sus_cp}")
                    LOGGER.debug(f"Final error after coupling robot and plug: "
                                f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

                    if sus_cp:
                        sus_lp, lin_ang_err = pilot.try2_lock_plug(T_base2socket)
                        LOGGER.info(f"Lock robot with plug successfully: {sus_lp}")
                        LOGGER.debug(f"Final error after locking robot with plug: "
                                    f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

                    if sus_lp:
                        sus_rp, lin_ang_err = pilot.try2_remove_plug()
                        LOGGER.info(f"Remove plug from socket successfully: {sus_lp}")
                        LOGGER.debug(f"Final error after removing plug from socket: "
                                    f"(Linear error={lin_ang_err[0]}[m] | Angular error={lin_ang_err[1]}[rad])")

        with pilot.context.position_control():
            if sus_rp:
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
