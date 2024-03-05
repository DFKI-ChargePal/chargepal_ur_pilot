""" Script for testing the detection and plugging process. Means inserting and removing the plug in one go. """
from __future__ import annotations

# libs
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import numpy as np
import camera_kit as ck
import spatialmath as sm

from pathlib import Path
from time import perf_counter

# configuration
from config import data

# typing
from ur_pilot import Pilot
from argparse import Namespace


LOGGER = logging.getLogger(__name__)

# Fixed configurations
_socket_obs_j_pos = [3.4035, -1.6286, 2.1541, -0.4767, 1.8076, -1.5921]  # (3.4045, -1.6260, 1.8820, 0.1085, 1.8359, -1.5003)

_T_socket2socket_pre = sm.SE3().Trans([0.0, 0.0, -0.02])  # Retreat pose with respect to the socket
_T_socket2socket_junction = sm.SE3().Trans([0.0, 0.0, 0.015])
_T_socket2fpi = sm.SE3().Trans([-0.005, 0.0, 0.034])


def detect(pilot: Pilot, cam: ck.CameraBase, opt: Namespace) -> tuple[bool, sm.SE3]:
    """ Function to run detect socket pose. """
    # Perception setup
    dtt = pd.factory.create(data.detector_dir.joinpath(opt.detector_config_file))
    dtt.register_camera(cam)
    found, T_base2socket = False, sm.SE3()
    _t_out, _t_start = 2.0, perf_counter()
    while perf_counter() - _t_start <= _t_out and not found:
        time.sleep(0.33)
        found, T_cam2socket = dtt.find_pose(render=True)
        if found:
            # Get transformation matrices
            T_flange2cam = pilot.cam_mdl.T_flange2camera
            T_base2flange = pilot.get_pose('flange')
            # Get searched transformations
            T_base2socket = T_base2flange * T_flange2cam * T_cam2socket
    return found, T_base2socket


def move_to_pre(pilot: Pilot, T_base2socket: sm.SE3) -> bool:
    T_base2socket_pre = T_base2socket * _T_socket2socket_pre
    success, _ = pilot.move_to_tcp_pose(T_base2socket_pre)
    return success


def get_in_touch(pilot: Pilot, T_base2socket: sm.SE3) -> bool:
    T_base2socket_junction = T_base2socket * _T_socket2socket_junction
    pilot.move_to_tcp_pose(T_base2socket_junction, time_out=4.0)
    # Check if robot is in target area
    xyz_base2jct_base_est = T_base2socket_junction.t
    xyz_base2jct_base_meas = pilot.robot.tcp_pos
    error_z = np.squeeze(sm.SO3(pilot.robot.tcp_pose.R) * (xyz_base2jct_base_est - xyz_base2jct_base_meas))[-1]
    if abs(error_z) < 0.015:
        success = True
    else:
        LOGGER.warning(f"Remaining position error {error_z} to alignment state is quite large. "
                       f"Robot is probably in an undefined condition.")
        success = False
    return success


def plug_in(pilot: Pilot, T_base2socket: sm.SE3) -> bool:
    # Try to fully plug in
    T_base2fpi = T_base2socket * _T_socket2fpi
    pilot.plug_in_force_ramp(f_axis='z', f_start=60.0, f_end=100, duration=3.0)
    # Check if robot is in target area
    xyz_base2fpi_base_est = T_base2fpi.t
    xyz_base2fpi_base_meas = pilot.robot.tcp_pos
    error = np.linalg.norm(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)
    if error < 0.01:
        success = True
    else:
        LOGGER.warning(f"Remaining position error {error} is to large. "
                       f"Robot is probably in an undefined condition.")
        success = False
    return success


def plug_out(pilot: Pilot, T_base2socket: sm.SE3) -> bool:
    T_base2fpi = T_base2socket * _T_socket2fpi
    # Start to apply some force
    with pilot.context.force_control():
        pilot.relax(2.0)
        # Plug out again
        plug_out_ft = np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
        success = pilot.tcp_force_mode(
            wrench=plug_out_ft,
            compliant_axes=[0, 0, 1, 0, 0, 0],
            distance=0.06,  # 6cm
            time_out=10.0)
        if not success:
            LOGGER.warning(f"Error while trying to unplug. Plug is probably still connected.")
    return success


def main(opt: Namespace) -> None:
    """ Main function to start process. """
    LOGGER.info(data)
    # Perception setup
    cam = ck.camera_factory.create(opt.camera_name, opt.logging_level)
    calib_dir = data.camera_info_dir.joinpath(opt.camera_name, 'calibration')
    cam.load_coefficients(calib_dir.joinpath('coefficients.toml'))
    cam.render()

    # Connect to arm
    with ur_pilot.connect(data.robot_dir) as pilot:
        pilot.register_ee_cam(cam, calib_dir)

        with pilot.context.position_control():
            pilot.move_to_joint_pos(_socket_obs_j_pos)

        found, T_base2socket = detect(pilot, cam, opt)
        if found:
            with pilot.context.position_control():
                move_to_pre(pilot, T_base2socket)

            with pilot.context.motion_control():
                get_in_touch(pilot, T_base2socket)

            with pilot.context.force_control():
                success = plug_in(pilot, T_base2socket)
                if not success:
                    raise RuntimeError(f"Robot did not succeed in the plugging the plug in. "
                                       f"Robot is probably in an undefined condition.")
                success = plug_out(pilot, T_base2socket)
                if not success:
                    raise RuntimeError(f"Error while trying to unplug. Plug is probably still connected.")

        with pilot.context.position_control():
            pilot.robot.move_home()

    # Stop camera stream
    cam.end()


if __name__ == '__main__':
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('detector_config_file', type=str,
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--camera_name', type=str, default='realsense_tcp_cam', help='Camera name')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    ur_pilot.utils.check_file_extension(Path(args.detector_config_file), '.yaml')
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    main(args)
