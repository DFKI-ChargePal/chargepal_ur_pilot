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

# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)

_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')

# Fixed configurations
_socket_obs_j_pos = (3.387, -1.469, 1.747, -0.016, 1.789, -1.565)
_T_socket2socket_pre = sm.SE3().Trans([0.0, 0.0, 0.0 - 0.02])  # Retreat pose with respect to the socket
_T_fpi2junction = sm.SE3().Trans([0.0, 0.0, -0.034 + 0.01])
_T_socket2fpi = sm.SE3().Trans([0.0, 0.0, 0.034])


def connect_to_socket(opt: Namespace) -> None:
    """ Function to run through the connection procedure

    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.camera_factory.create("realsense_tcp_cam", opt.logging_level)
    cam.load_coefficients()
    cam.render()
    dtt = pd.factory.create(_dtt_cfg_dir.joinpath(opt.config_file))
    dtt.register_camera(cam)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.register_ee_cam(cam)
        with pilot.context.position_control():
            # Start at home position
            pilot.robot.move_home()
            # Move to camera estimation pose to have all marker in camera field of view
            pilot.move_to_joint_pos(_socket_obs_j_pos)

        # Search for ArUco marker
        found_socket = False
        # Use time out to exit loop
        time_out = 5.0
        _t_start = time.time()
        while time.time() - _t_start <= time_out and not found_socket:
            time.sleep(1.0)
            found, T_cam2socket = dtt.find_pose(render=True)
            if found:
                # Get transformation matrices
                T_plug2cam = pilot.cam_mdl.T_flange2camera
                T_base2plug = pilot.get_pose('tool_tip')
                # Get searched transformations
                T_base2socket = T_base2plug * T_plug2cam * T_cam2socket
                print(ur_pilot.utils.se3_to_str(T_base2socket))
                T_base2socket_pre = T_base2socket * _T_socket2socket_pre
                found_socket = True
    
        if not found_socket:
            # Move back to home
            with pilot.context.position_control():
                pilot.robot.move_home()
        else:
            with pilot.context.position_control():
                # Move to socket with some safety distance
                pilot.move_to_tcp_pose(T_base2socket_pre)
                T_base2fpi = T_base2socket * _T_socket2fpi
                T_base2junction = T_base2fpi * _T_fpi2junction
            time.sleep(1.0)
            ck.user.wait_for_command()
            # Getting in junction between plug and socket
            with pilot.context.motion_control():
                pilot.move_to_tcp_pose(T_base2junction, time_out=4.0)
                # Check if robot is in target area
                xyz_base2jct_base_est = T_base2junction.t
                xyz_base2jct_base_meas = pilot.robot.tcp_pos
                error = np.linalg.norm(xyz_base2jct_base_est - xyz_base2jct_base_meas)
                if error > 0.01:
                    raise RuntimeError(f"Remaining position error {error} to alignment state is to large. "
                                       f"Robot is probably in an undefined condition.")
            # Start to apply some force
            with pilot.context.force_control():
                # Try to fully plug in
                pilot.plug_in_force_ramp(f_axis='z', f_start=50.0, f_end=100, duration=3.0)
                # Check if robot is in target area
                xyz_base2fpi_base_est = T_base2fpi.t
                xyz_base2fpi_base_meas = pilot.robot.tcp_pos
                error = np.linalg.norm(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)
                if error > 0.01:
                    print(f"The remaining position error {error} is quite large!")
                pilot.relax(2.0)
                # Release plug via twisting end-effector
                success = pilot.screw_ee_force_mode(4.0, -np.pi / 2, 12.0)
                if not success:
                    raise RuntimeError(f"Robot did not succeed in opening the twist lock. "
                                       f"Robot is probably in an undefined condition.")
                release_ft = np.array([0.0, 0.0, -25.0, 0.0, 0.0, 0.0])
                success = pilot.frame_force_mode(
                    wrench=release_ft,
                    compliant_axes=[1, 1, 1, 0, 0, 0],
                    distance=0.04,
                    time_out=5.0,
                    frame='flange'
                )
                if not success:
                    raise RuntimeError(
                        f"Error while trying to release the lock. Robot end-effector is probably still connected.")

            # End at retreat position
            time.sleep(2.0)
            with pilot.context.position_control():
                pilot.robot.movej([3.6226, -1.6558, 1.8276, 0.1048, 2.0615, -1.4946])

    # Stop camera stream
    cam.end()


def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('detector_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    ur_pilot.utils.check_file_extension(Path(args.detector_config_file), '.yaml')
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    # Connect to socket
    connect_to_socket(args)


if __name__ == '__main__':
    main()
