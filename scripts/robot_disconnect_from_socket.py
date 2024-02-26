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
from time import perf_counter as _t_now

# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)

# Fixed configurations
_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')

_start_j_pos = (3.6226, -1.6558, 1.8276, 0.1048, 2.0615, -1.4946)
_socket_obs_j_pos = ()

_T_socket2pre_connect = sm.SE3().Rt(R=sm.SO3.EulerVec((0.0, 0.0, -np.pi/2)), t=(0.0, 0.0, 0.034 - 0.02))
_T_socket2junction = sm.SE3().Rt(R=sm.SO3.EulerVec((0.0, 0.0, -np.pi/2)), t=(0.0, 0.0, 0.034))
_T_socket2post_connect = sm.SE3().Trans([0.0, 0.0, -0.02])
_T_socket2fpi = sm.SE3().Trans([0.0, 0.0, 0.034])


def disconnect_from_socket(opt: Namespace) -> None:
    """ Function to go through the disconnection procedure

    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.camera_factory.create('realsense_tcp_cam', logger_level=opt.logger_level)
    cam.load_coefficients()
    cam.render()
    dtt = pd.factory.create(_dtt_cfg_dir.joinpath(opt.detector_config_file))
    dtt.register_camera(cam)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.register_ee_cam(cam)
        with pilot.context.position_control():
            # Start at start position
            pilot.move_to_joint_pos(_start_j_pos)
            # Move to camera estimation pose to have all marker in camera field of view
            pilot.move_to_joint_pos(_socket_obs_j_pos)

        # Search for ArUco marker
        found = False
        # Use time out to exit loop
        _t_out = 5.0
        _t_start = _t_now()
        while _t_now() - _t_start <= _t_out and not found:
            # Give the robot time to stop
            time.sleep(1.0)
            found, T_cam2socket = dtt.find_pose(render=True)
            if found:
                # Get transformation matrices
                T_flange2cam = pilot.cam_mdl.T_flange2camera
                T_base2flange = pilot.get_pose('flange')
                # Search for transformation from base to socket
                T_base2socket = T_base2flange * T_flange2cam * T_cam2socket
                T_base2pre_connect = T_base2socket * _T_socket2pre_connect
                T_base2junction = T_base2socket * _T_socket2junction
                # T_base2post_connect = T_base2socket * _T_socket2post_connect

        if not found:
            # Move back to start
            with pilot.context.position_control():
                pilot.move_to_joint_pos(_start_j_pos)
        else:
            # Move to pre connect pose to get ready for taking the plug
            with pilot.context.position_control():
                pilot.move_to_tcp_pose(T_base2pre_connect)

            # Getting in junction between plug and robot
            with pilot.context.motion_control():
                pilot.move_to_tcp_pose(T_base2junction, time_out=5.0)
                # Check if robot is in target area
                xyz_base2jct_base_est = T_base2junction.t
                xyz_base2jct_base_meas = pilot.robot.tcp_pos
                error = np.linalg.norm(xyz_base2jct_base_est - xyz_base2jct_base_meas)
                if error > 0.01:
                    raise RuntimeError(f"Remaining position error {error} to alignment state is to large. "
                                       f"Robot is probably in an undefined condition.")
            # Start to apply some force
            with pilot.context.force_control():
                # Fix plug via twisting end-effector
                success = pilot.screw_ee_force_mode(4.0, np.pi / 2, 12.0)
                if not success:
                    raise RuntimeError(f"Robot did not succeed in closing the twist lock. "
                                       f"Robot is probably in an undefined condition.")
                # Plug out
                plug_out_ft = np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
                success = pilot.tcp_force_mode(
                    wrench=plug_out_ft,
                    compliant_axes=[0, 0, 1, 0, 0, 0],
                    distance=0.06,  # 6cm
                    time_out=10.0)
                if not success:
                    raise RuntimeError(f"Error while trying to unplug. Plug is probably still connected.")

            # End at home position
            time.sleep(2.0)
            with pilot.context.position_control():
                pilot.robot.move_home()

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
        args.logger_level = logging.DEBUG
    else:
        args.logger_level = logging.INFO
    # Start procedure
    disconnect_from_socket(args)


if __name__ == '__main__':
    main()
