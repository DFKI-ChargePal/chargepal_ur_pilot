from __future__ import annotations

# global
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import camera_kit as ck
from pathlib import Path
from time import perf_counter as _t_now
from rigmopy import Pose, Vector6d

# typing
from argparse import Namespace

LOGGER = logging.getLogger(__name__)

_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')

_marker_pose_estimation_cfg_joints = [3.409, -1.578, 2.062, -0.464, 1.809, -1.565]

_pose_socket2hook = Pose().from_xyz([0.0, 0.0, -0.097])

# 2 cm safety distance in Z-direction + shifted in x-direction
_pose_socket2hook_pre = Pose().from_xyz([0.03, 0.0, -(0.097 + 0.02)])  

# Intermediate step to hook up plug
_pose_socket2hook_itm = Pose().from_xyz([0.03, 0.0, -0.097])


def disconnect_from_socket(opt: Namespace) -> None:
    """ Function to go through the disconnection procedure
    
    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.create('realsense_tcp_cam', logger_level=opt.logger_level)
    cam.load_coefficients()
    cam.render()
    dtt = pd.ArucoMarkerDetector(_dtt_cfg_dir.joinpath(opt.marker_config_file))
    dtt.register_camera(cam)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.robot.register_ee_cam(cam)
        with pilot.position_control():
            # Start at home position
            pilot.move_home()
            # Move to camera estimation pose to have all marker in camera field of view
            pilot.move_to_joint_pos(_marker_pose_estimation_cfg_joints)

        # Search for ArUco marker
        found = False
        # Use time out to exit loop
        _t_out = 5.0
        _t_start = _t_now()
        while _t_now() - _t_start <= _t_out and not found:
            # Give the robot time to stop
            time.sleep(1.0)
            found, pose_cam2socket = dtt.find_pose(render=True)
            if found:
                # Search for transformation from base to plug hook
                # Get transformation matrices
                T_plug2cam = pilot.robot.cam_mdl.T_flange2camera
                T_base2plug = pilot.robot.get_tcp_pose().transformation
                T_socket2hook = _pose_socket2hook.transformation
                T_socket2hook_pre = _pose_socket2hook_pre.transformation
                T_socket2hook_itm = _pose_socket2hook_itm.transformation
                T_cam2socket = Pose().from_xyz_xyzw(*pose_cam2socket).transformation
                # Apply transformation chain
                T_base2socket = T_base2plug @ T_plug2cam @ T_cam2socket
                T_base2hook = T_base2socket @ T_socket2hook
                T_base2hook_pre = T_base2socket @ T_socket2hook_pre
                T_base2hook_itm = T_base2socket @ T_socket2hook_itm

        if not found:
            # Move back to home
            with pilot.position_control():
                pilot.move_home()
        else:
            with pilot.position_control():
                # Move to pre-pose to hook up to plug
                pilot.move_to_tcp_pose(T_base2hook_pre.pose)
            time.sleep(1.0)
            with pilot.motion_control():
                pilot.move_to_tcp_pose(T_base2hook_itm.pose, time_out=5.0)
                LOGGER.info(f"Push any key to continue.")
                ck.user.wait_for_command()
                pilot.move_to_tcp_pose(T_base2hook.pose, time_out=5.0)
            LOGGER.info(f"Push any key to continue.")
            ck.user.wait_for_command()

            with pilot.force_control():
                success = pilot.tcp_force_mode(
                    wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -150.0, 0.0, 0.0, 0.0]),
                    compliant_axes=[0, 0, 1, 0, 0, 0],
                    distance=0.05,
                    time_out=10.0)
                time.sleep(1.0)
            if success:
                # Move back to home
                with pilot.position_control():
                    pilot.move_home()
            else:
                LOGGER.error(f"Error while trying to disconnect. Plug might still be in the socket.\n"
                            f"Robot will stop moving and shut down...")
    # Stop camera stream
    cam.end()


def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('marker_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--with_sensing', action='store_true', 
                        help='Option to use active end-effector sensing for better depth estimation')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    ur_pilot.utils.check_file_extension(Path(args.marker_config_file), '.yaml')
    if args.debug:
        args.logger_level = logging.DEBUG
    else:
        args.logger_level = logging.INFO
    
    disconnect_from_socket(args)


if __name__ == '__main__':
    main()
