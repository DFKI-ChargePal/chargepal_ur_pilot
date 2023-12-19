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

# _marker_pose_estimation_cfg_joints = [3.445, -1.558, 1.940, -0.243, 1.854, -1.573]
_socket_obs_j_pos = (3.387, -1.469, 1.747, -0.016, 1.789, -1.565)

# _pose_socket2hook = Pose().from_xyz([0.0, 0.0, -0.097])
_pose_socket2hook = Pose().from_xyz([0.0-0.01, 0.0, 0.034])

# 2 cm safety distance in Z-direction + shifted in x-direction
# _pose_socket2hook_pre = Pose().from_xyz([0.03, 0.0, -(0.097 + 0.02)])
_pose_socket2hook_pre = Pose().from_xyz([0.03, 0.0, 0.034 - 0.02])

# Intermediate step to hook up plug
# _pose_socket2hook_itm = Pose().from_xyz([0.03, 0.0, -0.097])
_pose_socket2hook_itm = Pose().from_xyz([0.03, 0.0, 0.034])

_pose_socket2socket_post = Pose().from_xyz(xyz=[0.0, 0.0, 0.0 - 0.05])


def disconnect_from_socket(opt: Namespace) -> None:
    """ Function to go through the disconnection procedure
    
    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.create('realsense_tcp_cam', logger_level=opt.logger_level)
    cam.load_coefficients()
    cam.render()
    dtt: pd.Detector
    if opt.detector_config_file.startswith("aruco_marker"):
        dtt = pd.ArucoMarkerDetector(_dtt_cfg_dir.joinpath(opt.detector_config_file))
    elif opt.detector_config_file.startswith("charuco"):
        dtt = pd.CharucoDetector(_dtt_cfg_dir.joinpath(opt.detector_config_file))
    dtt.register_camera(cam)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.robot.register_ee_cam(cam)
        with pilot.position_control():
            # Start at home position
            pilot.move_home()
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
            found, pose_cam2socket = dtt.find_pose(render=True)
            if found:
                # Search for transformation from base to plug hook
                # Get transformation matrices
                T_plug2cam = pilot.robot.cam_mdl.T_flange2camera
                T_base2plug = pilot.robot.get_tcp_pose().transformation
                T_socket2hook = _pose_socket2hook.transformation
                T_socket2hook_pre = _pose_socket2hook_pre.transformation
                T_socket2hook_itm = _pose_socket2hook_itm.transformation
                T_socket2socket_post = _pose_socket2socket_post.transformation
                T_cam2socket = Pose().from_xyz_xyzw(*pose_cam2socket).transformation
                # Apply transformation chain
                T_base2socket = T_base2plug @ T_plug2cam @ T_cam2socket
                # print(T_base2socket.pose.xyz, T_base2socket.pose.axis_angle)
                T_base2socket = Pose().from_xyz((0.908, 0.294, 0.477)).from_axis_angle((-0.006, 1.568, 0.001)).transformation
                LOGGER.warn(f"Use fix pose for target")
                T_base2hook = T_base2socket @ T_socket2hook
                T_base2hook_pre = T_base2socket @ T_socket2hook_pre
                T_base2hook_itm = T_base2socket @ T_socket2hook_itm
                T_base2socket_post = T_base2socket @ T_socket2socket_post

        if not found:
            # Move back to home
            with pilot.position_control():
                pilot.move_home()
        else:
            with pilot.position_control():
                # Move to pre-pose to hook up to plug
                pilot.move_to_tcp_pose(T_base2hook_pre.pose)
            with pilot.motion_control():
                pilot.move_to_tcp_pose(T_base2hook_itm.pose, time_out=5.0)
                pilot.move_to_tcp_pose(T_base2hook.pose, time_out=5.0)
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
    parser.add_argument('detector_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--with_sensing', action='store_true', 
                        help='Option to use active end-effector sensing for better depth estimation')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    ur_pilot.utils.check_file_extension(Path(args.detector_config_file), '.yaml')
    if args.debug:
        args.logger_level = logging.DEBUG
    else:
        args.logger_level = logging.INFO
    
    disconnect_from_socket(args)


if __name__ == '__main__':
    main()
