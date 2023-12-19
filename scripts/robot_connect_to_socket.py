from __future__ import annotations

# global
import time
import logging
import argparse
import ur_pilot
import cv2 as cv
import cvpd as pd
import numpy as np
import camera_kit as ck
from pathlib import Path
from rigmopy import Vector3d, Vector6d, Pose, Quaternion

# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)

_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')

# Fixed configurations
# SOCKET_POSE_ESTIMATION_CFG_J = [3.148, -1.824, 2.096, -0.028, 1.590, -1.565]
# SOCKET_POSE_ESTIMATION_CFG_J = [3.409, -1.578, 2.062, -0.464, 1.809, -1.565]
_socket_obs_j_pos = (3.387, -1.469, 1.747, -0.016, 1.789, -1.565)
X_SOCKET_2_SOCKET_PRE = Pose().from_xyz(xyz=[0.0, 0.0, 0.0 - 0.02])  # Retreat pose with respect to the socket
# X_SOCKET_2_SOCKET_IN = Pose().from_xyz(xyz=[0.0, 0.0, 0.05])
X_SOCKET_2_FPI = Pose().from_xyz(xyz=[0.0, 0.0, 0.034])


def connect_to_socket(opt: Namespace) -> None:
    """ Function to run through the connection procedure

    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.create("realsense_tcp_cam", opt.logging_level)
    cam.load_coefficients()
    cam.render()
    dtt: pd.Detector
    if opt.detector_config_file.startswith("aruco_marker"):
        dtt = pd.ArucoMarkerDetector(_dtt_cfg_dir.joinpath(opt.detector_config_file))
    elif opt.detector_config_file.startswith("charuco"):
        dtt = pd.CharucoDetector(_dtt_cfg_dir.joinpath(opt.detector_config_file))
    else:
        raise RuntimeError(f"Configuration file can not be addressed to a detector class")
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
            # pilot.move_to_tcp_pose(SOCKET_POSE_ESTIMATION_CFG_X)

        # Search for ArUco marker
        found_socket = False
        # Use time out to exit loop
        time_out = 5.0
        _t_start = time.time()
        while time.time() - _t_start <= time_out and not found_socket:
            time.sleep(1.0)
            found, pose_cam2socket = dtt.find_pose(render=True)
            if found:
                # Get transformation matrices
                T_plug2cam = pilot.robot.cam_mdl.T_flange2camera
                T_base2plug = pilot.robot.get_tcp_pose().transformation
                T_socket2socket_pre = X_SOCKET_2_SOCKET_PRE.transformation
                T_cam2socket = Pose().from_xyz_xyzw(*pose_cam2socket).transformation

                # Get searched transformations
                T_base2socket = T_base2plug @ T_plug2cam @ T_cam2socket
                print(T_base2socket.pose.xyz, T_base2socket.pose.axis_angle)
                # T_base2socket = Pose().from_xyz((0.908, 0.294, 0.477)).from_axis_angle((-0.006, 1.568, 0.001)).transformation
                T_base2socket_pre = T_base2socket @ T_socket2socket_pre
                found_socket = True
    
        if not found_socket:
            # Move back to home
            with pilot.position_control():
                pilot.move_home()
        else:
            with pilot.position_control():    
                # Move to socket with some safety distance
                pilot.move_to_tcp_pose(T_base2socket_pre.pose)
            time.sleep(1.0)
            ck.user.wait_for_command()
            with pilot.force_control():
                pilot.one_axis_tcp_force_mode(axis='z', force=20.0, time_out=4.0)
                pilot.plug_in_force_ramp(f_axis='z', f_start=75.0, f_end=125, duration=4.0)
                pilot.relax(2.0)
                # Check if robot is in target area
                T_socket2fpi = X_SOCKET_2_FPI.transformation
                T_base2fpi = T_base2socket @ T_socket2fpi
                xyz_base2fpi_base_est = np.reshape(T_base2fpi.tau, 3)
                xyz_base2fpi_base_meas = np.reshape(pilot.robot.get_tcp_pose().xyz, 3)
                error = np.sqrt(np.sum(np.square(xyz_base2fpi_base_est - xyz_base2fpi_base_meas)))
                if error > 0.01:
                    # Plug-in process not successful
                    # Try to plug out again
                    success = pilot.tcp_force_mode(
                        wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -150.0, 0.0, 0.0, 0.0]),
                        compliant_axes=[0, 0, 1, 0, 0, 0],
                        distance=0.05,
                        time_out=10.0)
                    time.sleep(1.0)
                else:
                    with pilot.motion_control():
                        # Move 30 mm in tcp x direction to open the plug lock
                        pose_tcp2target = Pose().from_xyz([0.030, 0.0, 0.0]) 
                        pose_base2tcp = pilot.robot.get_tcp_pose()
                        pose_base2target = pose_base2tcp * pose_tcp2target
                        pilot.move_to_tcp_pose(pose_base2target, time_out=5.0)
                        # Move -20 mm in tcp z direction to disconnect from plug
                        pose_tcp2target = Pose().from_xyz([0.0, 0.0, -0.020])
                        pose_base2tcp = pilot.robot.get_tcp_pose()
                        pose_base2target = pose_base2tcp * pose_tcp2target
                        pilot.move_to_tcp_pose(pose_base2target)
                    success = True
                if success:
                    # Move back to home
                    with pilot.position_control():
                        pilot.move_home()
                else:
                    LOGGER.error(f"Error while trying to disconnect. Plug might still be in the socket.\n"
                                f"Robot will stop moving and shut down...")
    # Stop camera stream
    cam.end()


# def connect_to_socket_with_sensing() -> None:
#     # Build AruCo detection setup()
#     realsense = ca.RealSenseCamera("tcp_cam_realsense")
#     realsense.load_coefficients()

#     pattern_layout = {
#         51: (0, 60),
#         52: (60, 0),
#         53: (0, -100),
#         54: (-60, 0),
#     }
#     aru_pattern = ca.ArucoPattern("DICT_4X4_100", 25, pattern_layout)
#     pose_detector = ca.PatternDetector(realsense, aru_pattern, display=True)

#     # Connect to pilot
#     with ur_pilot.connect() as pilot:
#         # Link to camera
#         pilot.robot.register_ee_cam(realsense)

#         with pilot.position_control():
#             # Start at home position
#             pilot.move_home()
#             # Move to camera estimation pose to have all marker in camera field of view
#             pilot.move_to_joint_pos(SOCKET_POSE_ESTIMATION_CFG_J)

#         # Search for ArUco marker
#         found_socket = False
#         # Use time out to exit loop
#         time_out = 5.0
#         _t_start = time.time()
#         while time.time() - _t_start <= time_out and not found_socket:
#             time.sleep(0.5)
#             _ret, _board_pose = pose_detector.find_pose()
#             if _ret:
#                 r_vec, t_vec = _board_pose[0], _board_pose[1]
#                 # Convert to body motion object
#                 R_Cam2Board, _ = cv.Rodrigues(r_vec)
#                 q_Cam2Board = Quaternion().from_matrix(R_Cam2Board)
#                 Cam_x_Cam2Board = Vector3d().from_xyz(t_vec)
#                 # Pose of the camera frame with respect to the ArUco board frame
#                 X_Cam2Board = Pose().from_pq(Cam_x_Cam2Board, q_Cam2Board)

#                 # Pose of the plug frame with respect to the robot base frame
#                 X_Base2Plug = pilot.robot.get_tcp_pose()

#                 # Get transformation matrices
#                 T_Plug2Cam = pilot.robot.cam_mdl.T_flange2camera
#                 T_Base2Plug = X_Base2Plug.transformation
#                 T_Cam2Board = X_Cam2Board.transformation
#                 T_Board2Socket = X_SOCKET_2_PATTERN.transformation.inverse()
#                 T_Socket2SocketPre = X_SOCKET_2_SOCKET_PRE.transformation
#                 # T_Socket2SocketIn = X_SOCKET_2_SOCKET_IN.transformation

#                 # Get searched transformations
#                 T_Plug2Board = T_Plug2Cam @ T_Cam2Board
#                 T_Plug2Socket = T_Plug2Board @ T_Board2Socket 
#                 T_Base2Socket = T_Base2Plug @ T_Plug2Socket
#                 # T_Base2SocketIn = T_Base2Socket @ T_Socket2SocketIn
#                 T_Base2SocketPre = T_Base2Socket @ T_Socket2SocketPre
#                 found_socket = True

#         if not found_socket:
#             # Move back to home
#             with pilot.position_control():
#                 pilot.move_home()
#         else:
#             with pilot.position_control():
#                 # Switch to sense tcp
#                 pilot.robot.set_tcp('tool_sense')
#                 # Move to socket with some safety distance
#                 pilot.move_to_tcp_pose(T_Base2SocketPre.pose)
#             time.sleep(1.0)
#             with pilot.force_control():
#                 _, enh_T_Base2Socket = pilot.sensing_depth(T_Base2Target=T_Base2Socket, time_out=5.0)
#                 # Move to a safe retreat pose
#                 X_tcp = pilot.robot.get_tcp_pose()
#                 task_frame = X_tcp.xyz + X_tcp.axis_angle
#                 success_retreat, _ = pilot.retreat(task_frame=task_frame, direction=[0, 0, -1, 0, 0, 0])
#             if not success_retreat:
#                 raise RuntimeError("Moving to retreat pose was not successful. Stop moving.")
            
#             with pilot.position_control():
#                 # Switch to normal tool tip tcp
#                 pilot.robot.set_tcp('tool_tip')
#                 # Move to enhanced socket pose with some safety distance
#                 enh_T_Base2SocketPre = enh_T_Base2Socket @ T_Socket2SocketPre
#                 pilot.move_to_tcp_pose(enh_T_Base2SocketPre.pose)

#             with pilot.force_control():
#                 pair_succeed = pilot.pair_to_socket(enh_T_Base2Socket)
#                 if pair_succeed:
#                     plug_in_succeed = pilot.plug_in_with_target(100.0, enh_T_Base2Socket)
#                     if plug_in_succeed:
#                         LOGGER.info("Plugging successful!")
#                 pilot.relax(1.0)
#                 time.sleep(3.0)
#                 # Try to plug out
#                 success = pilot.plug_out_force_mode(
#                     wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -150.0, 0.0, 0.0, 0.0]),
#                     compliant_axes=[0, 0, 1, 0, 0, 0],
#                     distance=0.05,
#                     time_out=10.0)
#                 time.sleep(1.0)

#             if success:
#                 # Move back to home
#                 with pilot.position_control():
#                     pilot.move_home()
#             else:
#                 LOGGER.error(f"Error while trying to disconnect. Plug might still be in the socket.\n"
#                              f"Robot will stop moving and shut down...")

#     # Clean up
#     pose_detector.destroy(with_cam=False)
#     realsense.destroy()


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
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    # Connect to socket
    if args.with_sensing:
        pass
        # connect_to_socket_with_sensing()
    else:
        connect_to_socket(args)


if __name__ == '__main__':
    main()
