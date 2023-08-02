from __future__ import annotations

# global
import time
import argparse
import cv2 as cv
import numpy as np
import chargepal_aruco as ca
from rigmopy import Vector3d, Vector6d, Pose, Quaternion

# local
import ur_pilot


# Fixed configurations
SOCKET_POSE_ESTIMATION_CFG_J = [3.148, -1.824, 2.096, -0.028, 1.590, -1.565]
c_pi_4 = np.cos(np.pi/4)  # cos of 45 deg
board_2 = 0.075 / 2  # half board size 
X_SOCKET_2_PATTERN = Pose().from_xyz_xyzw(xyz=[0.005, 0.0, 0.0], xyzw=[0.0, 0.0, -c_pi_4, c_pi_4])
X_SOCKET_2_SOCKET_PRE = Pose().from_xyz(xyz=[0.0, 0.0, -0.045 - 0.02])  # Retreat pose with respect to the socket
# X_SOCKET_2_SOCKET_IN = Pose().from_xyz(xyz=[0.0, 0.0, 0.05])


def connect_to_socket() -> None:
    # Build AruCo detection setup()
    realsense = ca.RealSenseCamera("tcp_cam_realsense")
    realsense.load_coefficients()

    pattern_layout = {
        51: (0, 60),
        52: (60, 0),
        53: (0, -100),
        54: (-60, 0),
    }
    aru_pattern = ca.ArucoPattern("DICT_4X4_100", 25, pattern_layout)
    pose_detector = ca.PatternDetector(realsense, aru_pattern, display=True)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.robot.register_ee_cam(realsense)
    
        with pilot.position_control():
            # Start at home position
            pilot.move_home()
            # Move to camera estimation pose to have all marker in camera field of view
            pilot.move_to_joint_pos(SOCKET_POSE_ESTIMATION_CFG_J)
            # pilot.move_to_tcp_pose(SOCKET_POSE_ESTIMATION_CFG_X)

        # Search for ArUco marker
        found_socket = False
        # Use time out to exit loop
        time_out = 5.0
        _t_start = time.time()
        while time.time() - _t_start <= time_out and not found_socket:
            time.sleep(1.0)
            img = realsense.get_color_frame()
            _ret, _board_pose = pose_detector.find_pose()
            if _ret:
                r_vec, t_vec = _board_pose[0], _board_pose[1]
                # Convert to body motion object
                R_Cam2Board, _ = cv.Rodrigues(r_vec)
                q_Cam2Board = Quaternion().from_matrix(R_Cam2Board)
                Cam_x_Cam2Board = Vector3d().from_xyz(t_vec)
                # Pose of the camera frame with respect to the ArUco board frame
                X_Cam2Board = Pose().from_pq(Cam_x_Cam2Board, q_Cam2Board)

                # Pose of the plug frame with respect to the robot base frame
                X_Base2Plug = pilot.robot.get_tcp_pose()

                # Get transformation matrices
                T_Plug2Cam = pilot.robot.cam_mdl.T_flange2camera
                T_Base2Plug = X_Base2Plug.transformation
                T_Cam2Board = X_Cam2Board.transformation
                T_Board2Socket = X_SOCKET_2_PATTERN.transformation.inverse()
                T_Socket2SocketPre = X_SOCKET_2_SOCKET_PRE.transformation
                # T_Socket2SocketIn = X_SOCKET_2_SOCKET_IN.transformation

                # Get searched transformations
                T_Plug2Board = T_Plug2Cam @ T_Cam2Board
                T_Plug2Socket = T_Plug2Board @ T_Board2Socket 
                T_Base2Socket = T_Base2Plug @ T_Plug2Socket
                # T_Base2SocketIn = T_Base2Socket @ T_Socket2SocketIn
                T_Base2SocketPre = T_Base2Socket @ T_Socket2SocketPre
                found_socket = True
    
        if not found_socket:
            # Move back to home
            with pilot.position_control():
                pilot.move_home()
        else:
            with pilot.position_control():    
                # Move to socket with some safety distance
                pilot.move_to_tcp_pose(T_Base2SocketPre.pose)
            time.sleep(1.0)
            with pilot.force_control():
                pilot.plug_in_force_mode(
                    wrench=Vector6d().from_xyzXYZ([0.0, 0.0, 20.0, 0.0, 0.0, 0.0]),
                    compliant_axes=[1, 1, 1, 0, 0, 1], 
                    time_out=10.0)
                pilot.relax(3.0)
                # Try to plug out
                success, _ = pilot.plug_out_force_mode(
                    wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -25.0, 0.0, 0.0, 0.0]),
                    compliant_axes=[0, 0, 1, 0, 0, 0],
                    distance=0.05,
                    time_out=10.0)
                time.sleep(1.0)
            if success:
                # Move back to home
                with pilot.position_control():
                    pilot.move_home()
            else:
                print(f"Error while trying to disconnect. Plug might still be in the socket.")
                print(f"Robot will stop moving and shut down...")
                
    # Clean up
    pose_detector.destroy(with_cam=False)
    realsense.destroy()


def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    args = parser.parse_args()
    connect_to_socket()


if __name__ == '__main__':
    main()
