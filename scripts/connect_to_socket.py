from __future__ import annotations

# global
import time
import math
import argparse
import cv2 as cv
import numpy as np
import rigmopy as rp
import chargepal_aruco as ca

# local
from ur_pilot.core.robot import Robot
import ur_pilot.core.msgs.request_msgs as req_msg
from ur_pilot.core.move_to import move_to_tcp_pose
from ur_pilot.core.plug_in import plug_in
from ur_pilot.core.plug_out import plug_out


# Fixed configurations
SOCKET_POSE_ESTIMATION_CFG_J = [2.928, -1.373, -2.004, -3.436, -1.757, -1.678]
SOCKET_POSE_ESTIMATION_CFG_X = rp.Pose().from_xyz([-0.717, 0.250, 0.392]).from_axis_angle([-2.218, -0.007, 2.223])

c_pi_4 = np.cos(np.pi/4)  # cos of 45 deg
board_2 = 0.075 / 2  # half board size 
X_SOCKET_2_BOARD = rp.Pose().from_xyz_xyzw(xyz=[0.125 - board_2, board_2, 0.0], xyzw=[0.0, 0.0, -c_pi_4, c_pi_4])
X_SOCKET_2_SOCKET_PRE = rp.Pose().from_xyz(xyz=[0.0, 0.0, -0.01])  # Retreat pose with respect to the socket

# Request messages
PLUG_IN_REQ = req_msg.PlugInRequest(
    compliant_axes=[1, 1, 1, 0, 0, 1], 
    wrench=rp.Wrench().from_ft([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    )

PLUG_OUT_REQ = req_msg.PlugOutRequest(
    compliant_axes=[0, 0, 1, 0, 0, 0],
    wrench=rp.Wrench().from_ft([0.0, 0.0, -25.0, 0.0, 0.0, 0.0]),
    moving_distance=0.05,
    t_limit=10.0
    )


def connect_to_socket() -> None:
    # Build AruCo detection setup()
    realsense = ca.RealSenseCamera("realsense")
    img_display = ca.Display(camera=realsense, name="connect2socket")
    
    # Use ArUco detector for a 3x3 ArUco board
    board_detector = ca.Detector(realsense, 'DICT_4X4_100', aruco_size_mm=19, checker_grid_size=(3, 3), checker_size_mm=25)
    
    # Build arm
    ur10 = Robot()
    # Start at home position
    ur10.move_home()
    # Move to camera estimation pose to have all marker in camera field of view
    move_to_tcp_pose(ur10, req_msg.TCPPoseRequest(SOCKET_POSE_ESTIMATION_CFG_X))

    # Search for ArUco board
    found_board = False
    # Use time out to exit loop
    time_out = 5.0
    _t_start = time.time()
    while time.time() - _t_start <= time_out and not found_board:
        time.sleep(1.0)
        img = realsense.get_color_frame()
        _ret, r_vec, t_vec = board_detector.estimate_pose_charuco_board(img)
        if _ret:
            # Convert to body motion object
            R_Cam2Board, _ = cv.Rodrigues(r_vec)
            q_Cam2Board = rp.Orientation().from_rotation_matrix(R_Cam2Board)
            Cam_x_Cam2Board = rp.Position().from_xyz(t_vec)
            # Pose of the camera frame with respect to the ArUco board frame
            X_Cam2Board = rp.Pose(Cam_x_Cam2Board, q_Cam2Board)

            # Pose of the plug frame with respect to the robot base frame
            X_Base2Plug = ur10.get_tcp_pose()

            # Get transformation matrices
            T_Plug2Cam = ur10.T_Cam_Plug
            T_Base2Plug = rp.Transformation().from_pose(X_Base2Plug)
            T_Cam2Board = rp.Transformation().from_pose(X_Cam2Board)
            T_Board2Socket = rp.Transformation().from_pose(X_SOCKET_2_BOARD.inverse())
            T_Socket2SocketPre = rp.Transformation().from_pose(X_SOCKET_2_SOCKET_PRE)

            # Get searched transformations
            T_Plug2Board = T_Plug2Cam @ T_Cam2Board
            T_Plug2Socket = T_Plug2Board @ T_Board2Socket 
            T_Base2Socket = T_Base2Plug @ T_Plug2Socket
            T_Base2SocketPre = T_Base2Socket @ T_Socket2SocketPre

            found_board = True

            img_display.draw_frame_axes(img, r_vec, t_vec, length=0.025, thickness=2)
        img_display.show_img(img)
    
    if not found_board:
        # Move back to home
        ur10.move_home()
    else:
        # Move to socket with some safety distance
        move_to_tcp_pose(ur10, req_msg.TCPPoseRequest(T_Base2SocketPre.to_pose()))
        time.sleep(2.0)
        # Try to plug in
        plug_in(ur10, PLUG_IN_REQ, img_display)
        time.sleep(3.0)
        # Try to plug out
        plug_out_res = plug_out(ur10, PLUG_OUT_REQ)
        time.sleep(1.0)
        if plug_out_res.time_out:
            print(f"Error while trying to disconnect. Plug might still be in the socket.")
            print(f"Robot will stop moving and shut down...")
        else:
            # Move back to home
            ur10.move_home()
    
    # Clean up
    ur10.exit()
    img_display.destroy()
    realsense.destroy()
    

def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    args = parser.parse_args()

    connect_to_socket()


if __name__ == '__main__':
    main()
