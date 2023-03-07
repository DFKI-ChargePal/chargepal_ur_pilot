from __future__ import annotations

# global
import time
import math
import argparse
import cv2 as cv
import numpy as np
import rigmopy as rp
import chargepal_aruco as ca
import pytransform3d.rotations as pr

# local
from ur_pilot.core.robot import Robot
import ur_pilot.core.msgs.request_msgs as req_msg
from ur_pilot.core.move_to import move_to_tcp_pose


# Constants
SOCKET_POSE_ESTIMATION_CFG_J = [2.928, -1.373, -2.004, -3.436, -1.757, -1.678]
SOCKET_POSE_ESTIMATION_CFG_X = rp.Pose().from_xyz([-0.717, 0.250, 0.392]).from_axis_angle([-2.218, -0.007, 2.223])

SOCKET_MARKER = {
    51: rp.Position().from_xyz([-0.06, 0.0,  0.0]),
    52: rp.Position().from_xyz([ 0.0, -0.06, 0.0]),
    53: rp.Position().from_xyz([ 0.1,  0.0,  0.0]),
    54: rp.Position().from_xyz([ 0.0,  0.06, 0.0]),
}


def connect_to_socket() -> None:
    # Build AruCo detection setup()
    realsense = ca.RealSenseCamera("realsense")
    img_display = ca.Display(camera=realsense, name="connect2socket")
    aruco_detector = ca.Detector(realsense, aruco_type="DICT_4X4_250", aruco_size_mm=25, checker_size_mm=32)
    # Use ArUco detector for a 3x3 ArUco board
    # aruco_detector = ca.Detector(realsense, 'DICT_4X4_100', aruco_size_mm=19, checker_grid_size=(3, 3), checker_size_mm=25)
    # Build arm
    ur10 = Robot()
    # Start at home position
    # ur10.move_home()
    # Move to camera estimation pose to have all marker in camera field of view
    move_to_tcp_pose(ur10, req_msg.TCPPoseRequest(SOCKET_POSE_ESTIMATION_CFG_X))

    poses_Plug2Socket: list[rp.Pose] = []
    poses_Plug2Marker: list[rp.Pose] = []

    # Search for socket marker
    for marker_id, Socket_x_Socket2Marker in SOCKET_MARKER.items():
        _t_start = time.time()
        found = False
        # Pose of marker frame with respect to the socket frame
        c_pi_4 = math.cos(math.pi/4)
        q_Socket2Marker = rp.Orientation().from_xyzw((c_pi_4, -c_pi_4, 0.0, 0.0))
        X_Socket2Marker = rp.Pose(pos=Socket_x_Socket2Marker, ori=q_Socket2Marker)
        T_Marker2Socket = rp.Transformation().from_pose(X_Socket2Marker.inverse())  # Seems to be correct (03-01-2023 10:00)
        # Use time out of 2 seconds
        time_out = 5.0
        while not found and time.time() - _t_start <= time_out:
            time.sleep(0.5)
            img = realsense.get_color_frame()
            _ret, r_vec, t_vec = aruco_detector.search_marker_pose(img, marker_id)
            if _ret:
                # Convert to body motion object
                R_Cam2Marker, _ = cv.Rodrigues(r_vec)
                q_Cam2Marker = rp.Orientation().from_rotation_matrix(R_Cam2Marker)
                Cam_x_Cam2Marker = rp.Position().from_xyz(t_vec)
                # Pose of the camera frame with respect to the marker frame
                X_Cam2Marker = rp.Pose(Cam_x_Cam2Marker, q_Cam2Marker)
                # Pose of the plug frame with respect to the robot base frame
                X_Base2Plug = ur10.get_tcp_pose()

                # Get transformation matrices
                T_Plug2Cam = ur10.T_Cam_Plug
                T_Base2Plug = rp.Transformation().from_pose(X_Base2Plug)
                T_Cam2Marker = rp.Transformation().from_pose(X_Cam2Marker)
            
                # Get searched transformations
                T_Plug2Marker = T_Plug2Cam @ T_Cam2Marker  # Seems to be correct (02-28-2023 18:33) 
                T_Plug2Socket = T_Plug2Marker @ T_Marker2Socket  # Seems to be correct but the angular error is very huge (03-01-2023 10:21)
                T_Base2Socket = T_Base2Plug @ T_Plug2Socket

                euler = pr.intrinsic_euler_xyz_from_active_matrix(T_Plug2Socket.R)
                euler_deg = [math.degrees(ang) for ang in euler]
                print(f"\nMarker: {marker_id}")
                print(f"Active intrinsic rotation: {euler_deg}")
                print(f"Position vector:           {T_Plug2Socket.tau}")

                poses_Plug2Socket.append(T_Plug2Socket.to_pose())
                poses_Plug2Marker.append(T_Plug2Marker.to_pose())

                found = True
                img = img_display.draw_frame_axes(img, r_vec, t_vec)

            # print(img.shape, type(img[0, 0, 0]))
            img_display.show_img(img)

    poses = poses_Plug2Marker

    # Calculate disparity between poses
    print("Disparity marker 51 to marker n")
    ori_54 = np.array(poses[0].wxyz)
    for i in range(1, len(poses)):
        ori_i = np.array(poses[i].wxyz)
        error_ang = np.arccos(np.clip((2 * (ori_54.dot(ori_i))**2 - 1), -1.0, 1.0))
        print(f"Error angle: {np.rad2deg(error_ang)}")

    # Move back to home
    # ur10.move_home()
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
