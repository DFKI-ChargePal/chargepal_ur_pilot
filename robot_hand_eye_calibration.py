# global
import time
import argparse
import cv2 as cv
import numpy as np
import chargepal_aruco as ca
from rigmopy import Transformation
# import matplotlib.pyplot as plt
# import pytransform3d.camera as pc
# import pytransform3d.rotations as pr
# import pytransform3d.transformations as pt
# from pytransform3d.plot_utils import make_3d_axis



# local
from ur_pilot.core import URPilot
from robot_record_state_sequence import state_sequence_reader


def record_calibration_imgs(_debug: bool) -> None:

    # Create AruCo setup
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    cam.load_coefficients()
    aru_board = ca.CharucoBoard("DICT_4X4_100", 19, (10, 7), 25)
    calibration = ca.Calibration(cam)
    detector = ca.CharucoboardDetector(cam, aru_board, display=True)
    
    # Connect to arm
    ur10 = URPilot()
    ur10.move_home()
    # Move to all states in sequence
    counter = 1
    stop_reading = False
    calibration.hand_eye_calibration_clear_directories()
    for joint_pos in state_sequence_reader():
        n_debug_prints = 0
        ur10.move_j(joint_pos)
        _ret, board_pose = detector.find_board_pose()
        if _ret:
            r_vec, t_vec = board_pose[0], board_pose[1]
            # Build target to camera transformation
            R_cam2tgt, _ = cv.Rodrigues(r_vec)
            tau_cam2tgt = np.array(t_vec).squeeze()
            # Transformation matrix of target in camera frame
            T_cam2tgt = Transformation().from_rot_tau(R_cam2tgt, tau_cam2tgt)
            # Build TCP to arm base transformation
            tcp_pose = ur10.get_tcp_pose()
            T_base2tcp = Transformation().from_pose(tcp_pose)
            # Save calibration item
            file_name = f"hand_eye_{counter:02}"
            if _debug:
                if n_debug_prints <= 0:
                    print(f"\n\nTransformation ChArUco-board to camera\n {T_cam2tgt}")
                    print(f"\nTransformation UR10-TCP to UR10-base\n {T_base2tcp}")
                    print(f"Debug mode! No recordings will be saved.")
                    n_debug_prints += 1
            else:
                calibration.hand_eye_calibration_record_sample(
                    file_prefix=file_name,
                    T_base2tcp=T_base2tcp.trans_matrix,
                    T_cam2tgt=T_cam2tgt.trans_matrix,
                )
                counter += 1
        if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
            print("The recording process is terminated by the user.")
            stop_reading = True
        if _debug:
            # Pause by user
            ca.EventObserver.wait_for_user()
        else:
            time.sleep(0.1)

        if stop_reading:
            break

    # Move back to home
    ur10.move_home()
    # Clean up
    ur10.exit()
    cam.destroy()


def run_hand_eye_calibration(_debug: bool) -> None:
    """
    Function to execute the hand eye calibration. Please make sure to run the recording step first.
    :return: None
    """

    camera = ca.RealSenseCamera("tcp_cam_realsense")
    camera.load_coefficients()
    calibration = ca.Calibration(camera)

    # Get transformation matrix of camera in the tcp frame
    T_tcp2cam = calibration.hand_eye_calibration_est_transformation()
    rp_tcp2cam = Transformation().from_rot_tau(rot_mat=T_tcp2cam[:3, :3], tau=T_tcp2cam[:3, 3])

    # if _debug:
    #     # Change convention
    #     rp_cam2tcp = rp_tcp2cam
    #     pt_cam2tcp = pt.transform_from(R=rp_cam2tcp.R, p=rp_cam2tcp.tau)
    #     sensor_size = np.array([3855.0, 2919.0]) * 1.0e-6
    #     intrinsic_matrix = np.array([
    #         [0.005, 0,     sensor_size[0] / 2.0],
    #         [0,     0.005, sensor_size[1] / 2.0],
    #         [0,     0,                        1]
    #     ])
    #     virtual_image_distance = 0.5
    #     ax = make_3d_axis(ax_s=1, unit="m", n_ticks=15)
    #     pt.plot_transform(ax=ax)
    #     pt.plot_transform(ax=ax, A2B=pt_cam2tcp, s=0.2)
    #     pc.plot_camera(
    #         ax, cam2world=pt_cam2tcp, M=intrinsic_matrix, sensor_size=sensor_size,
    #         virtual_image_distance=virtual_image_distance
    #     )
    #     plt.tight_layout()
    #     plt.show()

    print(f"\n\nResult - TCP to camera transformation matrix (T_Cam_Plug):\n")
    print(rp_tcp2cam, '\n')
    
    # if verbose:
    #     # euler = pr.intrinsic_euler_xyz_from_active_matrix(T_tcp2cam[:3, :3])
    #     euler = pr.intrinsic_euler_yzx_from_active_matrix(T_tcp2cam[:3, :3])
    #     euler_deg = tuple([math.degrees(ang) for ang in euler])

    #     print(f"Euler angle to active rotate plug frame into camera frame: {euler_deg}")
    #     print(f"Position of the camera frame in plug frame:                {rp_tcp2cam.to_pose().xyz}")
    #     # print(f"RPY_Cam_Plug: {rp_tcp2cam.to_pose().rpy_degrees}")

    camera.destroy()


def main() -> None:
    """ Example script """
    parser = argparse.ArgumentParser(description='Hand-Eye calibration script.')
    parser.add_argument('--rec', action='store_true', help='Use this option to record new calibration images')
    parser.add_argument('--debug', action='store_true', help='Option to set global debug flag')
    # Parse input arguments
    args = parser.parse_args()
    # Check if a new recording step is needed
    if args.rec:
        record_calibration_imgs(args.debug)
    # Run calibration
    run_hand_eye_calibration(args.debug)


if __name__ == '__main__':
    main()
