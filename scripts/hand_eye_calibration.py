# global
import time
import argparse
import cv2 as cv
import numpy as np
import rigmopy as rp
import chargepal_aruco as ca
import matplotlib.pyplot as plt
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
from pytransform3d.plot_utils import make_3d_axis



# local
from ur_pilot.core.robot import Robot
from scripts.record_state_sequence import state_sequence_reader


def record_calibration_imgs(_debug: bool) -> None:

    # Create AruCo setup
    cam = ca.RealSenseCamera("realsense")
    dp = ca.Display(camera=cam, name="Rec")
    dtr = ca.Detector(cam, aruco_type="DICT_4X4_100", aruco_size_mm=19, checker_grid_size=(10, 7), checker_size_mm=25)
    # Connect to arm
    ur10 = Robot()
    ur10.move_home()
    # Move to all states in sequence
    counter = 1
    stop_reading = False
    for joint_pos in state_sequence_reader():
        continue_reading = False
        n_debug_prints = 0
        while not continue_reading:
            ur10.move_j(joint_pos)
            img = cam.get_color_frame()
            _ret, r_vec, t_vec = dtr.estimate_pose_charuco_board(img)
            if _ret:
                dp.draw_frame_axes(img, r_vec, t_vec, length=0.05, thickness=3)
                # Build target to camera transformation
                R_cam2tgt, _ = cv.Rodrigues(r_vec)
                tau_cam2tgt = np.array(t_vec, dtype=np.float32).squeeze()
                # Transformation matrix of target in camera frame
                T_cam2tgt = rp.Transformation().from_R_tau(R_cam2tgt, tau_cam2tgt)
                # Build TCP to arm base transformation
                tcp_pose = ur10.get_tcp_pose()
                tcp_pos = rp.Position().from_xyz(tcp_pose[:3])
                tcp_ori = rp.Orientation().from_axis_angle(tcp_pose[3:])
                T_base2tgt = rp.Transformation().from_pose(rp.Pose(tcp_pos, tcp_ori))
                # Save calibration item
                file_name = f"hand_eye_{counter:02}"
                if _debug:
                    if n_debug_prints <= 0:
                        print(f"\n\nTransformation ChArUco-board to camera\n {T_cam2tgt}")
                        print(f"\nTransformation UR10-TCP to UR10-base\n {T_base2tgt}")
                        print(f"Debug mode! No recordings will be saved.")
                        n_debug_prints += 1
                else:
                    cam.hand_eye_calibration_record(file_name, T_base2tgt.T, T_cam2tgt.T)
                    counter += 1
            dp.show_img(img)
            event = dp.event()
            if event == ca.Event.QUIT:
                print("The recording process is terminated by the user.")
                continue_reading = True
                stop_reading = True
            if _debug:
                # Pause by user
                if event == ca.Event.CONTINUE:
                    continue_reading = True
            else:
                continue_reading = True
                time.sleep(0.1)

        if stop_reading:
            break

    # Move back to home
    ur10.move_home()
    # Clean up
    ur10.exit()
    dp.destroy()
    cam.destroy()


def run_hand_eye_calibration(_debug: bool) -> None:
    """
    Function to execute the hand eye calibration. Please make sure to run the recording step first.
    :return: None
    """
    camera = ca.RealSenseCamera("realsense", load_coefficients=True, launch=False, verbose=True)
    # Get transformation matrix of camera in the tcp frame
    T_tcp2cam = camera.calibrate_hand_eye()
    rp_tcp2cam = rp.Transformation().from_R_tau(R=T_tcp2cam[:3, :3], tau=T_tcp2cam[:3, 3])
    if _debug:
        # Change convention
        rp_cam2tcp = rp_tcp2cam
        pt_cam2tcp = pt.transform_from(R=rp_cam2tcp.R, p=rp_cam2tcp.tau)
        sensor_size = np.array([3855.0, 2919.0]) * 1.0e-6
        intrinsic_matrix = np.array([
            [0.005, 0,     sensor_size[0] / 2.0],
            [0,     0.005, sensor_size[1] / 2.0],
            [0,     0,                        1]
        ])
        virtual_image_distance = 0.5
        ax = make_3d_axis(ax_s=1, unit="m", n_ticks=15)
        pt.plot_transform(ax=ax)
        pt.plot_transform(ax=ax, A2B=pt_cam2tcp, s=0.2)
        pc.plot_camera(
            ax, cam2world=pt_cam2tcp, M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance
        )
        plt.tight_layout()
        plt.show()

    repr_tcp2cam = repr(T_tcp2cam)
    print(repr_tcp2cam)
    
    camera.destroy()


def main() -> None:
    """ Example script """
    parser = argparse.ArgumentParser(description='Hand-Eye calibration script.')
    parser.add_argument('-r', '--rec', action='store_true', help='Use this option to record new calibration images')
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
