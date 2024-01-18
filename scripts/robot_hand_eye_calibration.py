# global
import time
import logging
import ur_pilot
import argparse
import cv2 as cv
import cvpd as pd
import numpy as np
import camera_kit as ck
from pathlib import Path
from argparse import Namespace
from rigmopy import Transformation
from ur_pilot import HandEyeCalibration

# local
from scripts.robot_teach_in import state_sequence_reader

LOGGER = logging.getLogger(__name__)

_charuco_cfg = Path(__file__).absolute().parent.parent.joinpath('detector', 'charuco_calibration.yaml')


def record_calibration_imgs(opt: Namespace) -> None:

    # Create perception setup
    cam = ck.camera_factory.create("realsense_tcp_cam", opt.logging_level)
    cam.load_coefficients()
    cam.render()
    detector = pd.CharucoDetector(_charuco_cfg)
    detector.register_camera(cam)

    # Connect to arm
    with ur_pilot.connect() as pilot:

        with pilot.position_control():
            pilot.move_home()

            # Move to all states in sequence
            counter = 1
            stop_reading = False
            dbg_lvl = ur_pilot.logger.get_logging_level()
            HandEyeCalibration.clear_directories(cam)
            file_path = ur_pilot.utils.get_pkg_path().joinpath(opt.data_dir).joinpath(opt.file_name)
            for joint_pos in state_sequence_reader(file_path):
                pilot.move_to_joint_pos(joint_pos)
                _ret, board_pose = detector.find_pose(render=True)
                if _ret:
                    # Get OpenCV style transformation
                    board_pose_cv = ck.converter.pq_to_cv(board_pose)
                    r_vec, t_vec = board_pose_cv[0], board_pose_cv[1]
                    # Build camera to target transformation
                    R_cam2tgt, _ = cv.Rodrigues(r_vec)
                    tau_cam2tgt = np.array(t_vec).squeeze()
                    # Transformation matrix of target in camera frame
                    T_cam2tgt = Transformation().from_rot_tau(R_cam2tgt, tau_cam2tgt)
                    # Build TCP to arm base transformation
                    tcp_pose = pilot.robot.get_tcp_pose()
                    T_base2tcp = Transformation().from_pose(tcp_pose)
                    # Save calibration item
                    file_name = f"hand_eye_{counter:02}"
                    if dbg_lvl <= logging.DEBUG:
                        LOGGER.debug(f"\n\nTransformation ChArUco-board to camera\n {T_cam2tgt}"
                                     f"\nTransformation UR10-TCP to UR10-base\n {T_base2tcp}"
                                     f"Debug mode! No recordings will be saved.")
                    else:
                        HandEyeCalibration.record_sample(
                            camera=cam,
                            file_prefix=file_name,
                            T_base2tcp=T_base2tcp.trans_matrix,
                            T_cam2tgt=T_cam2tgt.trans_matrix
                        )
                        counter += 1
                if ck.user.stop():
                    LOGGER.warning("The recording process is terminated by the user.")
                    stop_reading = True
                if dbg_lvl <= logging.DEBUG:
                    # Pause by user
                    LOGGER.info(f"Hit any bottom to continue.")
                    ck.user.wait_for_command()
                else:
                    time.sleep(0.1)
                if stop_reading:
                    break
            # Move back to home
            pilot.move_home()
    # Clean up
    cam.end()


def run_hand_eye_calibration(opt: Namespace) -> None:
    """ Function to execute the hand eye calibration. Please make sure to run the recording step first. """
    cam = ck.camera_factory.create("realsense_tcp_cam", opt.logging_level)
    cam.load_coefficients()
    # Get transformation matrix of camera in the tcp frame
    np_T_tcp2cam = HandEyeCalibration.est_transformation(cam)
    # Log results
    T_tcp2cam = Transformation().from_trans_matrix(np_T_tcp2cam)
    LOGGER.info(f"Result - TCP to camera transformation (T_tcp2cam):")
    X_tcp2cam = T_tcp2cam.pose
    LOGGER.info(f"Tau in [m]:              {X_tcp2cam.xyz}")
    LOGGER.info(f"Rotation in euler [deg]: {X_tcp2cam.to_euler_angle(degrees=True)}")
    # Save results
    HandEyeCalibration.save_transformation(cam, np_T_tcp2cam)


def main() -> None:
    """ Hand eye calibration script """
    parser = argparse.ArgumentParser(description='Hand-Eye calibration script.')
    parser.add_argument('file_name', type=str, help='.json file name with state sequence')
    parser.add_argument('--data_dir', type=str, default='data/teach_in')
    parser.add_argument('--rec', action='store_true',
                        help='Use this option to record new calibration images')
    parser.add_argument('--debug', action='store_true', help='Option to set global debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    # Check if a new recording step is needed
    if args.rec:
        record_calibration_imgs(args)
    if not args.logging_level == logging.DEBUG:
        # Run calibration
        run_hand_eye_calibration(args)


if __name__ == '__main__':
    main()
