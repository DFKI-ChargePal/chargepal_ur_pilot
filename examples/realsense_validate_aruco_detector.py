from __future__ import annotations
# global
import json
import time
import logging
import ur_pilot
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import chargepal_aruco as ca
from argparse import Namespace
from rigmopy import Quaternion, Vector3d, Pose

# local
from robot_teach_in import teach_in_joint_sequence, state_sequence_reader


LOGGER = logging.getLogger(__name__)


DETECTORS = {
    'aruco_pattern': {
        'detector': ca.PatternDetector,
        'geo_pattern': ca.ArucoPattern(
            "DICT_4X4_100", 25,
            {
                51: (0, 60),
                52: (60, 0),
                53: (0, -100),
                54: (-60, 0),
            }),
    },
    'cp_logo': {
        'detector': ca.FeatureDetector,
        'geo_pattern': ca.ImageTemplate(
            Path('./examples/object_templates/cpLogoLong.png'),
            (154, 50), (0, 100)
        )
    },
    'qr_code': {
        'detector': ca.QRCodeDetector,
        'geo_pattern': ca.QRCode(
            'https://www.dfki.de/en/web/research/research-departments/plan-based-robot-control',
            40, (0, 75)
        )
    },
    'qr_code_pattern': {
        'detector': ca.QRCodePatternDetector,
        'geo_pattern': None,
    },
}
c_pi_4 = np.cos(np.pi/4)  # cos of 45 deg
X_SOCKET_2_PATTERN = Pose().from_xyz_xyzw(xyz=[0.0, 0.0, 0.0], xyzw=[0.0, 0.0, -c_pi_4, c_pi_4])


def evaluate(opt: Namespace) -> None:

    # Build data file paths
    fp_teach = ur_pilot.utils.get_pkg_path().joinpath(opt.dir_teach).joinpath(opt.file_name_teach)
    ur_pilot.utils.check_file_extension(fp_teach, 'json')
    fp_eval = ur_pilot.utils.get_pkg_path().joinpath(opt.dir_eval).joinpath(opt.file_name_eval)
    ur_pilot.utils.check_file_extension(fp_eval, '.csv')
    # Update evaluation file name
    file_name = fp_eval.name.split('.')[0] + '_' + opt.detector_type + '_detector' + '.csv'
    fp_eval = fp_eval.parent.joinpath(file_name)

    # Create AruCo setup
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    cam.load_coefficients()

    pose_detector = DETECTORS[opt.detector_type]['detector'](  # type: ignore
        cam, DETECTORS[opt.detector_type]['geo_pattern'], display=True)  # type: ignore

    X_ref: Pose | None = None
    X_log: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.robot.register_ee_cam(cam)

        with pilot.position_control():
            pilot.move_home()

            # Move to all states in sequence
            stop_reading = False
            LOGGER.info(f"Relative measurement errors:")
            for i, joint_pos in enumerate(state_sequence_reader(fp_teach)):
                pilot.move_to_joint_pos(joint_pos)
                found, est_pose = pose_detector.find_pose()
                if found:
                    r_vec, t_vec = est_pose
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

                    # Get searched transformations
                    T_Plug2Board = T_Plug2Cam @ T_Cam2Board
                    T_Plug2Socket = T_Plug2Board @ T_Board2Socket
                    T_Base2Socket = T_Base2Plug @ T_Plug2Socket
                    # Get pose
                    X_Base2Socket: Pose = T_Base2Socket.pose
                    # log measurements
                    X_log.append((X_Base2Socket.xyz, X_Base2Socket.wxyz))
                    # First measurement is stored as reference pose
                    if X_ref is None:
                        X_ref = X_Base2Socket
                    else:
                        # Evaluate against reference pose
                        X_new = X_Base2Socket
                        q_new = np.array(X_new.q.wxyz)
                        q_ref = np.array(X_ref.q.wxyz)
                        p_diff = (X_ref.p - X_new.p).xyz
                        pos_error = np.sqrt(np.sum(np.square(p_diff))) * 1000.0
                        ang_error = np.arccos(np.clip((2 * (q_ref.dot(q_new))**2 - 1), -1.0, 1.0)) * 1000.0
                        LOGGER.info(
                            f"Nr. {i} - Position error: {int(pos_error)} [mm] - Angular error: {int(ang_error)} [mRad]")
                else:
                    dummy_res = Pose()
                    X_log.append((dummy_res.xyz, dummy_res.wxyz))

                if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
                    print("The recording process is terminated by the user.")
                    stop_reading = True
                if logging.DEBUG >= logging.root.level:
                    # Pause by user
                    ca.EventObserver.wait_for_user()
                else:
                    time.sleep(0.1)
                if stop_reading:
                    break
            # Move back to home
            pilot.move_home()
    # Log data
    fp_eval.parent.mkdir(exist_ok=True)
    data_frame = pd.DataFrame(X_log, columns=["Position [xyz]", "Orientation [wxyz]"])
    data_frame.to_csv(path_or_buf=fp_eval, index=False)
    # Clean up
    pose_detector.destroy(with_cam=True)


def main() -> None:
    """ Script to evaluate aruco marker detector. """
    parser = argparse.ArgumentParser(prog='camera-validation', description='ArUco marker detector evaluation script.')
    subparsers = parser.add_subparsers(title='jobs', description='valid jobs', required=True)
    parser.add_argument('--debug',
                        action='store_true', help='Option to set global debug flag')
    # Teach-in options
    parser_teach = subparsers.add_parser('teach')
    parser_teach.add_argument('--file_name', default='detector_validation_joint_poses.json',
                              type=str, help='.json file name to store teach-in configurations')
    parser_teach.add_argument('--data_dir', default='data/teach_in',
                              type=str, help='Path of data directory')
    parser_teach.add_argument('--no_camera', action='store_true',
                              help='Do not use end-effector camera')
    parser_teach.set_defaults(func=teach_in_joint_sequence)
    # Evaluation options
    parser_eval = subparsers.add_parser('evaluate')
    parser_eval.add_argument('--file_name_teach', default='detector_validation_joint_poses.json',
                             type=str, help='.json file name to load teach-in configurations')
    parser_eval.add_argument('--file_name_eval', default='results.csv',
                             type=str, help='.csv file name to store evaluation results')
    parser_eval.add_argument('--dir_teach', default='data/teach_in',
                             type=str, help='Path of teach-in data directory')
    parser_eval.add_argument('--dir_eval', default='data/eval',
                             type=str, help='Path of evaluation records directory')
    parser_eval.add_argument('--detector_type', default='aruco_pattern',
                             choices=DETECTORS.keys())
    parser_eval.set_defaults(func=evaluate)
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.logger.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.logger.set_logging_level(logging.INFO)
    # Call the right job with corresponding arguments
    args.func(args)


if __name__ == '__main__':
    main()
