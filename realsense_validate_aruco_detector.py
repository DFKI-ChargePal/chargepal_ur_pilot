from __future__ import annotations
# global
import json
import time
import logging
import argparse
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
import chargepal_aruco as ca
from rigmopy import Quaternion, Vector3d, Pose

# local
import ur_pilot

# typing
from typing import Generator, Sequence


LOGGER = logging.getLogger(__name__)

_T_IN_DIR = "data/teach_in/"
_EVAL_DIR = "data/eval/"

c_pi_4 = np.cos(np.pi/4)  # cos of 45 deg
X_SOCKET_2_PATTERN = Pose().from_xyz_xyzw(xyz=[0.0, 0.0, 0.0], xyzw=[0.0, 0.0, -c_pi_4, c_pi_4])


def teach_in(file_name: str) -> None:
    # Use camera for user interaction
    cam = ca.RealSenseCamera('tcp_cam_realsense')
    cam.load_coefficients()
    display = ca.Display('Monitor')
    # Connect to pilot
    with ur_pilot.connect() as pilot:

        with pilot.position_control():
            pilot.move_home()

        # Prepare recording
        state_seq: list[Sequence[float]] = []
        data_path = Path(_T_IN_DIR)
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path.joinpath(file_name)

        # Enable free drive mode
        with pilot.teach_in_control():
            LOGGER.info("Start teach in mode: ")
            LOGGER.info("You can now move the arm and record joint positions pressing 's' or 'S' ...")
            while True:
                img = cam.get_color_frame()
                display.show(img)
                if ca.EventObserver.state is ca.EventObserver.Type.SAVE:
                    # Save current joint position configuration
                    joint_pos = pilot.robot.get_joint_pos()
                    info_str = f"Save joint pos: " + " ".join(f"{q:.3f}" for q in joint_pos)
                    LOGGER.info(info_str)
                    state_seq.append(joint_pos)
                elif ca.EventObserver.state is ca.EventObserver.Type.QUIT:
                    LOGGER.info("The recording process is terminated by the user.")
                    break
            LOGGER.info(f"Save all configurations in {file_path}")
            with file_path.open('w') as fp:
                json.dump(state_seq, fp, indent=2)
        # Clean up
        display.destroy()
        cam.destroy()


def state_sequence_reader(file_name: str) -> Generator[list[float], None, None]:
    """ Helper function to read a state sequence from a JSON file

    Args:
        file_name: JSON file name

    Returns:
        A generator that gives the next robot state
    """
    # file_path = os.path.join(_T_IN_DIR, file_name)
    file_path = Path(_T_IN_DIR, file_name)

    if file_path.exists():
        with file_path.open('r') as fp:
            state_seq = json.load(fp)
            # Iterate through the sequence
            for joint_pos in state_seq:
                yield joint_pos
    else:
        LOGGER.warning(f"No file with path '{file_path}'")


def evaluate(file_name: str) -> None:

    # Create AruCo setup
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    cam.load_coefficients()
    pattern_layout = {
        51: (0, 60),
        52: (60, 0),
        53: (0, -100),
        54: (-60, 0),
    }
    aru_pattern = ca.ArucoPattern("DICT_4X4_100", 25, pattern_layout)
    calibration = ca.Calibration(cam)
    pose_detector = ca.PatternDetector(cam, aru_pattern, display=True)

    X_ref: Pose | None = None
    X_log: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        with pilot.position_control():
            pilot.move_home()

            # Move to all states in sequence
            stop_reading = False
            LOGGER.info(f"Relative measurement errors:")
            for i, joint_pos in enumerate(state_sequence_reader(file_name)):
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
    data_frame = pd.DataFrame(X_log, columns=["Position [xyz]", "Orientation [wxyz]"])
    # Update extension
    fn = "".join(file_name.split(".")[:-1]) + '.csv'
    file_path = Path(_T_IN_DIR, fn)
    data_frame.to_csv(path_or_buf=file_path, index=False)
    # Clean up
    pose_detector.destroy(with_cam=True)


def main() -> None:
    """ Script to evaluate aruco marker detector. """
    parser = argparse.ArgumentParser(description='ArUco marker detector evaluation script.')
    parser.add_argument('file_name', type=str, help='.json file name to store teach-in configurations')
    parser.add_argument('--teach', action='store_true', help='Option to teach in new evaluation poses')
    parser.add_argument('--debug', action='store_true', help='Option to set global debug flag')
    # Parse input arguments
    args = parser.parse_args()
    fn: str = args.file_name
    if not fn.endswith('.json'):
        raise ValueError(f"JSON file with extension .json is mandatory. Given file name: {fn}")
    if args.debug:
        ca.set_logging_level(logging.DEBUG)
    else:
        ca.set_logging_level(logging.INFO)

    if args.teach:
        teach_in(fn)
    else:
        evaluate(fn)


if __name__ == '__main__':
    main()
