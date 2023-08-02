# global
import time
import logging
import argparse
import cv2 as cv
import numpy as np
import chargepal_aruco as ca
from rigmopy import Transformation

# local
import ur_pilot
from robot_record_state_sequence import state_sequence_reader


LOGGER = logging.getLogger(__name__)

TEACH_IN_FILE_ = "hand_eye_calibration.json"


def record_calibration_imgs(_debug: bool) -> None:

    # Create AruCo setup
    cam = ca.RealSenseCamera("tcp_cam_realsense")
    cam.load_coefficients()
    aru_board = ca.CharucoBoard("DICT_4X4_100", 19, (10, 7), 25)
    calibration = ca.Calibration(cam)
    detector = ca.CharucoboardDetector(cam, aru_board, display=True)
    
    # Connect to arm
    with ur_pilot.connect() as pilot:

        with pilot.position_control():
            pilot.move_home()

            # Move to all states in sequence
            counter = 1
            stop_reading = False
            calibration.hand_eye_calibration_clear_directories()
            for joint_pos in state_sequence_reader(TEACH_IN_FILE_):
                n_debug_prints = 0
                pilot.move_to_joint_pos(joint_pos)
                _ret, board_pose = detector.find_board_pose()
                if _ret:
                    r_vec, t_vec = board_pose[0], board_pose[1]
                    # Build target to camera transformation
                    R_cam2tgt, _ = cv.Rodrigues(r_vec)
                    tau_cam2tgt = np.array(t_vec).squeeze()
                    # Transformation matrix of target in camera frame
                    T_cam2tgt = Transformation().from_rot_tau(R_cam2tgt, tau_cam2tgt)
                    # Build TCP to arm base transformation
                    tcp_pose = pilot.robot.get_tcp_pose()
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
            pilot.move_home()
    
    # Clean up
    detector.destroy(with_cam=True)


def run_hand_eye_calibration() -> None:
    """ Function to execute the hand eye calibration. Please make sure to run the recording step first.
    
    Return:
        None
    """
    camera = ca.RealSenseCamera("tcp_cam_realsense")
    camera.load_coefficients()
    calibration = ca.Calibration(camera)

    # Get transformation matrix of camera in the tcp frame
    np_T_tcp2cam = calibration.hand_eye_calibration_est_transformation()

    # Log results
    T_tcp2cam = Transformation().from_trans_matrix(np_T_tcp2cam)
    LOGGER.info(f"Result - TCP to camera transformation (T_tcp2cam):")
    X_tcp2cam = T_tcp2cam.pose
    LOGGER.info(f"Tau in [m]:              {X_tcp2cam.xyz}")
    LOGGER.info(f"Rotation in euler [deg]: {X_tcp2cam.to_euler_angle(degrees=True)}")

    # Save results
    calibration.hand_eye_calibration_save_transformation(camera, np_T_tcp2cam)


def main() -> None:
    """ Hand eye calibration script """
    parser = argparse.ArgumentParser(description='Hand-Eye calibration script.')
    parser.add_argument('--rec', action='store_true', help='Use this option to record new calibration images')
    parser.add_argument('--debug', action='store_true', help='Option to set global debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ca.set_logging_level(logging.DEBUG)
    else:
        ca.set_logging_level(logging.INFO)
    # Check if a new recording step is needed
    if args.rec:
        record_calibration_imgs(args.debug)
    # Run calibration
    run_hand_eye_calibration()


if __name__ == '__main__':
    main()
