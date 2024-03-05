# libs
import argparse
import logging
import ur_pilot
import cvpd as pd
from config import data
import camera_kit as ck
import spatialmath as sm
from ur_pilot import FlangeEyeCalibration
from scripts.robot_teach_in import state_sequence_reader

# typing
from argparse import Namespace

LOGGER = logging.getLogger(__name__)


def calibration_procedure(opt: Namespace) -> None:
    """ Function to run through the marker offset calibration procedure.

    Args:
        opt: Script arguments
    """
    LOGGER.info(data)
    # Perception setup

    cam = ck.camera_factory.create(opt.camera_name, opt.logging_level)
    calib_dir = data.camera_info_dir.joinpath(opt.camera_name, 'calibration')
    cam.load_coefficients(calib_dir.joinpath('coefficients.toml'))
    cam.render()
    dtt = pd.factory.create(data.detector_dir.joinpath('flange_eye_calibration.yaml'))
    dtt.register_camera(cam)

    # Connect to arm
    with ur_pilot.connect(data.robot_dir) as pilot:

        with pilot.context.position_control():
            pilot.robot.move_home()

            # Move to all states in sequence
            n_tf = 1
            FlangeEyeCalibration.clear_directories(cam)
            file_path = ur_pilot.utils.get_pkg_path().joinpath(opt.data_dir).joinpath(opt.file_name)
            for joint_pos in state_sequence_reader(file_path):
                pilot.move_to_joint_pos(joint_pos)
                found, T_cam2board = dtt.find_pose(render=True)
                if found:
                    # Get robot pose
                    T_base2flange = pilot.get_pose('flange')
                    # Save calibration item
                    file_name = f"flange_eye_{n_tf:02}"
                    LOGGER.debug(f"\n\nTransformation camera to ChArUco-board\n {T_cam2board}"
                                 f"\nTransformation UR10-flange to UR10-base\n {T_base2flange}")
                    FlangeEyeCalibration.record_sample(
                        camera=cam,
                        file_prefix=file_name,
                        T_base2flange=T_base2flange.A,
                        T_cam2tgt=T_cam2board.A,
                        dir_path=calib_dir
                    )
                    n_tf += 1
                if ck.user.stop():
                    LOGGER.warning("The recording process is terminated by the user.")
                    _quit = True
                    break
            # Move back to home
            pilot.robot.move_home()

        if not _quit:
            # Run calculation for calibration
            # Get transformation matrix of camera in the flange frame
            np_T_flange2cam = FlangeEyeCalibration.est_transformation(cam, calib_dir)
            # Log results
            T_flange2cam = sm.SE3(np_T_flange2cam)
            LOGGER.info(f"Result - flange to camera transformation (T_flange2cam):")
            LOGGER.info(f"   {ur_pilot.utils.se3_to_str(T_flange2cam)}")
            # Save results
            FlangeEyeCalibration.save_transformation(cam, np_T_flange2cam, calib_dir)

        # Tidy up
        if cam is not None:
            cam.end()


if __name__ == '__main__':
    des = """" Robot flange to eye calibration script """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('--camera_name', type=str, default='realsense_tcp_cam', help='Camera name')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    calibration_procedure(args)
