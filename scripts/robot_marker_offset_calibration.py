# libs
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import camera_kit as ck
import spatialmath as sm
from pathlib import Path
from config import config_data

# typing
from argparse import Namespace

LOGGER = logging.getLogger(__name__)

_T_fpi2socket = sm.SE3().Trans([0.0, 0.0, -0.034])


def calibration_procedure(opt: Namespace) -> None:
    """ Function to run through the marker offset calibration procedure.

    Args:
        opt: Script arguments
    """
    LOGGER.info(config_data)
    # Perception setup
    cam = ck.camera_factory.create(opt.camera_name, opt.logging_level)
    cam.load_coefficients(config_data.camera_info_dir.joinpath(opt.camera_name, 'calibration/coefficients.toml'))
    cam.render()
    dtt = pd.factory.create(config_data.detector_dir.joinpath(opt.detector_config_file))
    dtt.register_camera(cam)

    # Connect to arm
    with ur_pilot.connect(config_data.robot_dir) as pilot:
        pilot.register_ee_cam(cam)

        # Enable free drive mode
        with pilot.context.teach_in_control():
            LOGGER.info('Start teach in mode')
            LOGGER.info("   You can now move the arm to the fully-plugged-in socket pose")
            LOGGER.info("   Press key 'r' or 'R' to go to the next step")
            while not ck.user.resume():
                cam.render()
            LOGGER.info('Stop teach in mode\n')
        # Measure target pose
        time.sleep(0.5)
        T_base2fpi = pilot.get_pose('tool_tip')
        with pilot.context.teach_in_control():
            LOGGER.info('Start teach in mode')
            LOGGER.info("  You can now bring the arm into a pose where the marker can be observed")
            LOGGER.info("  Press key 'r' or 'R' to go to the next step")
            while not ck.user.resume():
                cam.render()
            LOGGER.info('Stop teach in mode\n')
        # Measure observation pose
        time.sleep(0.5)
        T_base2flange = pilot.get_pose('flange')
        found, T_cam2marker = dtt.find_pose(render=True)

        if found:
            LOGGER.info('Found marker')
            # Get pose from target to marker
            T_base2socket = T_base2fpi * _T_fpi2socket
            T_socket2base = T_base2socket.inv()
            T_flange2cam = pilot.cam_mdl.T_flange2camera

            T_flange2marker = T_flange2cam * T_cam2marker
            T_base2marker = T_base2flange * T_flange2marker
            T_socket2marker = T_socket2base * T_base2marker
            T_marker2socket = T_socket2marker.inv()

            LOGGER.debug(f"Base - Flange:   {ur_pilot.utils.se3_to_str(T_base2flange)}")
            LOGGER.debug(f"Base - Marker:   {ur_pilot.utils.se3_to_str(T_base2marker)}")
            LOGGER.debug(f"Base - Socket:   {ur_pilot.utils.se3_to_str(T_base2socket)}")
            LOGGER.debug(f"Flange - Cam:    {ur_pilot.utils.se3_to_str(T_flange2cam)}")
            LOGGER.debug(f"Flange - Marker: {ur_pilot.utils.se3_to_str(T_flange2marker)}")
            LOGGER.debug(f"Socket - Marker: {ur_pilot.utils.se3_to_str(T_socket2marker)}")
            LOGGER.debug(f"Cam - Marker:    {ur_pilot.utils.se3_to_str(T_cam2marker)}")

            dtt.adjust_offset(T_marker2socket)

            LOGGER.info(f"  Calculated transformation from marker to socket:")
            LOGGER.info(f"  {ur_pilot.utils.se3_to_str(T_marker2socket)}\n")
        else:
            LOGGER.info(f"Marker not found. Please check configuration and make sure marker is in camera fov")
        if cam is not None:
            cam.end()


if __name__ == '__main__':
    des = """ Marker offset calibration script """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('detector_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--camera_name', type=str, default='realsense_tcp_cam', help='Camera name')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    ur_pilot.utils.check_file_extension(Path(args.detector_config_file), '.yaml')
    calibration_procedure(args)
