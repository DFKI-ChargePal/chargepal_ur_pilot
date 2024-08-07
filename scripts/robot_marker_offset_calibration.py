from __future__ import annotations

# libs
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import numpy as np
import camera_kit as ck
import spatialmath as sm
from pathlib import Path
from time import perf_counter
from config import config_data

# typing
from argparse import Namespace

LOGGER = logging.getLogger(__name__)

_t_now = perf_counter
_time_out = 10.0
_max_samples = 20


def calibration_procedure(opt: Namespace) -> None:
    """ Function to run through the marker offset calibration procedure.

    Args:
        opt: Script arguments
    """
    ur_pilot.base_logger.set_logging_level(opt.logging_level)
    LOGGER.info(config_data)
    # Perception setup
    cam = ck.camera_factory.create(config_data.camera_name, opt.logging_level)
    cam.load_coefficients(config_data.camera_cc)
    cam.render()
    dtt = pd.factory.create(config_data.detector_dir.joinpath(opt.detector_config_file))
    dtt.register_camera(cam)

    # Connect to arm
    with ur_pilot.connect(config_data.robot_dir) as pilot:
        pilot.register_ee_cam(cam, config_data.camera_dir)
        with pilot.plug_model.context(opt.plug_type):
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
            T_base2socket = pilot.get_pose(ur_pilot.EndEffectorFrames.PLUG_LIP)
            with pilot.context.teach_in_control():
                LOGGER.info('Start teach in mode')
                LOGGER.info("  You can now bring the arm into a pose where the marker can be observed")
                LOGGER.info("  Press key 'r' or 'R' to go to the next step")
                while not ck.user.resume():
                    cam.render()
                LOGGER.info('Stop teach in mode\n')
            # Measure observation pose
            search_rate = 0.5 * _time_out / _max_samples
            n_max, t_max = int(abs(_max_samples)), abs(_time_out)
            T_base2target_meas: sm.SE3 | None = None
            t_start = _t_now()
            for _ in range(n_max):
                time.sleep(search_rate)
                found, T_cam2target = dtt.find_pose(render=True)
                if found:
                    # Get transformation matrices
                    T_flange2cam = pilot.cam_mdl.T_flange2camera
                    T_base2marker_ = pilot.get_pose(ur_pilot.EndEffectorFrames.FLANGE)
                    # Get searched transformation
                    if T_base2target_meas is None:
                        T_base2target_meas = T_base2marker_ * T_flange2cam * T_cam2target
                    else:
                        T_base2target_meas.append(T_base2marker_ * T_flange2cam * T_cam2target)
                # Check for time boundary
                if _t_now() - t_start > t_max:
                    break
            if T_base2target_meas is None:
                valid_result, T_base2marker = False, sm.SE3()
            elif len(T_base2target_meas) == 1:
                valid_result, T_base2marker = True, T_base2target_meas
            else:
                q_avg = ur_pilot.utils.quatAvg(sm.UnitQuaternion(T_base2target_meas))
                t_avg = np.mean(T_base2target_meas.t, axis=0)
                T_base2marker = sm.SE3().Rt(R=q_avg.SO3(), t=t_avg)
                valid_result = True

            if valid_result:
                LOGGER.info('Found marker')
                # Get pose from target to marker
                T_socket2base = T_base2socket.inv()
                T_socket2marker = T_socket2base * T_base2marker
                T_marker2socket = T_socket2marker.inv()

                LOGGER.debug(f"Base - Flange:   {ur_pilot.utils.se3_to_str(T_base2marker)}")
                LOGGER.debug(f"Base - Marker:   {ur_pilot.utils.se3_to_str(T_base2marker)}")
                LOGGER.debug(f"Base - Socket:   {ur_pilot.utils.se3_to_str(T_base2socket)}")
                LOGGER.debug(f"Socket - Marker: {ur_pilot.utils.se3_to_str(T_socket2marker)}")

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
    parser.add_argument('plug_type', type=str, choices=['type2_male', 'type2_female', 'ccs_female'],
                        help='Plug type used for calibration')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    ur_pilot.utils.check_file_extension(Path(args.detector_config_file), '.yaml')
    calibration_procedure(args)
