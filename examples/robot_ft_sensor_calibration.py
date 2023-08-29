from __future__ import annotations

# global
import time
import tomli
import shutil
import tomli_w
import logging
import argparse
from pathlib import Path
from tomlkit import document
from rigmopy import Vector3d, Vector6d

# local
import ur_pilot
from ur_pilot.end_effector.ft_calib import FTCalibration
from robot_record_state_sequence import record_state_sequence, state_sequence_reader


LOGGER = logging.getLogger(__name__)

CALIB_DIR_ = "data/ft_sensor/calibration"
CALIB_FILE_ = "parameters.toml"

MEAS_DIR_ = "data/ft_sensor/measurements"
MEAS_FILE_ = "measurement.toml"

TEACH_IN_FILE_ = "ft_sensor_calibration.json"


def teach_in() -> None:
    """ Function to record several robot states that are used to find calibration parameter afterward.

    Returns:
        None
    """
    LOGGER.info("Teach new robot sequence")
    record_state_sequence(TEACH_IN_FILE_)


def record_ft_measurements() -> None:
    """ Function to record multiple force torque measurements from several poses.

    Returns:
        None
    """
    LOGGER.info("Record FT-sensor measurements")
    LOGGER.info("ATTENTION: Robot is going to move!")
    # Create measurement directory or clear it.
    meas_dir = Path(MEAS_DIR_)
    if meas_dir.exists():
        shutil.rmtree(meas_dir)
    meas_dir.mkdir(parents=True, exist_ok=True)
    # Connect to robot
    with ur_pilot.connect() as pilot:
        with pilot.position_control():
            pilot.move_home()
            # Move to all states in sequence
            counter = 1
            for joint_pos in state_sequence_reader(TEACH_IN_FILE_):
                pilot.move_to_joint_pos(joint_pos)
                time.sleep(1.0)  # wait to have no more movements
                new_ft_meas_wrt_world = pilot.robot.average_ft_measurement(200)  # Get average measurement over 200 readings
                q_world2tcp = pilot.robot.q_world2arm * pilot.robot.get_tcp_pose().q
                g_world_wrt_world = Vector3d().from_xyz([0.0, 0.0, -9.80665])
                g_tcp_wrt_world = q_world2tcp.apply(g_world_wrt_world, inverse=True)

                file_name = f"{counter:02}_{MEAS_FILE_}"
                fp = meas_dir.joinpath(file_name)
                LOGGER.info(f"Record sample with:\nGravity: {g_tcp_wrt_world}\nFT: {new_ft_meas_wrt_world}")
                toml_doc = document()
                toml_doc.add("gravity", g_tcp_wrt_world.xyz)  # type: ignore
                toml_doc.add("ft_raw", new_ft_meas_wrt_world.xyzXYZ)  # type: ignore
                with fp.open(mode='wb') as f:
                    tomli_w.dump(toml_doc, f)
                counter += 1

            # Move back to home
            pilot.move_home()


def calibrate_ft_sensor() -> tuple[float, Vector3d, Vector6d]:
    """ Function to find ft-sensor calibration parameters

    Returns:
        Calibration parameters (mass, com, ft-bias)
    """
    LOGGER.info("Run FT-sensor calibration")
    ft_calib = FTCalibration()
    meas_dir = Path(MEAS_DIR_)
    mass = 0.0
    com = Vector3d()
    ft_bias = Vector6d()
    if not meas_dir.exists():
        LOGGER.warning(f"No directory found with path {meas_dir}. Return default parameters")
    else:
        for file_ in meas_dir.glob('*.toml'):
            with file_.open(mode='rb') as fp:
                f_in = tomli.load(fp)
                g = Vector3d().from_xyz(f_in["gravity"])
                ft_raw = Vector6d().from_xyzXYZ(f_in["ft_raw"])
                ft_calib.add_measurement(g, ft_raw)

        mass, com, ft_bias = ft_calib.get_calib()
    LOGGER.info(f"Calibration values:\nMass: {mass}\nCOM: {com}\nFT bias: {ft_bias}")
    return mass, com, ft_bias


if __name__ == '__main__':
    """ Script to calibrate force torque sensor. """
    parser = argparse.ArgumentParser(description="Calibrate force torque sensor.")
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    parser.add_argument('--teach_in', action='store_true', help='Option to teach a new robot state sequence')
    parser.add_argument('--record', action='store_true', help='Option to record sensor readings from state sequence')
    parser.add_argument('--calibrate', action='store_true',
                        help='Option to run the calibration step based on previous records.')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.set_logging_level(logging.INFO)
    # Run desired jobs
    if args.teach_in:
        teach_in()
    if args.record:
        record_ft_measurements()
    if args.calibrate:
        calibrate_ft_sensor()
