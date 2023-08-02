from __future__ import annotations

# global
import os
import json
import logging
import argparse
from pathlib import Path
import chargepal_aruco as ca

# local
import ur_pilot

# typing
from typing import Generator, Sequence


LOGGER = logging.getLogger(__name__)

_T_IN_DIR = "data/teach_in/"
_T_IN_FILE = "hand_eye_calibration.json"


def record_state_sequence(file_name: str) -> None:
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
    file_path = os.path.join(_T_IN_DIR, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r') as fp:
            state_seq = json.load(fp)
            # Iterate through the sequence
            for joint_pos in state_seq:
                yield joint_pos
    else:
        LOGGER.warning(f"No file with path '{file_path}'")


if __name__ == '__main__':
    """ Script to record a sequence of robot states (joint positions) """
    parser = argparse.ArgumentParser(description="Record a sequence of robot states")
    parser.add_argument('file_name', type=str, help='.json file name')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    fn: str = args.file_name
    if not fn.endswith('.json'):
        raise ValueError(f"JSON file with extension .json is mandatory. Given file name: {fn}")
    if args.debug:
        ca.set_logging_level(logging.DEBUG)
    else:
        ca.set_logging_level(logging.INFO)
    # Start recording
    record_state_sequence(fn)
