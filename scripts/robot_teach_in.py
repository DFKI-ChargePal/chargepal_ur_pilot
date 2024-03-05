from __future__ import annotations
# global
import json
import logging
import argparse
import ur_pilot
import numpy as np
import camera_kit as ck
from config import data
from pathlib import Path
from argparse import Namespace

# typing
from typing import Generator, Sequence

LOGGER = logging.getLogger(__name__)


def teach_in_joint_sequence(opt: Namespace) -> None:
    file_path = ur_pilot.utils.get_pkg_path().joinpath(opt.data_dir).joinpath(opt.file_name)
    ur_pilot.utils.check_file_extension(file_path, 'json')
    if not ur_pilot.utils.check_file_path(file_path):
        LOGGER.info(f"Nothing is going to happen. Exit script.")
        return
    # Use a display for user interaction
    display = ck.Display('Monitor')
    if not opt.no_camera:
        cam = ck.camera_factory.create("realsense_tcp_cam")
        cam.load_coefficients()
    else:
        cam = None
    # Connect to pilot
    with ur_pilot.connect(data.robot_dir) as pilot:
        with pilot.context.position_control():
            pilot.robot.move_home()

        # Prepare recording
        state_seq: list[Sequence[float]] = []
        file_path.parent.mkdir(exist_ok=True)

        # Enable free drive mode
        with pilot.context.teach_in_control():
            LOGGER.info('Start teach in mode:')
            LOGGER.info("You can now move the arm and record joint positions pressing 's' or 'S' ...")
            while True:
                if cam is not None:
                    img = cam.get_color_frame()
                else:
                    img = (np.random.rand(480, 640, 3) * 255).astype(dtype=np.uint8)
                display.show(img)
                if ck.user.save():
                    # Save current joint position configuration
                    joint_pos = pilot.robot.joint_pos
                    info_str = f"Save joint pos: {ur_pilot.utils.vec_to_str(joint_pos, 3)}"
                    LOGGER.info(info_str)
                    state_seq.append(joint_pos.tolist())
                elif ck.user.stop():
                    LOGGER.info("The recording process is terminated by the user.")
                    break
            LOGGER.info(f"Save all configurations in {file_path}")
            with file_path.open('w') as fp:
                json.dump(state_seq, fp, indent=2)
        # Clean up
        display.destroy()
        if cam is not None:
            cam.end()


def state_sequence_reader(file_path: Path) -> Generator[list[float], None, None]:
    """ Helper function to read a state sequence from a JSON file

    Args:
        file_path: file path to JSON file

    Returns:
        A generator that gives the next robot state
    """
    ur_pilot.utils.check_file_extension(file_path, 'json')
    if file_path.exists():
        with open(file_path, 'r') as fp:
            state_seq = json.load(fp)
            # Iterate through the sequence
            for joint_pos in state_seq:
                yield joint_pos
    else:
        LOGGER.info(f"File not exists. Nothing is going to happen. Exit")


if __name__ == '__main__':
    """ Script to teach-in a sequence of robot states (joint positions) """
    parser = argparse.ArgumentParser(description="Record a sequence of robot states")
    parser.add_argument('file_name', type=str, help='.json file name')
    parser.add_argument('--data_dir', type=str, default='data/teach_in')
    parser.add_argument('--no_camera', action='store_true', help='Do not use end-effector camera')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        ur_pilot.logger.set_logging_level(logging.DEBUG)
    else:
        ur_pilot.logger.set_logging_level(logging.INFO)
    teach_in_joint_sequence(args)
