from __future__ import annotations

# global
import os
import json
import logging
from pathlib import Path
import chargepal_aruco as ca

# local
from ur_pilot import URPilot

# typing
from typing import Generator, Sequence


LOGGER = logging.getLogger(__name__)

_T_IN_DIR = "data/teach_in/"
_T_IN_FILE = "hand_eye_calibration.json"


def record_state_sequence() -> None:
    # Use camera for user interaction
    cam = ca.RealSenseCamera('tcp_cam_realsense')
    cam.load_coefficients()
    display = ca.Display('Monitor')
    # Connect to robot
    ur10 = URPilot()
    ur10.move_home()

    # Prepare recording
    state_seq: list[Sequence[float]] = []
    data_path = Path(_T_IN_DIR)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path.joinpath(_T_IN_FILE)

    # Enable free drive mode
    ur10.teach_mode()
    LOGGER.info("Start teach in mode: ")
    LOGGER.info("You can now move the arm and record joint positions pressing 's' or 'S' ...")
    while True:
        img = cam.get_color_frame()
        display.show(img)
        if ca.EventObserver.state is ca.EventObserver.Type.SAVE:
            # Save current joint position configuration
            joint_pos = ur10.get_joint_pos()
            info_str = f"Save joint pos: " + " ".join(f"{q:.3f}" for q in joint_pos)
            LOGGER.info(info_str)
            state_seq.append(joint_pos)
        elif ca.EventObserver.state is ca.EventObserver.Type.QUIT:
            LOGGER.info("The recording process is terminated by the user.")
            break
    
    LOGGER.info(f"Save all configurations in {file_path}")
    with file_path.open('w') as fp:
        json.dump(state_seq, fp, indent=2)

    # Stop free drive mode
    ur10.stop_teach_mode()
    # Clean up
    ur10.exit()
    display.destroy()
    cam.destroy()



def state_sequence_reader() -> Generator[list[float], None, None]:

    file_path = os.path.join(_T_IN_DIR, _T_IN_FILE)

    state_seq: list[list[float]] = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as fp:
            state_seq = json.load(fp)
            # Iterate through the sequence
            for joint_pos in state_seq:
                yield joint_pos
    else:
        print(f"No file with path '{file_path}'")


if __name__ == '__main__':
    ca.set_logging_level(logging.INFO)
    record_state_sequence()