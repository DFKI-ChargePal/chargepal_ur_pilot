from __future__ import annotations

# global
import os
import json
import chargepal_aruco as ca

# local
from ur_pilot.core.robot import Robot


_T_IN_DIR = "data/ur10e/teach_in/"
_T_IN_FILE = "hand_eye_calibration.json"


def record_state_sequence() -> None:
    # Use camera for user interaction
    cam = ca.RealSenseCamera("realsense")
    dp = ca.Display(camera=cam, name="Rec")
    # Connect to robot
    ur10 = Robot()
    ur10.move_home()

    # Prepare recording
    state_seq: list[list[float]] = []
    os.makedirs(_T_IN_DIR, exist_ok=True)
    file_path = os.path.join(_T_IN_DIR, _T_IN_FILE)
    # Enable free drive mode
    ur10.teach_mode()
    print("Start teach in mode: ")
    print("You can now move the arm and record joint positions pressing 's' or 'S' ...")
    while True:
        dp.show()
        event = dp.event()
        if event in [ca.Event.CONTINUE, ca.Event.SAVE]:
            # Save current joint position configuration
            joint_pos = ur10.get_joint_pos()
            print(f"Save joint pos:", " ".join(f"{q:.3f}" for q in joint_pos))
            state_seq.append(joint_pos)
        elif event == ca.Event.QUIT:
            print("The recording process is terminated by the user.")
            break
    
    print(f"Save all configurations in {file_path}")
    with open(file_path, 'w') as fp:
        json.dump(state_seq, fp, indent=2)

    # Stop free drive mode
    ur10.stop_teach_mode()
    # Clean up
    ur10.exit()
    dp.destroy()
    cam.destroy()


if __name__ == '__main__':
    record_state_sequence()
