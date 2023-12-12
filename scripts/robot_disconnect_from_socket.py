from __future__ import annotations

# global
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import camera_kit as ck
from pathlib import Path
from time import perf_counter as _t_now
from rigmopy import Pose

# typing
from argparse import Namespace

LOGGER = logging.getLogger(__name__)

_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')

_marker_pose_estimation_cfg_joints = [3.409, -1.578, 2.062, -0.464, 1.809, -1.565]

_pose_socket2hook = Pose().from_xyz([0.0, 0.0, -0.097])



def disconnect_from_socket(opt: Namespace) -> None:
    """ Function to go through the disconnection procedure

    Args:
        opt: Script arguments
    """
    # Perception setup
    cam = ck.create('realsense_tcp_cam', logger_level=opt.logger_level)
    cam.load_coefficients()
    cam.render()
    dtt = pd.ArucoMarkerDetector(_dtt_cfg_dir.joinpath(opt.marker_config_file))
    dtt.register_camera(cam)

    # Connect to pilot
    with ur_pilot.connect() as pilot:
        # Link to camera
        pilot.robot.register_ee_cam(cam)
        with pilot.position_control():
            # Start at home position
            pilot.move_home()
            # Move to camera estimation pose to have all marker in camera field of view
            pilot.move_to_joint_pos(_marker_pose_estimation_cfg_joints)

        # Search for ArUco marker
        found = False
        # Use time out to exit loop
        _t_out = 5.0
        _t_start = _t_now()
        while _t_now() - _t_start <= _t_out and not found:
            # Give the robot time to stop
            time.sleep(1.0)
            found, pose_cam2socket = dtt.find_pose(render=True)
            if found:
                # Search for transformation from base to plug hook
                # Get transformation matrices
                break


def main() -> None:
    """ Main function to start process. """
    # Input parsing
    parser = argparse.ArgumentParser(description='Script to connect the plug with the socket.')
    parser.add_argument('marker_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--with_sensing', action='store_true', 
                        help='Option to use active end-effector sensing for better depth estimation')
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    args = parser.parse_args()
    ur_pilot.utils.check_file_extension(Path(args.marker_config_file), '.yaml')
    if args.debug:
        args.logger_level = logging.DEBUG
    else:
        args.logger_level = logging.INFO
    
    disconnect_from_socket(args)


if __name__ == '__main__':
    main()
