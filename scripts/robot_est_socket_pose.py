""" Script to estimate socket pose while moving robot by hand """
import logging
import argparse
import ur_pilot
import cvpd as pd
import numpy as np
import camera_kit as ck
from pathlib import Path
from rigmopy import Pose


# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)
_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')


def run(opt: Namespace) -> None:
    """ Function to run through the demo procedure.

    Args:
        opt: Script arguments
    """
    # Create perception setup
    cam = ck.create("realsense_tcp_cam", opt.logging_level)
    cam.load_coefficients()
    cam.render()
    dtt = pd.ArucoMarkerDetector(_dtt_cfg_dir.joinpath(opt.marker_config_file))
    dtt.register_camera(cam)

    # Connect to arm
    with ur_pilot.connect() as pilot:
        pilot.robot.register_ee_cam(cam)
        with pilot.teach_in_control():
            while not ck.user.stop():
                found, pose_cam2socket = dtt.find_pose(render=True)
                if found:
                    # Get transformation matrices
                    T_plug2cam = pilot.robot.cam_mdl.T_flange2camera  # Correct
                    T_base2plug = pilot.robot.get_tcp_pose().transformation
                    T_cam2socket = Pose().from_xyz_wxyz(*pose_cam2socket).transformation

                    # Get searched transformations
                    T_base2socket = T_base2plug @ T_plug2cam @ T_cam2socket
                    # print(T_cam2socket.pose.xyz, T_cam2socket.pose.to_euler_angle(degrees=True))
                    print(T_base2socket.pose.xyz, T_base2socket.pose.to_euler_angle(degrees=True))


if __name__ == '__main__':
    des = """ Estimate socket pose script """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('marker_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    ur_pilot.utils.check_file_extension(Path(args.marker_config_file), '.yaml')
    run(args)
