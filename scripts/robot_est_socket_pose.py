""" Script to estimate socket pose while moving robot by hand """
import logging
import argparse
import ur_pilot
import cvpd as pd
import camera_kit as ck
from pathlib import Path


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
                found, pose = dtt.find_pose(render=True)


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
