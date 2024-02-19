""" Script to calibrate the socket pose with the robot """
import time
import logging
import argparse
import ur_pilot
import camera_kit as ck
import spatialmath as sm
from pathlib import Path
from dataclasses import asdict
from rigmopy import Pose, Vector6d


# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)
_tgt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('target')

_T_fpi2socket = sm.SE3().Tz(-0.034)

X_FPI_2_SOCKET = Pose().from_xyz(xyz=[0.0, 0.0, -0.034])


def run(opt: Namespace) -> None:
    """ Function to work through the tasks
    
    Args:
        opt: Script arguments
    """
    # Create camera for user interaction
    cam = ck.camera_factory.create("realsense_tcp_cam", opt.logging_level)

    # Connect to arm
    with ur_pilot.connect() as pilot:
        with pilot.context.teach_in_control():
            LOGGER.info('Start teach in mode')
            LOGGER.info("  You can now move the arm to the target pose")
            LOGGER.info("  Press key 'r' or 'R' to go to the next step")
            while not ck.user.resume():
                cam.render()
            LOGGER.info('Stop teach in mode\n')

        # Measure target pose:
        time.sleep(1.0)
        pose_base2fpi = pilot.robot.get_tcp_pose()
        LOGGER.debug(f"Base - Fully-plugged-in: {pose_base2fpi}")
        T_base2fpi = pose_base2fpi.transformation
        T_fpi2socket = X_FPI_2_SOCKET.transformation
        T_base2socket = T_base2fpi @ T_fpi2socket
        pose_base2socket = T_base2socket.pose
        xyz = [float(v) for v in pose_base2socket.xyz]
        xyzw = [float(v) for v in pose_base2socket.xyzw]
        pose_base2socket_dict = asdict(ur_pilot.config_models.TargetConfig(xyz=xyz, xyzw=xyzw))
        LOGGER.info(f"New measured pose:")
        LOGGER.info(f"  Base - Socket: {pose_base2socket}")
        _tgt_cfg_dir.mkdir(parents=True, exist_ok=True)
        dump_fp = _tgt_cfg_dir.joinpath(f"pose_base2{opt.target_frame}.yaml")
        ur_pilot.yaml_helpers.dump_yaml(pose_base2socket_dict, dump_fp)

        with pilot.context.force_control():
            # Try to plug out
            success = pilot.tcp_force_mode(
                wrench=Vector6d().from_xyzXYZ([0.0, 0.0, -150.0, 0.0, 0.0, 0.0]),
                compliant_axes=[0, 0, 1, 0, 0, 0],
                distance=0.05,
                time_out=10.0)
            time.sleep(1.0)
            if success:
                # Move back to home
                with pilot.context.position_control():
                    pilot.move_home()
            else:
                LOGGER.error(f"Error while trying to disconnect. Plug might still be in the socket.\n"
                             f"Robot will stop moving and shut down...")
    cam.end()


if __name__ == '__main__':
    des = """ Script to calibrate the ground truth socket pose """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('target_frame', type=str, help='Name of the target frame')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    run(args)
