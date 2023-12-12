# global
import time
import logging
import argparse
import ur_pilot
import cvpd as pd
import camera_kit as ck
from pathlib import Path
from rigmopy import Pose

# typing
from argparse import Namespace


LOGGER = logging.getLogger(__name__)
_dtt_cfg_dir = Path(__file__).absolute().parent.parent.joinpath('detector')


def calibration_procedure(opt: Namespace) -> None:
    """ Function to run through the marker offset calibration procedure.

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
        if opt.target is not None:
            with pilot.position_control():
                pilot.move_to_tcp_pose(Pose().from_xyz(opt.target[:3]).from_axis_angle(opt.target[3:]))
        elif not opt.start_at_target:
            with pilot.teach_in_control():
                LOGGER.info('Start teach in mode')
                LOGGER.info("   You can now move the arm to the target pose")
                LOGGER.info("   Press key 'r' or 'R' to go to the next step")
                while not ck.user.resume():
                    cam.render()
                LOGGER.info('Stop teach in mode\n')
        # Measure target pose
        time.sleep(0.5)
        pose_base2target = pilot.robot.get_tcp_pose()
        if opt.observation is not None:
            with pilot.position_control():
                pilot.move_to_tcp_pose(Pose().from_xyz(opt.observation[:3]).from_axis_angle(opt.observation[3:]))
        else:
            with pilot.teach_in_control():
                LOGGER.info('Start teach in mode')
                LOGGER.info("  You can now bring the arm into a pose where the marker can be observed")
                LOGGER.info("  Press key 'r' or 'R' to go to the next step")
                while not ck.user.resume():
                    cam.render()
                LOGGER.info('Stop teach in mode\n')
        # Measure observation pose
        time.sleep(0.5)
        pose_base2tcp = pilot.robot.get_tcp_pose()
        found, pose_marker = dtt.find_pose(render=True)
        pose_cam2marker = Pose().from_xyz_wxyz(*pose_marker)

    if found:
        LOGGER.info('Found marker')
        # Get pose from target to marker
        T_base2tcp = pose_base2tcp.transformation
        T_cam2marker = pose_cam2marker.transformation
        T_base2target = pose_base2target.transformation
        T_target2base = T_base2target.inverse()
        T_tcp2cam = pilot.robot.cam_mdl.T_flange2camera
        
        T_tcp2marker = T_tcp2cam @ T_cam2marker
        T_base2marker = T_base2tcp @ T_tcp2marker
        T_target2marker = T_target2base @ T_base2marker
        LOGGER.debug(f"TCP - Cam: {T_tcp2cam}")
        LOGGER.debug(f"Base - TCP: {T_base2tcp}")
        LOGGER.debug(f"Base - Target: {T_base2target}")
        LOGGER.debug(f"Cam - Marker: {T_cam2marker}")
        LOGGER.debug(f"TCP - Marker: {T_tcp2marker}")
        LOGGER.debug(f"Base - Marker: {T_base2marker}")
        LOGGER.debug(f"Target - Marker: {T_target2marker}")
        # Convert to pose
        pose_target2marker = Pose().from_transformation(T_target2marker)
        xyz = [float(v) for v in pose_target2marker.xyz]
        xyzw = [float(v) for v in pose_target2marker.xyzw]
        # Save new offset position
        dtt.marker.adjust_configuration(xyz, xyzw)
        LOGGER.info(f"  Calculated transformation from target to marker:")
        LOGGER.info(f"  {pose_target2marker}\n")
    else:
        LOGGER.info(f"Marker not found. Please check configuration and make sure marker is in camera fov")
    time.sleep(2.0)
    cam.end()


if __name__ == '__main__':
    des = """ Marker offset calibration script """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('marker_config_file', type=str, 
                        help='Description and configuration of the used marker as .yaml file')
    parser.add_argument('--target', nargs='+', type=float, 
                        help="Target pose in format position [x y z] and axis-angle representation [Rx Ry Rz]")
    parser.add_argument('--observation', nargs='+', type=float, 
                        help="Observation pose in format position [x y z] and axis-angle representation [Rx Ry Rz]")
    parser.add_argument('--start_at_target', action='store_true', help='Optional if robot already at target position.')
    parser.add_argument('--debug', action='store_true', help='Option to set global logger level')
    # Parse input arguments
    args = parser.parse_args()
    if args.debug:
        args.logging_level = logging.DEBUG
    else:
        args.logging_level = logging.INFO
    if args.target:
        if len(args.target) != 6:
            raise ValueError(f"Not a valid target pose {args.target}. Need exact 6 float values")
    if args.observation:
        if len(args.observation) != 6:
            raise ValueError(f"Not a valid observation pose {args.observation}. Need exact 6 float values")
    ur_pilot.utils.check_file_extension(Path(args.marker_config_file), '.yaml')
    calibration_procedure(args)
