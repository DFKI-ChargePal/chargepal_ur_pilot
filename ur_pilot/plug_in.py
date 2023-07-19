""" This file contains functions to connect the plug and the socket. """
# global
import time
import numpy as np
import chargepal_aruco as ca

# local
from ur_pilot.core import URPilot
from ur_pilot.msgs.result_msgs import PlugInResult
from ur_pilot.msgs.request_msgs import PlugInForceModeRequest, PlugInMotionModeRequest


def plug_in_fm(rob: URPilot, req: PlugInForceModeRequest) -> PlugInResult:
    """ Execution function using force mode to connect the aligned plug with the socket

    Args:
        rob: Robot object
        req: Request message

    Returns:
        Result message with the final TCP pose
    """
    # Set the force mode up
    rob.set_up_force_mode()
    # Wrench will be applied with respect to the current TCP pose
    X_tcp = rob.get_tcp_pose()
    task_frame = X_tcp.xyz + X_tcp.axis_angle
    t_ref = time.time()
    x_ref = np.array(X_tcp.xyz, dtype=np.float32)
    # Time out
    time_out = False
    t_start = time.time()
    while True:
        # Apply wrench
        rob.force_mode(
            task_frame=task_frame,
            selection_vector=req.compliant_axes,
            wrench=req.wrench.to_numpy().tolist()
            )
        t_now = time.time()
        if t_now - t_ref > 1.0:
            x_now = np.array(rob.get_tcp_pose().xyz, dtype=np.float32)
            # Break loop if the pose for 1 second dose not change.
            if np.allclose(x_ref, x_now, atol=0.001):
                break
            t_ref = t_now
            x_ref = x_now
        if t_now - t_start > req.t_limit:
            time_out = True
            break
        if ca.EventObserver.state is ca.EventObserver.Type.QUIT:
            break

    # Exit force mode
    rob.stop_force_mode()
    # Gather result
    res_pose = rob.get_tcp_pose()
    return PlugInResult(res_pose, time_out)


def plug_in_mm(rob: URPilot, req: PlugInMotionModeRequest) -> PlugInResult:
    """ Execution function using motion mode to connect the aligned plug with the socket

    Args:
        rob: Robot object
        req: Request message

    Returns:
        Result message with the final TCP pose
    """
    # Set up motion mode
    rob.set_up_motion_mode(error_scale=req.error_scale, Kp_6=req.Kp, Kd_6=req.Kd)
    time_out = False
    t_start = time.time()
    while True:
        # Move linear to target
        rob.motion_mode(req.tcp_target)

        # Check error in plugging direction
        act_pose = rob.get_tcp_pose()
        error_p = req.tcp_target.p - act_pose.p
        # Rotate in tcp frame
        error_p_tcp = act_pose.q.apply(error_p, inverse=True)
        if abs(error_p_tcp.xyz[-1]) <= 0.005:
            break
        elif time.time() - t_start > req.t_limit:
            time_out = True
            break
    
    # Exit motion mode
    rob.stop_motion_mode()
    # Gather result
    res_pose = rob.get_tcp_pose()
    return PlugInResult(res_pose, time_out)
