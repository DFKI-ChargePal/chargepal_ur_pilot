""" This file contains functions to disconnect the plug from the socket. """
# global
import time
import numpy as np

# local
from ur_pilot.core import URPilot
from ur_pilot.msgs.result_msgs import PlugOutResult
from ur_pilot.msgs.request_msgs import PlugOutRequest



def plug_out(rob: URPilot, req: PlugOutRequest) -> PlugOutResult:
    """
    Execution function to disconnect the inserted plug from the socket.
    :param rob: Robot object
    :param req: Request message
    :return: Request message with the new TCP pose
    """
    # Set the force mode up
    rob.set_up_force_mode()
    # Wrench will be applied with respect to the current TCP pose
    X_tcp = rob.get_tcp_pose()
    task_frame = X_tcp.xyz + X_tcp.axis_angle
    time_out = False
    t_start = time.time()
    x_ref = np.array(X_tcp.xyz, dtype=np.float32)
    while True:
        # Apply wrench
        rob.force_mode(
            task_frame=task_frame,
            selection_vector=req.compliant_axes,
            wrench=req.wrench.to_numpy().tolist()
            )
        # Get state information
        x_now = np.array(rob.get_tcp_pose().xyz, dtype=np.float32)
        l2_norm_dist = np.linalg.norm(x_now - x_ref)
        t_now = time.time()
        # Break if moving distance or the time limit is reached
        if l2_norm_dist >= req.moving_distance:
            break
        elif t_now - t_start > req.t_limit:
            time_out = True
            break

    # Exit force mode
    rob.stop_force_mode()
    # Gather result
    res_pose = rob.get_tcp_pose()
    return PlugOutResult(res_pose, time_out)
