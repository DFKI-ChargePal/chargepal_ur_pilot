""" Short demo to demonstrate motion mode. Goal is to keep the current pose """
import ur_pilot

def keep_pose(soft: bool, time_out: float) -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:

        if soft:
            Kp = [1.0, 1.0, 1.0, 0.01, 0.01, 0.01]
        else:
            Kp = [25.0, 25.0, 25.0, 1.0, 1.0, 1.0]
        
        with pilot.motion_control(Kp_6=Kp):
            init_pose = pilot.robot.get_tcp_pose()
            pilot.move_to_tcp_pose(init_pose, time_out=time_out)


if __name__ == '__main__':
    keep_pose(True, 30.0)
