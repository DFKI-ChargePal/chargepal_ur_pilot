import ur_pilot

_jp1 = [3.132, -1.371, 2.127, -0.746, 1.602, -1.565]
_jp2 = [3.269, -1.392, 1.901, -0.250, 1.657, -1.565]
_jp3 = [3.418, -1.282, 1.939, -0.598, 1.827, -1.565]
_jp4 = [3.362, -1.211, 2.001, -0.782, 1.798, -1.565]


def move_joint_path() -> None:

    # Connect to pilot/robot arm
    with ur_pilot.connect() as pilot:
        # Move home
        with pilot.position_control():
            pilot.move_home()
            blend_1 = 0.02
            blend_2 = 0.02
            blend_3 = 0.02
            blend_4 = 0.0
            velocity = 0.1
            acceleration = 0.1

            path_jp1 = [*_jp1, velocity, acceleration, blend_1]
            path_jp2 = [*_jp2, velocity, acceleration, blend_2]
            path_jp3 = [*_jp3, velocity, acceleration, blend_3]
            path_jp4 = [*_jp4, velocity, acceleration, blend_4]
            path = [path_jp1, path_jp2, path_jp3, path_jp4]
            pilot.robot.rtde_controller.moveJ(path)
            # pilot.robot.movej(_jp1)
            # pilot.robot.movej(_jp2)
            # pilot.robot.movej(_jp3)
            # pilot.robot.movej(_jp4)



if __name__ == "__main__":
    move_joint_path()
