This part of the project documentation focuses on a
**problem-oriented** approach. You'll tackle common
tasks that you might have, with the help of the code
provided in this project.

## How to Configure a new Robot
The configuration for the modules is read from different configuration files that can be located anywhere on your storage.
We recommend to create a configuration folder to have all files in one place. The structure could look like this:

    config/
    |
    ├── robot_arm/
    |
    ├── camera_info/
    |
    └─— detector/

### Robot arm
For just using the robot arm without camera you need a configuration folder with two files:

    config/
    |
    └─— robot_arm/
        ├─— ur_control.toml
        └─— ur_pilot.toml

It is important that the files names has the exact name `ur_control.toml` and `ur_pilot.toml` to be able to assign the 
configurations precisely to the corresponding class. You can find some examples with real values in the configuration 
repository. However, a minimal example could be:

```toml
# ur_control.toml
name = "my_ur_control"
description = "Configuration file for the ur_control package."
author = "Elia Jordan"
email = "elia.jordan@host.com"

[robot]
tool = "None"
home_radians = [1.234, 1.234, 1.234, 1.234, 1.234, 1.234]  # [rad]
ip_address = "123.123.11.22"
```

```toml
# ur_pilot.toml
name = "my_ur_pilot"
description = "Configuration file for the ur_pilot package."
author = "Elia Jordan"
email = "elia.jordan@host.com"

[robot]

[pilot.coupling]
mass = 0.99  # [kg]
com = [0.01, 0.02, 0.03]  # [m]
link_frame = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [m]
```

### Camera Info
In the folder `camera_info/` you can add the camera calibration coefficients created by the calibration process of the 
camera module. More details can be found in the [Camera Kit module description](https://github.com/DFKI-ChargePal/chargepal_camera_kit). In addition,
the folder is a good place to add the matrix values for the transformation between the camera frame and the robot flange frame.
How to create them can be found in the section [How to Perform Flange-Eye-Calibration](#how-to-perform-flange-eye-calibration)

    camera_info/
    |
    └─— camera_name/
        └─—calibration
            ├─— coefficients.toml
            └─— T_flange2cam.toml

### Detector

In the folder 'detector/' you can add description files to use the Computer Vision Pattern Detector (CVPD) package. Note that 
the name of the configuration files determines the type of the detector instances that will be created. More details can
be found in the CVPD package documentation [add_link](https://google.com). A minimal example could be:

    detector/
    ├─— aruco_marker_71.yaml
    └─— charuco_board.yaml

```yaml
# aruco_marker_71.yaml
marker_id: 71
marker_type: 'DICT_4X4_100'
marker_size: 100  # [mm]
invert_img: true
offset:
  xyz: [0.0, 0.0, 0.0]  # [m]
  xyzw: [0.0, 0.0, 0.0, 1.0]  # [Quaternion]
```

```yaml
# charuco_board.yaml
marker_size: 19
marker_type: 'DICT_4X4_100'
checker_size: 25
checker_grid_size: [10, 7]
```


## How to work with the UR-Pilot

Assume you have a configuration folder as described in the section 
[How to Configure a new Robot](#how-to-configure-a-new-robot), this section will show you how to set up a Pilot object 
to control your robot arm.

```python
# control_robot.py
import ur_pilot          # Import pilot api
import cvpd as pd        # Import detector module
import camera_kit as ck  # Import camera module

from pathlib import Path

""" Perception setup """
cam = ck.camera_factory.create(name="camera_name")  # The name defines the camera type
cam.load_coefficients(file_path='./config/camera_info/calibration/coefficients.toml')
# The file name defines the detector type
dtt = pd.factory.create("./config/detector/aruco_marker_71.yaml")
dtt.register_camera(cam)

""" Connect to pilot """
# There are two ways to connect to the Pilot
# i) By using the Pilot class
pilot = ur_pilot.Pilot(config_dir=Path("./config/robot_arm"))
pilot.connect()

pilot.register_ee_cam(cam)  # Register camera to load spatial transformation values
# ... do robot stuff
pilot.disconnect()

# ii) By using the context manager
with ur_pilot.connect(config_dir=Path("./config/robot_arm")) as pilot:
    pilot.register_ee_cam(cam)
    # ... do robot stuff

# We prefer the second approach since disconnecting is done automatically when exiting the context.

```


## How to Perform Flange-Eye-Calibration

Before the detector can output the results in the correct coordinate frame, the pose between camera frame and robot 
flange frame has to be found by calibration. Therefore, the `ur_pilot` module provides the class `FlangeEyeCalibration`.
The following steps demonstrate how a calibration can be done.

The calibration process is divided into three steps:

1. Recording sample poses using a calibration board. The board must be positioned so that it can be seen from every
 shot and must not move during the recording
2. Move to the recorded poses and log the robot state as well as the images
3. Calculate the transformation matrix and store it

### Recording sample poses

In the first step we record robot poses by hand using the teach-in mode of the pilot. Example code can be found under 
the folder `scripts/` in the file `robot_teach_in.py`. The following command will start the teach-in process and stores
the results under the path `./data/teach_in/calibration_positions.json`:

```shell
python scripts/robot_teach_in.py calibration_positions.json --data_dir ./data/teach_in --camera_name camera_name
```

### Log robot state and calculate transformation matrix

Step two and three will be performed in one program. To log the robot state, with all information the robot needs for 
calibration, the robot arm will move to the teach-in positions again and afterward perform the Hand-Eye calibration
calculation. The resulting transformation matrix is then stored in the camera calibration folder. 

```shell
python scripts/robot_flange_eye_calibration.py calibration_positions.json --data_dir ./data/teach_in --camera_name camera_name
```




Information about the Hand-Eye calibration can be found at OpenCV [Hand-Eye calibration](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b)
