# global
import os
import sys
import json

# typing
from typing import Any, Dict

_CFG_DUMP_FOLDER = ".cache"
_CFG_DUMP_FILE_NAME = "config_dump.json"

""" 
////////////////////////////////////
/////////                  /////////
/////////  CONFIGURATIONS  /////////
/////////                  ///////// 
////////////////////////////////////
"""

top_down_task = {
    "tool": "cylinder",
    "robot": {
        "home_joint_config": (3.14, -0.611, -2.18, -1.92, 1.57, -1.57),
    }
}

banana_wall_task = {
    "tool": "banana",
    "robot": {
        "home_joint_config": (3.034, -1.158, -2.260, -3.391, -1.665, -1.619),
        "_T_Cam_Plug": [
            [ 0.06819847,  0.92839549,  0.36528179, -0.17570487],
            [-0.9975047 ,  0.07015258,  0.00793621,  0.04608208],
            [-0.01825752, -0.36491154,  0.93086317, -0.23798058],
            [ 0.        ,  0.        ,  0.        ,  1.        ]],
    }
}

cfg_s = {
    "top_down_task": top_down_task,
    "banana_wall_task": banana_wall_task,

}


def set_up_demo(cfg: Dict[str, Any]) -> None:
    os.makedirs(_CFG_DUMP_FOLDER, exist_ok=True)
    fn = os.path.join(_CFG_DUMP_FOLDER, _CFG_DUMP_FILE_NAME)
    with open(fn, 'w') as fp:
        json.dump(cfg, fp=fp, indent=4, sort_keys=True)


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print(f"Please enter a configuration name as script argument.")
    else:
        cfg_name = sys.argv[1]
        if cfg_name in cfg_s:
            cfg_s[cfg_name]["_setup_id_name"] = cfg_name
            set_up_demo(cfg_s[cfg_name])
            print(f"Set up configuration file with name {cfg_name}...")
        else:
            print(f"Configuration with name {cfg_name} not found.\n"
                  f"Please create it or enter a valid file name")
