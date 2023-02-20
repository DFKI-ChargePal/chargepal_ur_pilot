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
        "T_CP": [
            [ 0.07806851,  0.92719907,  0.36634293, -0.17861411],
            [-0.99686325,  0.0773926 ,  0.01655629,  0.04555491],
            [-0.01300125, -0.36648634,  0.9303326 , -0.24013902],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ],
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
        else:
            print(f"Configuration with name {cfg_name} not found.\n"
                  f"Please create it or enter a valid file name")
