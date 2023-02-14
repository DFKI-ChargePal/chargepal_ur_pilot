# global
import sys
import json

# typing
from typing import Any, Dict


_FILE_NAME_CFG_DUMP = ".cache/config_cache.json"

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
        "home_joint_config": (2.489, -1.315, -2.173, -3.421, -2.126, -1.930),
    }
}



cfg_s = {
    "top_down_task": top_down_task,
    "banana_wall_task": banana_wall_task,

}


def set_up_demo(cfg: Dict[str, Any]) -> None:
    with open(_FILE_NAME_CFG_DUMP, 'w') as fp:
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
