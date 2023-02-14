# global
import os
import sys
import json
import copy
import logging

# local
from set_up_pilot import _CFG_DUMP_FOLDER, _CFG_DUMP_FILE_NAME
from src.utils import query_yes_no

# typing
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


class ConfigServer:
    
    def __init__(self) -> None:
        # Check if there is a configuration file
        self.cfg_fp = os.path.join(_CFG_DUMP_FOLDER, _CFG_DUMP_FILE_NAME)
        if os.path.isfile(self.cfg_fp):
            # Load configurations
            self.cfg = self._load_cfg_from_file()
        else:
            self.cfg = {}
            LOGGER.warning(f"No configuration found under {_CFG_DUMP_FOLDER}!")
            continue_prog = query_yes_no("Continue with default values?", default="no")
            if not continue_prog:
                sys.exit()

    def _load_cfg_from_file(self) -> Dict[str, Any]:
        with open(self.cfg_fp, 'r') as fp:
            cfg: Dict[str, Any] = json.load(fp)
        return cfg

    def load(self, name: str, obj: object) -> None:
        cfg_name = name.split(".")[-1]
        sub_cfg: Dict[str, Any] = {} if cfg_name not in self.cfg else copy.deepcopy(self.cfg[cfg_name])
        for att_name, att_value in sub_cfg.items():
            setattr(obj, att_name, att_value)
