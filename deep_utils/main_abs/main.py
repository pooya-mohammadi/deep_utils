import inspect
import os
from abc import ABC
from typing import Any, Dict, Union

from deep_utils.utils.lib_utils.main_utils import import_module
from deep_utils.utils.os_utils.os_path import split_all


class MainClass(ABC):
    def __init__(self, name, file_path, **kwargs):
        self.config = None
        self.model = None
        self.name = name
        self.download_variables: Union[tuple, None] = kwargs.get(
            "download_variables", None
        )
        self.load_config(file_path, **kwargs)
        self.load_model()

    def __repr__(self):
        message = f"{self.name} config attributes:"
        for name, val in inspect.getmembers(self.config):

            if not name.startswith("_"):
                if not inspect.ismethod(val):
                    message += f"\n{name} = {val}"
        return message

    def load_model(self):
        raise NotImplementedError("load_model is not implemented")

    def load_config(self, file_path, **kwargs):
        separated_files = os.path.join(
            "deep_utils", os.path.split(
                file_path)[0].split("deep_utils")[-1][1:]
        )
        file_ = ".".join(split_all(separated_files) + ["config"])
        config = import_module(file_, "Config")
        self.config = config()
        self.update_config(**kwargs)

    def __dict_update(self, out_dict: Dict[str, Any], in_dict: Dict[str, Any]):
        for arg, val in in_dict.items():
            if arg in out_dict:
                source = out_dict[arg]
                if isinstance(val, dict) and isinstance(source, dict):
                    target = self.__dict_update(source, val)
                    out_dict[arg] = target
                else:
                    out_dict[arg] = val
            else:
                out_dict[arg] = val

        return out_dict

    def update_config(self, **kwargs):
        for arg, val in kwargs.items():
            if hasattr(self.config, arg):
                if val is None:
                    continue
                source = getattr(self.config, arg)
                if isinstance(val, dict) and isinstance(source, dict):
                    target = self.__dict_update(source, val)
                    setattr(self.config, arg, target)
                else:
                    setattr(self.config, arg, val)
            else:
                raise ValueError(
                    f"[ERROR] the config file does not contain argument: {arg}"
                )
