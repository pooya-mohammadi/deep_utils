from pathlib import Path
from typing import Union

from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.utils.yaml_utils import dump_yaml, load_yaml, yaml_post_process


class KeyValStruct:
    """
    This is a simple class for hyperparameter tuning
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                self.__dict__[k] = KeyValStruct(**v)
            else:
                self.__dict__[k] = v


class YamlConfig(KeyValStruct):
    """
    A simple yaml config generator!
    """

    @classmethod
    def load_config(cls, config_path: str):
        config_dict = load_yaml(config_path)
        config_dict = yaml_post_process(config_dict)
        return cls(**config_dict)


def keyval_struct2dict(key_val_struct: KeyValStruct):
    """
    Converts a keyval structure to dictionary
    Args:
        key_val_struct:

    Returns:

    """
    output = dict()
    for k, v in vars(key_val_struct).items():
        if isinstance(v, KeyValStruct):
            output[k] = keyval_struct2dict(v)
        else:
            output[k] = v
    return output


def yaml_config2yaml_file(
    yaml_config: YamlConfig, yaml_path: Union[str, Path], logger=None, verbose=1
):
    """
    Dumps aa yaml config object to a yaml file
    Args:
        yaml_config:
        yaml_path:
        logger:
        verbose:

    Returns:

    """
    dict_obj = keyval_struct2dict(yaml_config)
    dump_yaml(dict_obj, yaml_path, logger=logger, verbose=verbose)
    log_print(
        logger, f"Successfully saved yaml_config in {yaml_path}", verbose=verbose)
