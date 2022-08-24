from pathlib import Path
from typing import Union

from deep_utils.utils.logging_utils import log_print


def load_yaml(yaml_path: Union[Path, str], logger=None, verbose=1) -> dict:
    """
    Load a yaml file to a dictionary object
    Args:
        yaml_path:
        logger:
        verbose:

    Returns:

    """
    import yaml

    with open(yaml_path, mode='r') as f:
        data_map = yaml.safe_load(f)
    log_print(logger, f"Successfully loaded {yaml_path}", verbose=verbose)
    return data_map


def dump_yaml(
    yaml_dict: dict, yaml_path: Union[Path, str], logger=None, verbose=1
) -> None:
    """
    Dumps a dict to yaml file
    Args:
        yaml_dict:
        yaml_path:
        logger:
        verbose:

    Returns:

    """
    import yaml

    with open(yaml_path, "w") as outfile:
        yaml.dump(yaml_dict, outfile, default_flow_style=False)
    log_print(
        logger, f"Successfully converted input dict to {yaml_path}", verbose=verbose
    )


def yaml_post_process(yaml_dict: dict):
    res = dict()
    for k, v in yaml_dict.items():
        if isinstance(v, dict):
            res[k] = yaml_post_process(v)
        elif v in ["none", "None"]:
            res[k] = None
        else:
            res[k] = v
    return res
