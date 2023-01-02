import importlib
import warnings
from collections.abc import Callable


def import_module(module_name: str, things_to_import):
    assert isinstance(module_name, str), "module_name is not string, make sure you have correctly defined it"
    try:
        new_module = importlib.import_module(module_name)
        return getattr(new_module, things_to_import)
    except ModuleNotFoundError as e:
        warnings.warn(
            f"\n{e}. If you don't use {things_to_import} ignore this message.",
            stacklevel=2,
        )
        return e


def list_utils(module_dict):
    def list_models():
        detection_models = ""
        for name, _ in module_dict.items():
            detection_models += f"{name}\n"
        return detection_models

    return list_models


def loader(module_dict, list_models) -> Callable:
    def module_loader(name, **kwargs):
        if name not in module_dict:
            raise Exception(
                f"{name} model is not supported. Supported models are:\n{list_models()}"
            )
        model = module_dict[name](**kwargs)
        return model

    return module_loader
