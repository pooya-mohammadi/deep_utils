import importlib
import warnings


def import_module(module_name, things_to_import):
    try:
        new_module = importlib.import_module(module_name)
        return getattr(new_module, things_to_import)
    except ModuleNotFoundError as e:
        warnings.warn(f"\n{e}. If you don't use {things_to_import} ignore this message.", stacklevel=2)
