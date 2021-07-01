import importlib
import warnings


# def import_module(module_dict, module_name, things_to_import):
#     if type(things_to_import) is list:
#         for name in things_to_import:
#             import_single_module(module_dict, module_name, name)
#     elif type(things_to_import) is str:
#         import_single_module(module_dict, module_name, things_to_import)
#     else:
#         raise Exception('The imported type is not supported.')


# def import_module(module_dict, module_name, things_to_import):
#     try:
#         new_module = importlib.import_module(module_name)
#         module_dict[things_to_import] = getattr(new_module, things_to_import)
#     except ModuleNotFoundError as e:
#         warnings.warn(f"\n{e}. If you don't use {things_to_import} ignore this message.", stacklevel=2)

def import_module(module_name, things_to_import):
    try:
        new_module = importlib.import_module(module_name)
        # module_dict[things_to_import] = getattr(new_module, things_to_import)
        return getattr(new_module, things_to_import)
    except ModuleNotFoundError as e:
        warnings.warn(f"\n{e}. If you don't use {things_to_import} ignore this message.", stacklevel=2)
