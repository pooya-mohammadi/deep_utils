from typing import Union, List, Dict, Optional
import importlib

from .main_utils import import_module
from ..constants import Backends, DUMMY_PATH, LIB_NAME


def is_backend_available(backend: Union[Backends, List[Backends]]):
    """
    Check if the backend package is installed or not

    Args:
        backend: Package name or a list of package names

    Returns:
        Whether the package is available or not
    """

    def check(x):
        return importlib.util.find_spec(x) is not None  # noqa

    if not isinstance(backend, list):
        backend = [backend]

    return all([check(backend) for backend in backend])


def import_lazy_module(cls_name: str,
                       cls_main_dir: str,
                       import_structure: Optional[Dict[str, List[Union[str, type]]]] = None,
                       dummy_path: str = DUMMY_PATH,
                       ):
    """
    Imports modules and dummy modules
    :param cls_name:
    :param cls_main_dir:
    :param import_structure:
    :param dummy_path:
    :return:
    """
    klass = import_module(f"{LIB_NAME}.{dummy_path}", cls_name)
    klass._module = f"{LIB_NAME}.{cls_main_dir}"
    required_backends = klass._backend
    cls_main_dir = cls_main_dir.replace("/", ".").replace("\\", ".")
    if import_structure is None:
        import_structure = import_module(LIB_NAME, "_import_structure")
    if is_backend_available(required_backends):
        if cls_main_dir in import_structure:
            import_structure[cls_main_dir].append(cls_name)
        else:
            import_structure[cls_main_dir] = [cls_name]
    else:
        if dummy_path in import_structure:
            import_structure[dummy_path].append(klass)
        else:
            import_structure[dummy_path] = [klass]
