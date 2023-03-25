import importlib.util
import importlib.util
import os
from collections import OrderedDict
from itertools import chain
from types import ModuleType
from typing import Any

from deep_utils.utils.lib_utils.main_utils import import_module
from deep_utils.utils.utils.str_utils import color_str


def is_available(module_name: str, lib_name: str = None):
    lib_name = module_name if lib_name is None else lib_name
    import_error = (
            "{0} "
            + f"""
    requires {module_name} library but it was not found in your environment. You can install it with the following instructions:
    ```
    pip install {lib_name}
    ```
    In a notebook or a colab, you can install it by executing a cell with
    ```
    !pip install {lib_name}
    ```
    """
    )
    return module_name, (
        lambda: importlib.util.find_spec(module_name) is not None,
        import_error,
    )


_is_tf_available = importlib.util.find_spec("tensorflow") is not None
_is_torch_available = importlib.util.find_spec("torch") is not None
_is_cv2_available = importlib.util.find_spec("cv2") is not None
_is_torchvision_available = importlib.util.find_spec("torchvision") is not None
_is_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
_is_pyannote_audio_available = importlib.util.find_spec("pyannote") is not None
_is_transformers_available = importlib.util.find_spec("transformers") is not None


def is_transformers_available():
    return _is_transformers_available


def is_torch_available():
    return _is_torch_available


def is_torchvision_available():
    return _is_torchvision_available


def is_torchaudio_available():
    return _is_torchaudio_available


def is_tf_available():
    return _is_tf_available


def is_cv2_available():
    return _is_cv2_available


def is_pyannote_audio_available():
    return _is_pyannote_audio_available


CV2_IMPORT_ERROR = """
{0} requires OpenCV library but it was not found in your environment. You can install it with:
```
pip install opencv-python
# or
pip install deep-utils[cv]
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install opencv-python
or
!pip install deep-utils[cv]
```
"""

TENSORFLOW_IMPORT_ERROR = """
{0} requires TensorFlow library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of tensorflow.org:
```
pip install tensorflow
# or
pip install deep-utils[tf]
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install tensorflow
or
!pip install deep-utils[tf]
```
"""

PYTORCH_IMPORT_ERROR = """
{0} requires PyTorch library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torch
# or
pip install deep-utils[torch]
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torch
or
!pip install deep-utils[torch]
```
"""

TORCHVISION_IMPORT_ERROR = """
{0} requires Torchvision library but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torchvision
# or
pip install deep-utils[torchvision]
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torchvision
or
!pip install deep-utils[torchvision]
```
"""

TORCHAUDIO_IMPORT_ERROR = """
{0} requires Torchaudio library, but it was not found in your environment. You can install it with the following instruction or check out the main webpage of pytorch.org:
```
pip install torchaudio
# or
pip install deep-utils[torch]
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install torchaudio
or
!pip install deep-utils[torch]
```
"""

PYANNOTE_AUDIO_IMPORT_ERROR = """
{0} requires PYANNOTE_AUDIO library, but it was not found in your environment. You can install it with the following instruction or check out the main repository page of pyannote(https://github.com/pyannote/pyannote-audio):
```
pip install pyannote.audio
```
In a notebook or a colab, you can install it by executing a cell with
```
!pip install pyannote.audio
```
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("torchvision", (is_torchvision_available, TORCHVISION_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
        is_available("seaborn"),
        is_available("numpy"),
        is_available("albumentations"),
        is_available("sklearn"),
        is_available("PIL"),
        is_available("pyannote"),
        is_available("librosa"),
        is_available("transformers"),
        is_available("soundfile"),
        is_available("psutil"),
        is_available("yaml", "pyyaml"),
        is_available("ipython", "IPython"),
    ]
)


def requires_backends(obj, backends, module_name: str = None, cls_name: str = None):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        color_str("".join(failed), color="red", mode=["bold", "underline"])
        print("".join(failed))
    else:
        assert isinstance(module_name, str), f"module_name is not defined for obj: {name}"
        error = import_module(module_name, cls_name)
        raise error


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        cls()


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by huggingface/transformers
    # https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)
