import importlib.util
from collections import OrderedDict
from deep_utils.utils.utils.str_utils import color_str


def is_available(lib_name: str):
    import_error = (
            "{0} "
            + f"""
    requires {lib_name} library but it was not found in your environment. You can install it with the following instructions:
    ```
    pip install {lib_name}
    ```
    In a notebook or a colab, you can install it by executing a cell with
    ```
    !pip install {lib_name}
    ```
    """
    )
    return lib_name, (
        lambda: importlib.util.find_spec(lib_name) is not None,
        import_error,
    )


_is_tf_available = importlib.util.find_spec("tensorflow") is not None
_is_torch_available = importlib.util.find_spec("torch") is not None
_is_cv2_available = importlib.util.find_spec("cv2") is not None
_is_torchvision_available = importlib.util.find_spec("torchvision") is not None
_is_torchaudio_available = importlib.util.find_spec("torchaudio") is not None
_is_pyannote_audio_available = importlib.util.find_spec("pyannote") is not None


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
        is_available("pyaml"),
        is_available("numpy"),
        is_available("albumentations"),
        is_available("sklearn"),
        is_available("PIL"),
        is_available("pyannote"),
        is_available("librosa"),
        is_available("transformers"),
        is_available("soundfile")
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        color_str("".join(failed), color="red", mode=["bold", "underline"])
        print("".join(failed))


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        cls()
