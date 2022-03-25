import importlib.util
from collections import OrderedDict


def is_available(lib_name: str):
    import_error = "{0} " + f"""
    requires {lib_name} library but it was not found in your environment. You can install it with the following instructions:
    ```
    pip install {lib_name}
    ```
    In a notebook or a colab, you can install it by executing a cell with
    ```
    !pip install {lib_name}
    ```
    """
    return lib_name, (lambda: importlib.util.find_spec(lib_name) is not None, import_error)


_is_tf_available = importlib.util.find_spec("tensorflow") is not None
_is_torch_available = importlib.util.find_spec("torch") is not None
_is_cv2_available = importlib.util.find_spec("cv2") is not None


def is_torch_available():
    return _is_torch_available


def is_tf_available():
    return _is_tf_available


def is_cv2_available():
    return _is_cv2_available


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

BACKENDS_MAPPING = OrderedDict(
    [
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        is_available("seaborn"),
        is_available("pyaml")
    ]
)
