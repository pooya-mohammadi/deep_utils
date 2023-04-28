from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class TensorboardTorch(metaclass=DummyObject):
    _backend = ["torch"]
    _module = "deep_utils.callbacks.torch.torch_tensorboard"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class TorchUtils(metaclass=DummyObject):
    _backend = ["torch"]
    _module = "deep_utils.utils.torch_utils.torch_utils"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)


class BlocksTorch(metaclass=DummyObject):
    _backend = [("torch", "1.13.1", "pip")]
    _module = "deep_utils.blocks.torch.blocks_torch"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
