from deep_utils.dummy_objects.dummy_framework import DummyObject, requires_backends


class TensorboardTorch(metaclass=DummyObject):
    _backend = ["torch"]
    _module = "deep_utils.callbacks.torch.torch_tensorboard"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
