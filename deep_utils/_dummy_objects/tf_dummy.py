from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class TorchUtils(metaclass=DummyObject):
    _backend = ["tf2"]
    _module = "deep_utils.utils.tf_utils.main"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
