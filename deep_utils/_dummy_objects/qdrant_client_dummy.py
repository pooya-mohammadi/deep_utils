from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class QdrantUtils(metaclass=DummyObject):
    _backend = ["qdrant_client"]
    _module = "deep_utils.utils.qdrant_utils.qdrant_utils"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
