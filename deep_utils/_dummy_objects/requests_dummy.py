from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class DownloadUtils(metaclass=DummyObject):
    _backend = [
        ("requests", "", "pip"),
    ]
    _module = "deep_utils.utils.download_utils.download_utils"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
