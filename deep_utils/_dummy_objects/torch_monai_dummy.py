from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class MonaiChannelBasedContrastEnhancementD(metaclass=DummyObject):
    _backend = ["monai", "torch"]
    _module = "deep_utils.preprocessing.monai.monai_segmentation"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
