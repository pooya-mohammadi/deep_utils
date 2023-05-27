from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class ImageEditingGLIDE(metaclass=DummyObject):
    _backend = [("torch", "1.13.1", "pip"),
                ("glide_text2im", "0.1.0", "pip"),
                ]
    _module = "deep_utils.vision.image_editing.glide.glide_image_editing"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
