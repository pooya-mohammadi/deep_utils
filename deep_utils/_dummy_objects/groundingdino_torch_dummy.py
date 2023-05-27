from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class Text2BoxVisualGroundingDino(metaclass=DummyObject):
    _backend = [
        ("torch", "1.13.1", "pip"),
        ("groundingdino", "0.1.0", "pip"),
        ("PIL", "9.3.0", "pip"),
    ]
    _module = "deep_utils.vision.text2box_visual_grounding.dino.visual_grounding_dino_torch"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
