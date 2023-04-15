from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class BlipTorchImageCaption(metaclass=DummyObject):
    _backend = [("torch", "1.13.1", "pip"),
                ("torchvision", "0.14.1", "pip"),
                ("timm", "0.4.12", "pip"),
                ("transformers", "4.16.*", "pip"),
                ("fairscale", "0.4.4", "pip")
                ]
    _module = "deep_utils.callbacks.torch.torch_tensorboard"

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend, module_name=self._module, cls_name=self.__class__.__name__)
