from typing import Union, List

import numpy as np
import torch

from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.lib_utils.lib_decorators import (
    expand_input,
    get_elapsed_time,
    get_from_config,
    lib_rgb2bgr
)
from deep_utils.utils.pickle_utils.pickle_utils import PickleUtils
from deep_utils.vision.face_recognition.main.main_face_recognition import FaceRecognition, OUTPUT_CLASS
from .config import Config
from .src import load_model


class VggFace2TorchFaceRecognition(FaceRecognition):
    def __init__(self, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            download_variables=("model",),
            **kwargs
        )
        self.config: Config

    @download_decorator
    def load_model(self):
        # LOAD MODELS
        model = load_model(self.config.model_name)
        self.load_state_dict(model, self.config.model)
        model = model.eval()
        self.model = model

    @expand_input(3)
    @get_elapsed_time
    @get_from_config
    def extract_faces(self, img: Union[List[np.ndarray], np.ndarray], is_rgb, get_time=False) -> OUTPUT_CLASS:
        img = torch.cat(
            [self.config.transform(image=lib_rgb2bgr(img_, target_type="bgr", is_rgb=False))["image"].unsqueeze(0) for
             img_ in img], dim=0)

        with torch.no_grad():
            img = img.to(self.config.device)
            output = self.model(img).cpu().numpy()
            output = self.normalizer.transform(output)
        output = OUTPUT_CLASS(encodings=output, )
        return output

    @staticmethod
    def load_state_dict(model, weight_path):
        """
        Set parameters converted from Caffe models authors of VGGFace2 provide.
        See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

        Arguments:
            model: model
            weight_path: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
        """
        weights = PickleUtils.load_pickle(weight_path, mode="rb", encoding='latin1')
        own_state = model.state_dict()
        for name, param in weights.items():
            if name in own_state:
                try:
                    own_state[name].copy_(torch.from_numpy(param))
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                        'dimensions in the checkpoint are {}.'.format(name, own_state[name].shape, param.shape))
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))
