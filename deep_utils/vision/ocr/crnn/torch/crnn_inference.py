from pathlib import Path
from typing import Union, List
import numpy as np
import torch
from PIL import Image
from .crnn_model import CRNNModelTorch
from deep_utils.utils.ctc_decoder.ctc_decoder import CTCDecoder
from deep_utils.vision.vision_utils.torch_vision_utils.torch_vision_utils import TorchVisionUtils


class CRNNInferenceTorch:
    def __init__(self, model_path, decode_method='greedy', device='cpu'):
        self.device = device
        self.decode_method = decode_method
        state_dict = torch.load(model_path, map_location=self.device)
        self.model = CRNNModelTorch(state_dict['img_h'], n_channels=state_dict['n_channels'],
                                    n_classes=state_dict['n_classes'], n_hidden=state_dict['n_hidden'],
                                    lstm_input=state_dict['lstm_input'])
        try:
            self.model.load_state_dict(state_dict["state_dict"])
        except:
            self.model.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict["state_dict"].items()
                }
            )
        self.model.eval()
        self.model.to(self.device)

        self.label2char = state_dict['label2char']
        self.transformer = state_dict['val_transform']
        del state_dict

    def infer(self, img: Union[str, Path, np.ndarray], get_string=True):
        """
        :param img:
        :param get_string: Get the string,
        :return:
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        else:
            img = Image.open(img)
        image = self.transformer(img)
        image = image.view(1, *image.size())
        with torch.no_grad():
            preds = self.model(image).cpu().squeeze(1).numpy()
        sim_pred = CTCDecoder.ctc_decode(preds, decoder_name=self.decode_method, label2char=self.label2char)
        if get_string:
            sim_pred = "".join(sim_pred)
        return sim_pred

    def infer_group(self, images: Union[List[np.ndarray]]):
        images = TorchVisionUtils.transform_concatenate_images(images, self.transformer, device=self.device)
        with torch.no_grad():
            preds = self.model(images).squeeze(0).numpy()
        sim_preds = CTCDecoder.ctc_decode_batch(preds, decoder_name=self.decode_method, label2char=self.label2char)
        return sim_preds

    @staticmethod
    def get_stats(cls_logits):
        probs = torch.max(cls_logits, 2)[0].T.detach().numpy()
        min, max, mean, std = np.min(probs, 1), np.max(probs, 1), np.mean(probs, 1), np.std(probs, 1)
        return dict(min=min.tolist(), max=max.tolist(), mean=mean.tolist(), std=std.tolist())

    def get_reliability(self, stats):
        reliables = []
        for i in range(len(stats['min'])):
            min = stats['min'][i]
            std = stats['std'][i]
            mean = stats['mean'][i]
            if min < self.min_t or std > self.std_t or mean < self.mean_t:
                reliables.append(False)
            else:
                reliables.append(True)
        return reliables