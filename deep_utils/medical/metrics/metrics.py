import numpy as np
from typing import Sequence, Union

import torch


class MedMetricsTorch:

    @staticmethod
    def _one_hot_encoder(input_tensor, n_classes: int = None):
        n_classes = n_classes or (int(input_tensor.max()) + 1)
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    @staticmethod
    def _dice_score(y_true, y_pred, smooth):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / torch.clip((torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth), 1e-8)
        return score.item()

    @staticmethod
    def dice_score(y_true: Union[torch.Tensor, Sequence], y_pred: Union[torch.Tensor, Sequence],
                   smooth: float = 1.0,
                   include_background: bool = True,
                   n_class: int = None, nan_mean: bool = True) -> float:
        """
        Dice score for binary segmentation!
        score = (2*TP)/(2*TP + FP + FN)
        score should be between 0 and one
        :param y_true:
        :param y_pred:
        :param smooth:
        :param include_background: Whether to include background or not
        :param n_class: Number of classes in case onehot is required

        :return:
        """
        if isinstance(y_true, Sequence):
            y_true = torch.concat([t.unsqueeze(0) for t in y_true])
        if isinstance(y_pred, Sequence):
            y_pred = torch.concat([t.unsqueeze(0) for t in y_pred])
        if y_true.max() > 1:
            y_true = MedMetricsTorch._one_hot_encoder(y_true, n_classes=n_class)
            n_class = y_true.shape[1]
        if y_pred.max() > 1 or y_pred.shape != y_true.shape:
            y_pred = MedMetricsTorch._one_hot_encoder(y_pred, n_classes=n_class)
        if not include_background:
            y_true = y_true[:, 1:]
            y_pred = y_pred[:, 1:]
        scores = []
        if y_true.shape[1] > 1:
            for i in range(y_true.shape[1]):
                scores.append(MedMetricsTorch._dice_score(y_true[:, i], y_pred[:, i], smooth))
            return np.nanmean(scores)
        else:
            return MedMetricsTorch._dice_score(y_true, y_pred, smooth)

if __name__ == '__main__':
    t = torch.randint(1, 2, (1000, 1, 64, 64, 64))
    p = torch.randint(1, 2, (1000, 1, 64, 64, 64))
    # print(t, "\n", p)
    print(MedMetricsTorch.dice_score(t, p))
