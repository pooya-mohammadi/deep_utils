from typing import Sequence, Union

import torch


class MedMetricsTorch:
    @staticmethod
    def dice_score(y_true: Union[torch.Tensor, Sequence], y_pred: Union[torch.Tensor, Sequence], smooth: float = 1.0) -> float:
        """
        Dice score for binary segmentation!
        score = (2*TP)/(2*TP + FP + FN)
        score should be between 0 and one
        :param y_true:
        :param y_pred:
        :param smooth:
        :return:
        """
        if isinstance(y_true, Sequence):
            y_true = torch.concat([t.unsqueeze(0) for t in y_true])
        if isinstance(y_pred, Sequence):
            y_pred = torch.concat([t.unsqueeze(0) for t in y_pred])
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score.item()


if __name__ == '__main__':

    t = torch.randint(1, 2, (1000, 1, 64, 64, 64))
    p = torch.randint(1, 2, (1000, 1, 64, 64, 64))
    # print(t, "\n", p)
    print(MedMetricsTorch.dice_score(t, p))
