import torch


class MedMetricsTorch:
    @staticmethod
    def dice_score(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1.0) -> float:
        """
        Dice score for binary segmentation!
        score = (2*TP)/(2*TP + FP + FN)
        :param y_true:
        :param y_pred:
        :param smooth:
        :return:
        """
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score
