import torch
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.utils.object_utils.object_utils import get_obj_variables
from torch import nn


class TorchUtils:
    @staticmethod
    def save_config_to_weight(weight_path, config, logger=None, verbose=1, **kwargs):

        best_weight = torch.load(weight_path)
        for k, v in get_obj_variables(config).items():
            if k not in best_weight:
                best_weight[k] = v
            else:
                log_print(logger, f"[Warning] Did not save {k} = {v} because there is a variable with the same name!",
                          verbose=verbose)
        # Add kwargs
        for k, v in kwargs.items():
            if k not in best_weight:
                best_weight[k] = v
            else:
                log_print(logger, f"[Warning] Did not save {k} = {v} because there is a variable with the same name!",
                          verbose=verbose)
        torch.save(best_weight, weight_path)

    @staticmethod
    def load_model(model: nn.Module, state_dict: dict) -> nn.Module:
        """
        Load model from state dict. If it cannot, it will try pytorch lightning format.
        :param model:
        :param state_dict:
        :return:
        """
        try:
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(
                {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                }
            )
        return model
