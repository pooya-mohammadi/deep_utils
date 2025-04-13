import torch
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.utils.object_utils.object_utils import get_obj_variables
from torch import nn


class TorchUtils:
    @staticmethod
    def save_config_to_weight(weight_path, config, logger=None, verbose=1, **kwargs):
        if torch.__version__ >= "2.6.0":
            best_weight = torch.load(weight_path, weights_only=False)
        else:
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

    @staticmethod
    def one_hot_encoder(input_tensor, n_classes: int = None):
        n_classes = n_classes or (int(input_tensor.max()) + 1)
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
