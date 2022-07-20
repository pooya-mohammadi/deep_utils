import torch
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.utils.object_utils.object_utils import get_obj_variables


class TorchUtils:
    @staticmethod
    def save_config_to_weight(weight_path, config, logger=None, verbose=1):

        best_weight = torch.load(weight_path)
        for k, v in get_obj_variables(config).items():
            if k not in best_weight:
                best_weight[k] = v
            else:
                log_print(logger, f"[Warning] Did not save {k} = {v} because there is a variable with the same name!",
                          verbose=verbose)
        torch.save(best_weight, weight_path)
