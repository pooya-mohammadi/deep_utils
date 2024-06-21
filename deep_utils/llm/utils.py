from torch import nn


class LLMUtils:
    @staticmethod
    def get_max_length(model: nn.Module) -> int:
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(model.config, length_setting, None)
            if max_length:
                print(f"Found max length: {max_length}")
                break
        if not max_length:
            max_length = 1024
            print(f"Using default max length: {max_length}")
        return max_length
