import tiktoken


class TikTokenUtils:
    @staticmethod
    def count_tokens(input_context: str, model_name: str = "gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(input_context))
        return num_tokens
