from typing import List, Tuple, Union


class NEROperation:
    """
    This is a basic ner operation that validates ner-type in __init__ and implemented a simple __call__ method for the
    operation at hand.
    """

    NER_TYPES = ["BIO"]

    def __init__(self, ner_type="BIO"):
        self.ner_type = ner_type
        assert (
            ner_type in self.NER_TYPES
        ), f"[ERROR] ner_type: {self.ner_type} is not supported!"

    def __call__(
        self,
        tokens_x: Union[Tuple[str], List[str]],
        labels_y: Union[Tuple[str], List[str]],
    ):
        assert len(tokens_x) == len(
            labels_y
        ), f"[ERROR] number of tokens and labels is not equal"
