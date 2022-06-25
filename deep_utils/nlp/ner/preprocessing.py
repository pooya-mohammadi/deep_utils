from copy import deepcopy
from typing import Callable, List, Tuple, Union

from deep_utils.nlp.ner.base_operations import NEROperation


class SubTokenReplacement(NEROperation):
    def __init__(
        self,
        tokenizer: Callable[[str], List[str]] = str.split,
        sub_token="\u200c",
        replacement=" ",
        ner_type="BIO",
        others_label="o",
        make_copy=True,
    ):
        """
        This class replaces a sub-token with space and fixes the labels. The default sub-token is zwnj
        :param tokenizer:
        :param ner_type:
        :param make_copy: whether to create copy of the input tokens and labels to prevent data modification, the default is True
        :param sub_token: default is zwnj
        :param replacement: replacement
        :param others_label: default is o
        >>> sub_token_replacement = SubTokenReplacement()
        >>> tokens = ["تهران",
        ... r"نعمت\u200cآباد",
        ... "نور"]
        >>> labels = ["B-province", "B-section", "B-street"]
        >>> tokens, labels = sub_token_replacement(tokens, labels)
        >>> tokens
        ['تهران', 'نعمت', 'آباد', 'نور']
        >>> labels
        ['B-province', 'B-section', 'I-section', 'B-street']
        """
        super(SubTokenReplacement, self).__init__(ner_type=ner_type)
        self.sub_token = sub_token
        self.tokenizer = tokenizer
        self.make_copy = make_copy
        self.replacement = replacement
        self.others_label = others_label

    def __call__(
        self,
        tokens_x: Union[Tuple[str], List[str]],
        labels_y: Union[Tuple[str], List[str]],
    ):
        if self.make_copy:
            tokens_x, labels_y = deepcopy(tokens_x), deepcopy(labels_y)
        super().__call__(tokens_x, labels_y)
        token_index = 0
        while token_index < len(tokens_x):
            token = tokens_x[token_index]
            label = labels_y[token_index]
            if self.sub_token in token:
                replacement = token.replace(self.sub_token, self.replacement)
                sub_tokens = self.tokenizer(replacement)
                del tokens_x[token_index]
                del labels_y[token_index]
                if label.startswith("B-"):
                    sub_labels = [label] + [
                        "I-" + label.replace("B-", "")
                        for _ in range(len(sub_tokens) - 1)
                    ]
                elif label.startswith("I-") or label == self.others_label:
                    sub_labels = [label for _ in sub_tokens]
                else:
                    raise ValueError()

                for i, (sub_token, sub_label) in enumerate(zip(sub_tokens, sub_labels)):
                    tokens_x.insert(token_index + i, sub_token)
                    labels_y.insert(token_index + i, sub_label)

                token_index += len(sub_tokens)
                assert len(tokens_x) == len(labels_y)
                continue
            token_index += 1
        return tokens_x, labels_y
