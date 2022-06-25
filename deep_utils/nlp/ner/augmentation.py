import random
from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union

from deep_utils.nlp.ner.base_operations import NEROperation
from deep_utils.nlp.ner.taggers import bio_tagger
from deep_utils.nlp.ner.utils_ import check_bio_labels
from deep_utils.utils.random_utils.random_utils import random_choice_group


class NERAugRemove(NEROperation):
    """
    This augmentation randomly removes tokens and their corresponding labels, This is not just for persian language.
    >>> remove_dict = { ("-", "o"): 0.5, (",", "o"): 0.5, ("استان", "province"): 0.5, ("شهر", "city"): 0.5}

    >>> seed = 1234
    >>> remove_aug = NERAugRemove(remove_dict,random_seed=seed)
    >>> tokens = ['استان', 'تهران', '-', 'شهر', 'تهران', '-', 'نعمت\u200cآباد', '-', 'خیابان', 'نور', '-', 'کوچه', '21', '-', 'پلاک', '23', '-', 'تلفن', '982155871442']
    >>> labels = ['B-province',  'I-province', 'o', 'B-city', 'I-city', 'o', 'B-section', 'o', 'B-street', 'I-street', 'o', 'B-alley', 'I-alley', 'o', 'B-house_number', 'I-house_number', 'o', 'B-phone', 'I-phone']
    >>> remove_aug(tokens, labels)
    (['استان', 'تهران', '-', 'تهران', 'نعمت\u200cآباد', 'خیابان', 'نور', '-', 'کوچه', '21', '-', 'پلاک', '23', '-', 'تلفن', '982155871442'], ['B-province', 'I-province', 'o', 'B-city', 'B-section', 'B-street', 'I-street', 'o', 'B-alley', 'I-alley', 'o', 'B-house_number', 'I-house_number', 'o', 'B-phone', 'I-phone'])
    """

    def __init__(
        self,
        remove_dict: Dict[Tuple[str, str], float],
        tokenizer: Callable[[str], List[str]] = str.split,
        ner_type="BIO",
        make_copy=True,
        random_seed=None,
    ):
        """

        :param remove_dict:
        :param tokenizer:
        :param ner_type:
        :param make_copy: whether to create copy of the input tokens and labels to prevent data modification, the default is True
        :param random_seed:
        """
        super(NERAugRemove, self).__init__(ner_type=ner_type)
        if random_seed:
            random.seed(random_seed)
        self.tokenizer = tokenizer
        self.make_copy = make_copy
        self.p_dict = {}
        self.augmentation_len_dict = defaultdict(list)
        for (query_tokens, query_label), prob in remove_dict.items():
            query_tokens = tuple(self.tokenizer(query_tokens))
            key = (query_tokens, query_label)
            # Getting probabilities from the vals
            self.p_dict[key] = prob
            # This code will convert str to list[str] if the input labels were string!
            # Getting the labels from vals
            self.augmentation_len_dict[len(query_tokens)].append(
                (query_tokens, query_label)
            )
        self.aug_tokens_lengths = sorted(
            list(self.augmentation_len_dict.keys()), reverse=True
        )

    def __call__(self, tokens_x: List[str], labels_y: List[str]):
        if self.make_copy:
            tokens_x, labels_y = deepcopy(tokens_x), deepcopy(labels_y)
        super().__call__(tokens_x, labels_y)

        for token_len in self.aug_tokens_lengths:
            for query_tokens, query_label in self.augmentation_len_dict[token_len]:
                token_index = 0
                p = self.p_dict[(query_tokens, query_label)]
                while token_index <= (len(tokens_x) - token_len):
                    selected_tokens = tokens_x[token_index: token_index + token_len]
                    selected_labels = labels_y[token_index: token_index + token_len]
                    if (
                        query_tokens == tuple(selected_tokens)
                        and (
                            not query_label
                            or check_bio_labels(selected_labels, query_label)
                        )
                        and random.random() <= p
                    ):
                        remove_label = labels_y[token_index: token_index + token_len][
                            0
                        ]
                        if remove_label.startswith("B"):
                            # If it starts with B it means by removing this one the others fragments won't have Beginning
                            # tag, so the next tag should be changed to start with "B" instead of I
                            next_index = token_index + token_len
                            if (
                                next_index < len(labels_y)
                                and labels_y[next_index].startswith("I")
                                and labels_y[next_index][2:] == query_label
                            ):
                                labels_y[next_index] = (
                                    "B-" + query_label
                                )  # start with B

                        del tokens_x[token_index: token_index + token_len]
                        del labels_y[token_index: token_index + token_len]
                        token_index -= token_len
                        # continue
                    token_index += 1

        return tokens_x, labels_y


class NERAugReplacement(NEROperation):
    """
    This class replaces a single token with a single token
    """

    def __init__(
        self,
        replacement_dict: Dict[Tuple[str, Union[None, str]], Tuple[List[str], float]],
        tokenizer: Callable[[str], List[str]] = str.split,
        ner_type="BIO",
        make_copy=True,
        random_seed=None,
    ):
        """

        :param remove_dict:
        :param tokenizer:
        :param ner_type:
        :param make_copy: whether to create copy of the input tokens and labels to prevent data modification, the default is True.
        :param random_seed:
        """
        super(NERAugReplacement, self).__init__(ner_type=ner_type)
        if random_seed:
            random.seed(random_seed)
        self.tokenizer = tokenizer
        self.make_copy = make_copy
        self.p_dict = {}
        self.augmentation_len_dict = defaultdict(list)
        for (query_tokens, query_label), (
            replacement_tokens,
            prob,
        ) in replacement_dict.items():
            query_tokens = tuple(self.tokenizer(query_tokens))
            key = (query_tokens, query_label)
            # Getting probabilities from the vals
            self.p_dict[key] = prob
            # This code will convert str to list[str] if the input labels were string!
            # Getting the labels from vals
            replacement_tokens = [
                self.tokenizer(replacement_token)
                for replacement_token in replacement_tokens
            ]
            replacement_labels = [
                bio_tagger(r_t, query_label) for r_t in replacement_tokens
            ]
            self.augmentation_len_dict[len(query_tokens)].append(
                (query_tokens, query_label, replacement_tokens, replacement_labels)
            )
        self.aug_tokens_lengths = sorted(
            list(self.augmentation_len_dict.keys()), reverse=True
        )

    def __call__(
        self,
        tokens_x: Union[Tuple[str], List[str]],
        labels_y: Union[Tuple[str], List[str]],
    ):
        if self.make_copy:
            tokens_x, labels_y = deepcopy(tokens_x), deepcopy(labels_y)
        super().__call__(tokens_x, labels_y)
        for token_len in self.aug_tokens_lengths:
            for (
                query_tokens,
                query_label,
                replace_tokens,
                replace_labels,
            ) in self.augmentation_len_dict[token_len]:
                token_index = 0
                p = self.p_dict[(query_tokens, query_label)]
                while token_index <= (len(tokens_x) - token_len):
                    selected_tokens = tokens_x[token_index: token_index + token_len]
                    selected_labels = labels_y[token_index: token_index + token_len]
                    if (
                        query_tokens == tuple(selected_tokens)
                        and (
                            not query_label
                            or check_bio_labels(selected_labels, query_label)
                        )
                        and random.random() <= p
                    ):
                        del tokens_x[token_index: token_index + token_len]
                        del labels_y[token_index: token_index + token_len]
                        replacement_token, replacement_label = random_choice_group(
                            replace_tokens, replace_labels
                        )
                        tokens_x[token_index:token_index] = replacement_token
                        labels_y[token_index:token_index] = replacement_label
                        token_index += len(replacement_token)
                        continue
                    token_index += 1

        return tokens_x, labels_y


if __name__ == "__main__":
    replacement = NERAugReplacement(
        replacement_dict={
            ("خیابان", "street"): (["خ", "خ."], 0.5),
            ("کوچه", "alley"): (["ک", "ک."], 0.5),
            ("-", "o"): ([",", "-"], 0.5),
            ("بلوار", "boulevard"): (["بل", "بل."], 0.5),
        }
    )
    removal = NERAugRemove(
        remove_dict={("-", "o"): 0.5, ("استان", "province"): 0.2})
    address = {
        "tokens": [
            "استان",
            "قم",
            "-",
            "شهر",
            "قم",
            "-",
            "-",
            "خیابان",
            "بعثت",
            "-",
            "بلوار",
            "تهرانی",
            "مقدم",
            "-",
            "پلاک",
            "146",
            "-",
            "طبقه",
            "2",
            "-",
            "تلفن",
            "989373584544",
        ],
        "labels": [
            "B-province",
            "I-province",
            "o",
            "B-city",
            "I-city",
            "o",
            "o",
            "B-street",
            "I-street",
            "o",
            "B-boulevard",
            "I-boulevard",
            "I-boulevard",
            "o",
            "B-house_number",
            "I-house_number",
            "o",
            "B-floor",
            "I-floor",
            "o",
            "B-phone",
            "I-phone",
        ],
    }
    tokens, labels = replacement(
        tokens_x=address["tokens"], labels_y=address["labels"])
    tokens, labels = removal(tokens_x=tokens, labels_y=labels)
    for t, l in zip(tokens, labels):
        print(t, l)
