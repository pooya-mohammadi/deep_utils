import random
from collections import defaultdict
from typing import Dict, Callable, List, Tuple, Union
from deep_utils.nlp.ner.taggers import bio_tagger
from deep_utils.nlp.ner.utils_ import check_bio_labels
from deep_utils.utils.random_utils.random_utils import random_choice_group


class NERAugmentation:
    NER_TYPES = ["BIO"]

    def __init__(self, ner_type="BIO"):
        self.ner_type = ner_type
        assert ner_type in self.NER_TYPES, f"[ERROR] ner_type: {self.ner_type} is not supported!"

    def __call__(self, tokens_x: Union[Tuple[str], List[str]],
                 labels_y: Union[Tuple[str], List[str]]):
        assert len(tokens_x) == len(labels_y), f"[ERROR] number of tokens and labels is not equal"


class NERAugReplacementDict(NERAugmentation):
    """
    This class replaces a single token with a single token
    """

    def __init__(self,
                 replacement_dict: Dict[Tuple[str, Union[None, str]], Tuple[List[str], float]],
                 tokenizer: Callable[[str], List[str]] = str.split,
                 ner_type="BIO"):
        super(NERAugReplacementDict, self).__init__(ner_type=ner_type)
        self.tokenizer = tokenizer

        self.replacement_p_dict = {}
        self.augmentation_len_dict = defaultdict(list)
        for (query_tokens, query_label), (replacement_tokens, prob) in replacement_dict.items():
            query_tokens = tuple(self.tokenizer(query_tokens))
            key = (query_tokens, query_label)
            # Getting probabilities from the vals
            self.replacement_p_dict[key] = prob
            # This code will convert str to list[str] if the input labels were string!
            # Getting the labels from vals
            replacement_tokens = [self.tokenizer(replacement_token) for replacement_token in replacement_tokens]
            replacement_labels = [bio_tagger(r_t, query_label) for r_t in replacement_tokens]
            self.augmentation_len_dict[len(query_tokens)].append((query_tokens, query_label,
                                                                  replacement_tokens, replacement_labels))
        self.aug_tokens_lengths = sorted(list(self.augmentation_len_dict.keys()), reverse=True)

    def __call__(self, tokens_x: Union[Tuple[str], List[str]],
                 labels_y: Union[Tuple[str], List[str]]):
        super().__call__(tokens_x, labels_y)
        for token_len in self.aug_tokens_lengths:
            for query_tokens, query_label, replace_tokens, replace_labels in self.augmentation_len_dict[token_len]:
                token_index = 0
                p = self.replacement_p_dict[(query_tokens, query_label)]
                while token_index <= (len(tokens_x) - token_len):
                    selected_tokens = tokens_x[token_index: token_index + token_len]
                    selected_labels = labels_y[token_index: token_index + token_len]
                    if query_tokens == tuple(selected_tokens) and (
                            not query_label or check_bio_labels(selected_labels, query_label)) and random.random() <= p:
                        del tokens_x[token_index: token_index + token_len]
                        del labels_y[token_index: token_index + token_len]
                        replacement_token, replacement_label = random_choice_group(replace_tokens, replace_labels)
                        tokens_x[token_index:token_index] = replacement_token
                        labels_y[token_index:token_index] = replacement_label
                        token_index += len(replacement_token)
                        continue
                    token_index += 1

        return tokens_x, labels_y


if __name__ == '__main__':
    replacement = NERAugReplacementDict(replacement_dict={("خیابان", "street"): (["خ", "خ."], 0.5),
                                                          ("کوچه", "alley"): (["ک", "ک."], 0.5),
                                                          ("-", "o"): (["  ", ",", "_", "-"], 0.5),
                                                          ("بلوار", "boulevard"): (["بل", "بل."], 0.5),
                                                          })
    address = {'tokens': ['استان',
                          'قم',
                          '-',
                          'شهر',
                          'قم',
                          '-',
                          '-',
                          'خیابان',
                          'بعثت',
                          '-',
                          'بلوار',
                          'تهرانی',
                          'مقدم',
                          '-',
                          'پلاک',
                          '146',
                          '-',
                          'طبقه',
                          '2',
                          '-',
                          'تلفن',
                          '989373584544'],
               'labels': ['B-province',
                          'I-province',
                          'o',
                          'B-city',
                          'I-city',
                          'o',
                          'o',
                          'B-street',
                          'I-street',
                          'o',
                          'B-boulevard',
                          'I-boulevard',
                          'I-boulevard',
                          'o',
                          'B-house_number',
                          'I-house_number',
                          'o',
                          'B-floor',
                          'I-floor',
                          'o',
                          'B-phone',
                          'I-phone']}
    tokens, labels = replacement(tokens_x=address['tokens'], labels_y=address['labels'])
    for t, l in zip(tokens, labels):
        print(t, l)
