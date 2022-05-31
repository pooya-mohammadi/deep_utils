from typing import Union, List, Tuple


def check_bio_labels(input_labels: Union[str, List[str], Tuple[str]], query_label: str):
    """
    Checks whether the input-labels and queried labels are the same.
    :param input_labels:
    :param query_label:
    :return:
    >>> input_labels = ["B-Loc", "I-Loc"]
    >>> query_label = "Loc"
    >>> check_bio_labels(input_labels, query_label)
    >>> True
    """
    if isinstance(input_labels, str):
        input_labels = input_labels.replace("B-", "").replace("I-", "")
        if input_labels == query_label:
            return True
    elif isinstance(input_labels, list):
        input_labels = [lbl.replace("B-", "").replace("I-", "") for lbl in input_labels]
        if len(set(input_labels)) == 1 and input_labels[0] == query_label:
            return True
    else:
        raise ValueError("Input_labels's type is not supported!")
    return False
