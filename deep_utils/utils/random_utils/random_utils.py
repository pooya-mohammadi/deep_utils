from typing import List, Tuple, Union


def random_choice_group(
    group_1: Union[List, Tuple], group_2: Union[List, Tuple]
) -> tuple:
    assert len(group_1) == len(
        group_2), "lengths of the input groups are not the same!"
    from random import choice

    index = choice(list(range(len(group_1))))
    return group_1[index], group_2[index]
