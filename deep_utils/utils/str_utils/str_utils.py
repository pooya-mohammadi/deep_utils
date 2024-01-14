from typing import Union, List


class StringUtils:
    @staticmethod
    def split(input_str: str, *split, remove_empty: bool = True) -> List[str]:
        """

        :param input_str:
        :param split:
        :param remove_empty:
        :return:
        >>> StringUtils.split("wow.r pooya, wowo", " ", ",")
        ['wow.r', 'pooya', 'wowo']
        >>> StringUtils.split("wow.r pooya, wowo", " ", ",", ".")
        ['wow', 'r', 'pooya', 'wowo']
        """
        if isinstance(input_str, str):
            input_str = [input_str]
        while split:
            inner_list = []
            split_val, *split = split
            for str_val in input_str:
                inner_list.extend([s for s in str_val.split(split_val) if not remove_empty or s])
            input_str = inner_list
        return input_str
