from typing import Union
from typing import List


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

    @staticmethod
    def right_replace(input_str: str, replace: Union[str, list[str]], replace_with: str, count: int = 1):
        replace_list = [replace] if isinstance(replace, str) else replace
        reverse_input_str = input_str[::-1]
        reverse_replace_with = replace_with[::-1]
        for replace in replace_list:
            reverse_replace = replace[::-1]
            reverse_input_str = reverse_input_str.replace(reverse_replace,
                                                          reverse_replace_with,
                                                          count)
        return reverse_input_str[::-1]
