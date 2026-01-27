from typing import Union, List, Optional, Tuple

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
    def replace(input_str: str, replace: Union[str, Tuple[str, ...]], replace_with: Union[str , Tuple[str]] = None) -> str:

        replaces = (replace, ) if isinstance(replace, str) else replace
        replace_withs = [""] * len(replace) if replace_with is None else replace_with
        replace_withs = [replace_withs] * len(replaces) if isinstance(replace_withs, str) else replace_withs
        for replace, replace_with in zip(replaces, replace_withs):
            input_str = input_str.replace(replace, replace_with)
        return input_str


    @staticmethod
    def right_replace(input_str: str, replace: Union[str, List[str]], replace_with: str, count: int = 1):
        replace_list = [replace] if isinstance(replace, str) else replace
        reverse_input_str = input_str[::-1]
        reverse_replace_with = replace_with[::-1]
        for replace in replace_list:
            reverse_replace = replace[::-1]
            reverse_input_str = reverse_input_str.replace(reverse_replace,
                                                          reverse_replace_with,
                                                          count)
        return reverse_input_str[::-1]


    @staticmethod
    def color_str(text: str, color: Optional[str] = "yellow", mode: Union[str, list] = "bold"):
        """
        colorful texts!
        :param text: input text
        :param color: text color
        :param mode: defines text's modes. Valid modes: [ underline, bold ]. Pass a list of modes in case more one mode is needed!
        :return: colored text
        """
        if isinstance(mode, str):
            mode = [mode]
        colors = {
            "black": "\033[30m",  # basic colors
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bright_black": "\033[90m",  # bright colors
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
            "end": "\033[0m",  # misc
            "bold": "\033[1m",
            "underline": "\033[4m",
        }
        return (colors[color.lower()] if color is not None else "") + (
            "".join(colors[m.lower()] for m in mode) if mode is not None else "") + text + \
            colors["end"]

    @staticmethod
    def print(*args, sep=' ', end='\n', file=None, color: Optional[str] = "red", mode: Union[str, list] = "bold", verbose: bool = True):
        """
        colorful print!
        :param args:
        :param sep:
        :param end:
        :param file:
        :param color:
        :param mode: text mode: available modes: bold, underline
        :param verbose:
        :return:
        """
        args = [StringUtils.color_str(str(arg), color=color, mode=mode) for arg in args]
        if verbose:
            print(*args, sep=sep, end=end, file=file)


if __name__ == '__main__':
    import string
    print(StringUtils.replace("7/Don'tLabel!People_Nouma nAliKhan_0.mp4",
                        tuple(string.punctuation + " "), ""))