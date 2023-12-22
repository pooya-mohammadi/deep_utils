from re import sub
from typing import Union, Optional


class PyUtils:

    @staticmethod
    def camel2snake_case(camel_case: str):
        """
        Convert camel case to snake case
        :param camel_case:
        :return:
        >>> PyUtils.camel2snake_case('CamelCase')
        'camel_case'
        >>> PyUtils.camel2snake_case('ConfigClass')
        'config_class'
        """
        return '_'.join(
            sub('([A-Z][a-z]+)', r' \1',
                sub('([A-Z]+)', r' \1',
                    camel_case.replace('-', ' '))).split()).lower()

    @staticmethod
    def static_upper_case2snake_case(static_upper_case: str):
        """
        Convert static upper case to snake case
        :param static_upper_case:
        :return:
        >>> PyUtils.static_upper_case2snake_case('STATIC_UPPER_CASE')
        'static_upper_case'
        """
        return static_upper_case.lower()

    @staticmethod
    def color_str(text: str, color: Optional[str], mode: Union[str, list]):
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
    def print(*args, sep=' ', end='\n', file=None, color: Optional[str] = None, mode: Union[str, list] = None):
        """
        colorful print!
        :param args:
        :param sep:
        :param end:
        :param file:
        :param color:
        :param mode: text mode: available modes: bold, underline
        :return:
        """
        args = [PyUtils.color_str(str(arg), color=color, mode=mode) for arg in args]
        print(*args, sep=sep, end=end, file=file)


if __name__ == '__main__':
    PyUtils.print("pooya is walking", color="yellow")
    PyUtils.print("pooya is walking", color="red")
    PyUtils.print("pooya is walking", mode="bold", color="black")
