import inspect
from argparse import Namespace
from re import sub
from typing import Optional
from typing import Union

from deep_utils.utils.logging_utils.logging_utils import log_print


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

    @staticmethod
    def print_args(args: Optional[Union[dict, Namespace]] = None, show_file=True, show_func=False):
        """
        Prints arguments in a parser!
        :param args:
        :param show_file:
        :param show_func:
        :return:
        """
        # Print function arguments (optional args dict)
        x = inspect.currentframe().f_back  # previous frame
        file, _, func, _, _ = inspect.getframeinfo(x)
        if args is None:  # get args automatically
            args, _, _, frm = inspect.getargvalues(x)
            args = {k: v for k, v in frm.items() if k in args}
        else:
            args = dict(vars(args))
        # try:
        #     file = Path(file).resolve().relative_to(ROOT).with_suffix('')
        # except ValueError:
        #     file = Path(file).stem
        s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
        log_print(None, message=PyUtils.color_str(s + ', '.join(f'{k}={v}' for k, v in args.items())))

    @staticmethod
    def get_obj_variables(obj) -> dict:
        """
        Gets the attributes of an object not the methods.
        :param obj:
        :return:
        """
        out = dict()
        for key in dir(obj):
            val = getattr(obj, key)
            if (key.startswith("__") and key.endswith("__")) or type(val).__name__ == "method":
                continue
            else:
                out[key] = val
        return out

    @staticmethod
    def get_attributes(obj) -> dict:
        """
        Gets the attributes of an object not the methods. Deprecated: The naming was not appropriate, make sure to use
        `get_variables` instead.
        :param obj:
        :return:
        """
        return PyUtils.get_obj_variables(obj)

    @staticmethod
    def update_obj_params(obj_instance: object, args: Union[dict, Namespace]):
        """
        This func/method is used for updating parameters of a class especially a config class with input arguments
        :param obj_instance:
        :param args:
        :return:
        """
        if isinstance(args, Namespace):
            variables = vars(args)
        elif isinstance(args, dict):
            variables = args
        else:
            raise ValueError()
        for k, v in variables.items():
            if hasattr(obj_instance, k):
                setattr(obj_instance, k, v)
            else:
                raise ValueError(f"value {k} is not defined in {obj_instance.__class__.__name__}")

    @staticmethod
    def variable_repr(self):
        """
        A representation func for objects
        :param self:
        :return:
        """
        variables = PyUtils.get_obj_variables(self)
        return f"{self.__class__.__name__} -> " + ", ".join(f"{k}: {v}" for k, v in variables.items())


if __name__ == '__main__':
    PyUtils.print("pooya is walking", color="yellow")
    PyUtils.print("pooya is walking", color="red")
    PyUtils.print("pooya is walking", mode="bold", color="black")
