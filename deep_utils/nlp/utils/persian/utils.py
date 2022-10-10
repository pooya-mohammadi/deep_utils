from typing import Union


def persian_num2english(input_string: str, reverse: bool = False):
    """
    Converts persian numbers to english
    Args:
        input_string:
        reverse: If set to True, converts english 2 persian!

    Returns:

    """
    NUM_MAP = {
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "۰": "0",
    }
    if reverse:
        NUM_MAP = {v: k for k, v in NUM_MAP.items()}
    output_string = "".join([NUM_MAP.get(c, c) for c in input_string])
    return output_string


def arabic_char2fa_char(input_string: str):
    arabic2persian = {
        "ك": "ک",
        "ي": "ی",
    }
    out_string = "".join(arabic2persian.get(s, s) for s in input_string)
    return out_string


def num2fa_spoken(num_string: Union[str, int]):
    """
    converting string number to the spoken format. This function converts both persian and english workds to persian
    spoken words
    :param num_string:
    :return:
    >>> num2fa_spoken("30")
    'سی ام'
    >>> num2fa_spoken("21")
    'بیست و یکم'
    >>> num2fa_spoken("۳۲")
    'سی و دوم'
    >>> num2fa_spoken(2)
    'دوم'
    """
    from num2fawords import words
    num_string = str(num_string)
    if num_string.isdigit():
        written_num = words(num_string)
        if written_num[-2:] == "سه":
            written_num = written_num[:-2] + "سوم"
        elif written_num[-1] in ["ی", ]:
            written_num += " ام"
        else:
            written_num += "م"
        return written_num
    else:
        return num_string
