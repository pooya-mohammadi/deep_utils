from typing import Union
from deep_utils.utils.re_utils.re_utils import REUtils


class PersianUtils:

    @staticmethod
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

    @staticmethod
    def arabic_char2fa_char(input_string: str):
        arabic2persian = {
            "ك": "ک",
            "ي": "ی",
        }
        out_string = "".join(arabic2persian.get(s, s) for s in input_string)
        return out_string

    @staticmethod
    def num2fa_spoken(num_string: Union[str, int], split_num_char: bool = False):
        """
        converting string number to the spoken format. This function converts both persian and english workds to persian
        spoken words
        :param num_string:
        :param split_num_char: if set to True, splits connected numbers and characters
        :return:
        >>> num2fa_spoken("30")
        'سی ام'
        >>> num2fa_spoken("21")
        'بیست و یکم'
        >>> num2fa_spoken("۳۲")
        'سی و دوم'
        >>> num2fa_spoken(2)
        'دوم'
        >>> num2fa_spoken("204غربی", split_num_char=True)
        'دویست و چهارم غربی'
        """

        num_string = str(num_string)
        if split_num_char:
            num_string = REUtils.split_char_number(num_string)
        num_string = " ".join([PersianUtils._num2fa_spoken(word) for word in num_string.split(" ")])
        return num_string

    @staticmethod
    def _num2fa_spoken(num_string):
        from num2fawords import words
        if num_string.isdigit():
            spoken_num = words(num_string)
            if spoken_num[-2:] == "سه":
                spoken_num = spoken_num[:-2] + "سوم"
            elif spoken_num[-1] in ["ی", ]:
                spoken_num += " ام"
            else:
                spoken_num += "م"
        else:
            spoken_num = num_string
        return spoken_num

    @staticmethod
    def num2fa_spoken_sentence(sentence: str, split_num_char=False) -> str:
        """
        Applies num2fa_spoken on a sentence of words!
        :param sentence:
        :param split_num_char: if set to True, splits connected numbers and characters
        :return:
        """
        return " ".join(
            [PersianUtils.num2fa_spoken(word, split_num_char=split_num_char) for word in sentence.split(" ")])
