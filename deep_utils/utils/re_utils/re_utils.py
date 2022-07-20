import re


class REUtils:
    @staticmethod
    def cleaning(input_string: str, delimiter: str = "-"):
        cleanings = {
            "  ": " ",
            "- -": delimiter,
            "--": delimiter,
        }
        for cleaning_str, clean_val in cleanings.items():
            input_string = input_string.replace(cleaning_str, clean_val)
        return input_string.strip()

    @staticmethod
    def split_char_number_sentence(input_address):
        return " ".join(
            REUtils.split_char_number(var) for var in input_address.split(" ")
        )

    @staticmethod
    def split_word_punctuation_sentence(input_address):
        return " ".join(
            REUtils.split_word_punctuation(var) for var in input_address.split(" ")
        )

    @staticmethod
    def split_char_number(input_string, punctuations=",*!+-.،"):
        """
        creates a space between english numbers and other characters
        :param input_string:
        :param punctuations:
        :return:
         >>> REUtils.split_char_number("s12")
         's 12'
         >>> REUtils.split_char_number(",")
         ','
        """
        if "+" in punctuations:
            punctuations = punctuations.replace("+", "")
            plus = r"|\+"
        else:
            plus = ""
        return " ".join(
            re.findall(rf"[^\W\d]+|\d+|[{punctuations}]+{plus}", input_string)
        )

    @staticmethod
    def split_word_punctuation(input_string, punctuations=",*!+-.،"):
        r"""
        creates a space between characters and punctuations.
        \W is opposite of \w and ^ means it's reverse
        :param input_string:
        :param punctuations
        :return:
        >>> REUtils.split_word_punctuation("تهران+")
        'تهران +'
        >>> REUtils.split_word_punctuation("21+")
        '21 +'
        >>> REUtils.split_word_punctuation("۲۱+")
        '۲۱ +'
        >>> REUtils.split_word_punctuation("pooya+")
        'pooya +'
        """

        if "+" in punctuations:
            punctuations = punctuations.replace("+", "")
            plus = r"|\+"
        else:
            plus = ""
        return " ".join(re.findall(rf"[^\W^_]+|[{punctuations}]+{plus}", input_string))

    @staticmethod
    def replace(input_string, pattern, result):
        input_string = re.sub(pattern, result, input_string)
        return input_string

    @staticmethod
    def get_left_right_linespaces(value, left="[\s,]*", right="\s+"):
        expression = left + value + right
        return expression

    @staticmethod
    def replace_single_char(
            input_string, replace_expression, result_expression, left="[\s,]+", right="\s+"
    ):
        """
        Replaces a single character with a complete word
        :param input_string:
        :param replace_expression:
        :param result_expression:
        :param left:
        :param right:
        :return:
        """
        input_string = " " + input_string  # for those starting at index zero
        pattern = REUtils.get_left_right_linespaces(
            replace_expression, left=left, right=right
        )
        input_string = REUtils.replace(
            input_string, pattern, " " + result_expression + " "
        )
        input_string = REUtils.cleaning(
            input_string
        )  # This is done to clean up extra spaces and other characters!
        return input_string

    @staticmethod
    def remove_en_punctuations(input_string):
        """
        Removes punctuations in a string
        :param input_string:
        :return:
        >>> REUtils.remove_en_punctuations('!hi. wh?at is the weat[h]er lik?e)()()(/\.')
        'hi what is the weather like'
        """
        return re.sub(r'[^\w\s]', '', input_string)

    @staticmethod
    def replace_map(text: str, chars_to_map: dict):
        """
        This function is used to replace a dictionary of characters inside a text string
        :param text:
        :param chars_to_map:
        :return:
        """
        import re

        pattern = "|".join(map(re.escape, chars_to_map.keys()))
        return re.sub(pattern, lambda m: chars_to_map[m.group()], str(text))


if __name__ == "__main__":
    str_ = "s12"
    output = REUtils.split_char_number(str_)
    print(output)
