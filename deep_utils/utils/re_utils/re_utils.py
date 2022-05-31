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
    def replace(input_string, pattern, result):
        input_string = re.sub(pattern, result, input_string)
        return input_string

    @staticmethod
    def get_left_right_linespace(value, left="[\s,]*", right="\s+"):
        expression = left + value + right
        return expression

    @staticmethod
    def replace_single_char(input_string, replace_expression, result_expression, left="[\s,]+", right="\s+"):
        """
        Replaces a single character with a whole word
        :param input_string: 
        :param replace_expression:
        :param result_expression:
        :param left:
        :param right:
        :return:
        """
        input_string = " " + input_string  # for those starting at index zero
        pattern = REUtils.get_left_right_linespace(replace_expression, left=left, right=right)
        input_string = REUtils.replace(input_string, pattern, " " + result_expression + " ")
        input_string = REUtils.cleaning(input_string)  # This is done to clean up extra spaces and other characters!
        return input_string


if __name__ == '__main__':
    REUtils.replace_single_char("")
