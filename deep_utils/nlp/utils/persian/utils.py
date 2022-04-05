def persian_num2english(input_string: str):
    """
    Converts persian numbers to english
    Args:
        input_string:

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
        "۰": "0"}
    output_string = "".join([NUM_MAP.get(c, c) for c in input_string])
    return output_string


def arabic_char2fa_char(input_string: str):
    arabic2persian = {
        "ك": "ک",
        "ي": "ی",
    }
    out_string = "".join(arabic2persian.get(s, s) for s in input_string)
    return out_string