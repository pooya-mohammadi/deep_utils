def multiple_replace(text: str, chars_to_mapping: dict):
    """
    This function is used to replace a dictionary of characters inside a text string
    :param text:
    :param chars_to_mapping:
    :return:
    """
    import re

    pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
    return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))
