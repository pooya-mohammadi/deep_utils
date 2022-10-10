def get_counter_name(name:str, name_dict: dict, counter=0, separator="_"):
    """
    increase the counter till it reaches a number that is not in the name_dict
    :param name:
    :param name_dict:
    :param counter:
    :param separator:
    :return:
    """
    while name + f"{separator}{counter}" in name_dict:
        counter += 1
    return name + f"{separator}{counter}"
