def check_num_exists(input_, lan="en"):
    """
    This function is used to check whether a character exists in 
    :param input_:
    :param lan:
    :return:
    """
    language = dict(en="0123456789", fa="۰۱۲۳۴۵۶۷۸۹")

    for num in language[lan]:
        if num in input_:
            return True
    return False
