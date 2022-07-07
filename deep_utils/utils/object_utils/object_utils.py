def get_attributes(obj) -> dict:
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
