def load_json(json_path: str, encoding="utf-8") -> dict:
    """
    :param json_path: Path to json file to load
    :param encoding: encoding format
    :return: returns a json file
    """
    import json
    with open(json_path, encoding=encoding) as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_path, dict_object: dict, encoding="utf-8", ensure_ascii=True) -> None:
    """
    dumps a json file
    Args:
        dict_object:
        json_path: path to json file
        encoding: encoding format
        ensure_ascii: set to False for persian characters

    Returns:

    """
    import json
    with open(json_path, mode='w', encoding=encoding) as f:
        json.dump(dict_object, f, ensure_ascii=ensure_ascii)
