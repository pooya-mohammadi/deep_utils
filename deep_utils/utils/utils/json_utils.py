def load_json(json_path: str):
    """
    :param json_path: Path to json file to load
    :return: returns a json file
    """
    import json
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_path, dict_object: dict):
    """
    dumps a json file
    Args:
        dict_object:
        json_path: path to json file

    Returns:

    """
    import json
    with open(json_path, mode='w') as f:
        json.dump(dict_object, f)
