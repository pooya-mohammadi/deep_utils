def load_json(json_path: str):
    """
    :param json_path: Path to json file to load
    :return: returns a json file
    """
    import json
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data
