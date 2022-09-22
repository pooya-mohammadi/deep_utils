def get_request(ip, down_key, down_message="Down") -> dict:
    """
    This simple function wraps the get request
    :param ip:
    :param down_key: key of the output if it fails
    :param down_message:
    :return:
    """
    import requests
    try:
        status = requests.get(ip).json()
    except:
        status = {down_key: down_message}
    return status
