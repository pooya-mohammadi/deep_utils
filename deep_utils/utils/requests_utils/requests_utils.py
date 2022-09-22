def get_request(ip, down_message="Down"):
    """
    This simple function wraps the get request
    :param ip:
    :param down_message:
    :return:
    """
    import requests
    try:
        status = requests.get(ip).json()
    except:
        status = down_message
    return status
