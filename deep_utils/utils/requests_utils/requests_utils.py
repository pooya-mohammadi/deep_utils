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


async def post_json(input_url, data: dict, header="application/json"):
    """
    Async format for sending requests
    :param input_url:
    :param data:
    :param header:
    :return:
    """
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(input_url, json=data,
                                     headers={"Content-Type": header})

        return response.json()
