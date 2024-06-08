import json
from typing import Union, Optional, Dict, Any, Callable, List

import aiohttp


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


def _get_file_content(file_path: str, file_content_type: str = 'multipart/form-data') -> tuple:
    import os
    file_name = os.path.basename(file_path)
    file_content = open(file_path, 'rb')
    return file_name, file_content, file_content_type


async def post_form(input_url, data_key: str, data_path: str):
    """
    Async format for sending form requests!
    :param input_url:
    :param data_path:
    :param data_key:
    :return:
    """
    import httpx
    file_name, file_content, file_content_type = _get_file_content(data_path)
    async with httpx.AsyncClient() as client:
        response = await client.post(input_url, files={data_key: (file_name, file_content, file_content_type)})
        return response.json()


async def post_form_upload(input_url, data_key: str, upload_file):
    """
    Async format for sending form requests. It gets upload file as input
    :param input_url:
    :param upload_file:
    :param data_key:
    :return:
    """
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(input_url, files={
            data_key: (upload_file.filename, upload_file.file, upload_file.content_type)})
        return response.json()


class Requests:
    def __init__(self, json_serialize: Callable = json.dumps):
        """

        :param json_serialize:  ujson.dumps is also acceptable which is a bit faster but might be incompatible!
        """
        self.session = aiohttp.ClientSession(json_serialize=json_serialize)  # creating session

    async def form_post_async(
            self,
            url: str,
            data: Optional[Dict[str, Any]],
            files: Optional[List[Dict[str, Any]]] = None,
            ssl: bool = False,
            encoding: Optional[str] = None
    ):
        """
        The files should have the following items:
        [{'name': variable_name, 'value': file_content, 'content_type': multipart/form-data||application/vnd.ms-excel||etc,
        }, {...}]
        :param url:
        :param data:
        :param files:
        :param ssl:
        :param encoding:
        :return:
        """
        form_data = aiohttp.FormData()
        if data is not None:
            for key, value in data.items():
                form_data.add_field(key, str(value))
        if files is not None:
            for file_obj in files:
                form_data.add_field(file_obj.get("name", "file"), file_obj["value"],
                               filename=file_obj.get("filename"),
                               content_type=file_obj.get("content_type", 'multipart/form-data'))
        output = await self.session.post(url, data=data)
        return output
