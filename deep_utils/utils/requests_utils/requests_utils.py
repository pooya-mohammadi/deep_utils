import json
from typing import Optional, Dict, Any, Callable, List, Literal

import aiohttp
import requests


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


class AIOHttpRequests:

    @staticmethod
    async def put_request(url, data=None, json=None):
        async with aiohttp.ClientSession() as session:
            if data is not None:
                async with session.put(url, data=data) as response:
                    # Check if the request was successful
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        print("Failed to update data:", response.status)
                        print("Response text:", await response.text())
            elif json is not None:
                async with session.put(url, json=data) as response:
                    # Check if the request was successful
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        print("Failed to update data:", response.status)
                        print("Response text:", await response.text())


    @staticmethod
    async def _encoding(response, encoding: str | None):
        if encoding == "content":
            return await response.content.read()
        elif encoding == "text":
            return await response.text()
        elif encoding == "json":
            return await response.json()
        elif encoding is None:
            return response
        else:
            raise ValueError(f"encoding: {encoding} is not supported!")

    @staticmethod
    async def form_post_async(
            url: str,
            data: Optional[Dict[str, Any]] = None,
            files: Optional[List[Dict[str, Any]]] = None,
            ssl: bool = False,
            encoding: Optional[str] = None,
            json_serialize: Callable = json.dumps,
            json: dict = None
    ):
        """
        The files should have the following items:
        [{'name': variable_name, 'value': file_content, 'content_type': multipart/form-data||application/vnd.ms-excel||etc,
        }, {...}]
        data or files one of them should be filled!
        If you want to send a dictionary as json, set the json to True. in that case you cannot send files!
        :param url:
        :param data:
        :param files:
        :param ssl:
        :param encoding:
        :param json_serialize:
        :param json: json data. Data alone is form format. But if you want to send a dict as json simply pass it to json
        :return:
        """
        if json is None and data is None and files is None:
            raise ValueError("Both data and files cannot be ")
        async with aiohttp.ClientSession(json_serialize=json_serialize) as session:
            if not json:
                form_data = aiohttp.FormData()
                if data is not None:
                    for key, value in data.items():
                        form_data.add_field(key, str(value))
                if files is not None:
                    for file_obj in files:
                        form_data.add_field(file_obj.get("name", "file"), file_obj["value"],
                                            filename=file_obj.get("filename"),
                                            content_type=file_obj.get("content_type", 'multipart/form-data'))
                output = await session.post(url, data=form_data, ssl=ssl)
            else:
                output = await session.post(url, json=json, ssl=ssl)
            output = await AIOHttpRequests._encoding(output, encoding)
        return output

    @staticmethod
    async def get_async(url, encoding: Optional[Literal['content', 'text', 'json']] = None):
        """
        Asynchronous get
        :param url:
        :param encoding:
        :return:
        """
        async with aiohttp.ClientSession() as session:
            response = await session.get(url)
            response.raise_for_status()
            if response.status == 200:
                output = await AIOHttpRequests._encoding(response, encoding=encoding)
                return output


class RequestsUtils:
    @staticmethod
    def post_file(url: str, key_name: str, filepath: str, **kwargs):
        files = {key_name: open(filepath, 'rb')}

        output = requests.post(url, files=files, data=kwargs)
        output.raise_for_status()
        return output.json()

    @staticmethod
    def post(url: str, data: dict[str, Any]):
        output = requests.post(url, json=data)
        output.raise_for_status()
        return output.json()


if __name__ == '__main__':
    print(RequestsUtils.post("http://0.0.0.0:8001/llama", {
        "messages": [{"role": "user", "content": "Who are you?"}]
    }))

    # async def port():
    #     import cv2
    #     url = "http://ai.8001.kookaat.ir/predict/single?model_name=word"
    #     cropped_img = cv2.imread("/home/ai/Desktop/sample-ocr.png")
    #     _, buffer = cv2.imencode('.jpg', cropped_img)
    #     file_bytes = buffer.tobytes()
    #     files = [{'name': "image", 'value': file_bytes,
    #               # 'content_type': "multipart / form - data | | application / vnd.ms - excel | | etc,
    #               }]
    #     data = []
    #     output = await AIOHttpRequests.form_post_async(url=url, files=files, encoding="json")
    #     print(output)
    #     return output
    #
    #
    # # url = "http://ai.8001.kookaat.ir/predict/single?model_name=word"
    # # filepath = "/home/ai/Desktop/sample-ocr.png"
    # # output = RequestsUtils.post_file(url, "image", filepath)
    # # print(output)
    # import asyncio
    #
    # asyncio.run(port())
