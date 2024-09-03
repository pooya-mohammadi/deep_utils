import json
from typing import Union
from pathlib import Path

from deep_utils.utils.decorators.main import method_deprecated


class JsonUtils:
    @staticmethod
    def load(json_path: Union[str, Path], encoding="utf-8") -> dict:
        """
        loads a json file
        :param json_path: Path to json file to load
        :param encoding: encoding format
        :return: returns a json file
        """

        with open(json_path, encoding=encoding) as f:
            json_data = json.load(f)
        return json_data

    @staticmethod
    def dump(
            json_path: Union[str, Path], json_object: Union[list, dict], encoding="utf-8", ensure_ascii=True
    ) -> None:
        """
        dumps a json file
        Args:
            json_object:
            json_path: path to json file
            encoding: encoding format
            ensure_ascii: set to False for persian characters

        Returns:

        """

        with open(json_path, mode="w", encoding=encoding) as f:
            json.dump(json_object, f, ensure_ascii=ensure_ascii)


JsonUtils.load_json = method_deprecated(JsonUtils.load)
JsonUtils.dump_json = method_deprecated(JsonUtils.dump)
