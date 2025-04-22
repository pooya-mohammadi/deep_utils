import json
import math
import os
import re
from ast import literal_eval
from typing import Union, List


class PromptUtils:
    def __init__(self):
        """
        Creates a prompt generator
        """

    @staticmethod
    def get_segments(strings: str, segment_limit) -> list:
        """
        Remove me if I'm not used anywhere!
        :param strings:
        :param segment_limit:
        :return:
        """
        # encoding = tiktoken.encoding_for_model("text-davinci-003")
        # num_segments = len(encoding.encode(strings))
        # Adjust as per your needs
        print(f"Generating video script for {len(strings)} characters of text")
        print(segment_limit)

        num_segments = math.ceil(len(strings) / segment_limit)
        print(num_segments)
        segments = [strings[i * segment_limit:(i + 1) * segment_limit] for i in range(0, num_segments)]
        print('len', len(segments))
        return segments

    @staticmethod
    def load_txt_prompt(path: str, replace_new_line=False) -> str:
        """
            Loads the txt prompt from the given path.
            the new lines that are in txt file are skipped and only those with [new-line] are considered as new line.
            :param path: path to prompt txt file
            :param replace_new_line: if True, replaces [new-line] with \n
            :return:
        """
        if os.path.isfile(path):
            txt = open(path, mode="r", encoding="utf-8").read()
        else:
            txt = path
        if replace_new_line:
            txt = txt.replace("\n", "").replace("[new-line]", "\n")
        return txt

    @staticmethod
    def replace_comments(txt: str) -> str:
        pattern = r"#.*?#"
        flag = re.DOTALL
        txt = re.sub(pattern, "", txt, flags=flag)
        return txt

    @staticmethod
    def fill_comments(text: str, **variables) -> str:
        """
        I believe this is not used anywhere anymore!
        :param text:
        :param variables:
        :return:
        """
        pattern_values = r'#(.*?)#'
        pattern_names = r'{(.*?)}'

        commented_texts = re.findall(pattern_values, text)
        for commented_text in commented_texts:
            variable_names = re.findall(pattern_names, commented_text)
            processed_text = commented_text
            flag = True if len(variable_names) else False
            for variable_name in variable_names:
                variable_value = variables.get(variable_name, None)
                if not variable_value:
                    flag = False
                    break
                processed_text = processed_text.replace("{" + f"{variable_name}" + "}", str(variable_value))
            if flag:
                # replace comment with data if all the variables are provided by user!
                text = text.replace("#" + f"{commented_text}" + "#", processed_text)

        return text

    @staticmethod
    def fill_variables(txt, **variables) -> str:
        """
        fills the curly brackets in the prompts!
        :param txt:
        :param variables:
        :return:
        """
        pattern = r"{(\w+):-([^'}\s]*)}"
        matches = re.findall(pattern, txt)
        processed_text = txt
        for variable_name, default_value in matches:
            value = variables.pop(variable_name) if variable_name in variables else default_value
            # replace the variable with the value for the default variables and non defaults
            processed_text = processed_text.replace("{" + variable_name + ":-" + default_value + "}", value)
            # non defaults
            processed_text = processed_text.replace("{" + f"{variable_name}" + "}", str(value))
        # match parameters without default values
        pattern = r"{(\w+)}"
        matches = set(re.findall(pattern, txt))
        for variable_name in matches:
            if variable_name not in variables:
                raise ValueError(f"variable {variable_name} is not provided!")
            value = variables.pop(variable_name)
            processed_text = processed_text.replace("{" + f"{variable_name}" + "}", str(value))
        if len(variables):
            raise ValueError(f"variables {variables} are not used!")
        return processed_text

    @staticmethod
    def get_bulleted_list(video_description_list):
        video_description_list = [f"• {video_description}" for video_description in video_description_list]
        video_description_list = "\n".join(video_description_list)
        return video_description_list

    @staticmethod
    def parse_markdown(input_string: str, keyword="json", load_mechanism="json", leading_str="```",
                       trailing_str="```", get_list_inner_eval=False,
                       remove_patterns: Union[str, List[str], None] = None) -> Union[dict, list, str]:
        """
        Parse a JSON string from a Markdown string.

        Args:
            input_string: The Markdown string.
            keyword: The keyword to look for in the Markdown string.
            load_mechanism: The mechanism to use to load the string. Can be either "json" or None.
            leading_str: The leading string to look for in the Markdown string.
            trailing_str: The trailing string to look for in the Markdown string.
            get_list_inner_eval: Whether to evaluate the inner items of a list. For example, if there is an item with
                value "['1', '2', '3']", this will be converted to a list of integers.
            remove_patterns: A list of patterns to remove from the input string.
        Returns:
            The parsed JSON object as a Python dictionary.
        """
        # Try to find JSON string within triple backticks
        match = re.search(rf"{leading_str}({keyword})?(.*?){trailing_str}", input_string, re.DOTALL)

        if leading_str and trailing_str and keyword and match is None:
            # if match is None but trailing_str is provided, I add the trailing str.
            # Sometimes the final trailing str is missing!
            if not input_string.endswith(trailing_str):
                input_string = input_string + trailing_str
            match = re.search(rf"{leading_str}({keyword})?(.*?){trailing_str}", input_string, re.DOTALL)
            # if match is still None add the leading str with keyword
            if match is None and not input_string.startswith(leading_str + keyword):
                if input_string.startswith(keyword):
                    input_string = leading_str + input_string
                else:
                    input_string = leading_str + keyword + input_string
                match = re.search(rf"{leading_str}({keyword})?(.*?){trailing_str}", input_string, re.DOTALL)

        # If no match found, assume the entire string is a JSON string
        if match is None:
            input_str = input_string
        else:
            # If match found, use the content within the backticks
            input_str = match.group(2)

        # Strip whitespace and newlines from the start and end
        input_str = input_str.strip()

        # Remove patterns
        if remove_patterns is not None:
            if isinstance(remove_patterns, str):
                remove_patterns = [remove_patterns]
            for pattern in remove_patterns:
                input_str = re.sub(pattern, "", input_str)

        # Parse the JSON string into a Python dictionary
        if load_mechanism == "json":
            parsed = PromptUtils.replace_and_parse_json(input_str)
        elif load_mechanism == "list":
            input_str = PromptUtils.extract_text_between_first_and_last_brackets(input_str)
            parsed = PromptUtils.replace_and_parse_list(input_str)
            # sometimes it only returns a string instead of a list then convert it to list manually
            parsed = [parsed] if isinstance(parsed, str) else parsed
            if get_list_inner_eval:
                parsed = [literal_eval(item) for item in parsed]
            # remove ellipsis
            parsed = [item for item in parsed if not isinstance(item, type(Ellipsis))]
        elif load_mechanism in [None, ""]:
            parsed = input_str
        else:
            raise ValueError(f"Invalid load_mechanism: {load_mechanism}")

        return parsed

    @staticmethod
    def extract_text_between_first_and_last_brackets(text: str):
        """
        Extract based on brackets
        :param text:
        :return:
        """
        start_index = text.find('[')
        if start_index == -1:
            start_index = 0
            text = "[" + text
        end_index = text.rfind(']')
        if end_index == -1:
            end_index = len(text)
            text = text + "]"
        return text[start_index:end_index + 1]

    @staticmethod
    def replace_double_quotation_with_triple_quotation(text: str):
        """
        "This is "pooya" who is playing!" This cannot be parsed with string
        Converting the first and the last " to triple "
        :param text:
        :return:
        """
        start_index = text.find('"')
        if start_index == -1:
            start_index = 0
            text = '"' + text
        end_index = text.rfind('"')
        if end_index == -1:
            end_index = len(text)
            text = text + '"'

        text = text[:start_index] + '"""' + text[start_index + 1: end_index] + '"""' + text[end_index + 1:]
        return text

    @staticmethod
    def replace_double_quotation_with_triple_quotation_all(input_str: str):
        lines = input_str.split("\n")
        output = []
        for line in lines:
            if '"' in line:
                line = PromptUtils.replace_double_quotation_with_triple_quotation(line)
            output.append(line)
        return "\n".join(output)

    @staticmethod
    def replace_and_parse_list(input_str: str) -> Union[str, list[str]]:
        try:
            # simple parse
            parsed = literal_eval(input_str)
            return parsed
        except:
            pass

        try:
            # Convert last "]] to ."]
            parsed = literal_eval(input_str.replace("\n", "").replace('] ]', ']'))
            return parsed
        except:
            pass

        try:
            # Convert last "]] to ."]
            parsed = literal_eval(input_str.replace("\n", "").replace(']]', ']'))
            return parsed
        except:
            pass

        try:
            # Convert last ".] to ."]
            parsed = literal_eval(input_str.replace('".]', '."]'))
            return parsed
        except:
            pass

        try:
            # maybe we have ... without comma!
            parsed = literal_eval(input_str.replace("...,", "").replace("...", ""))
            return parsed
        except:
            pass

        # maybe we have double quotation error!
        try:
            parsed = literal_eval(PromptUtils.replace_double_quotation_with_triple_quotation_all(input_str))
            return parsed
        except:
            pass

        return input_str
        # raise ValueError("Couldn't parse!")

    @staticmethod
    def replace_and_parse_json(input_str: str) -> Union[str, list]:
        try:
            parsed = json.loads(input_str)
            return parsed
        except:
            pass

        # seems that literal_eval is working quite nice!
        try:
            parsed = literal_eval(input_str)
            return parsed
        except:
            pass
        try:
            # let's remove new lines :D
            parsed = literal_eval(input_str.replace("\n", ""))
            return parsed
        except:
            pass
        raise ValueError(f"Couldn't parse: {input_str}")
