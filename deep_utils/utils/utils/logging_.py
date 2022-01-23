import os
import logging
import sys
from typing import Union


def get_logger(name: str, log_path: Union[str, None] = None) -> logging.Logger:
    """
    Creates a logger for a given name
    :param name: The name that logger will be created for
    :param log_path: In which logger information will be saved
    :return:
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        log_dir, file_name = os.path.split(log_path)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    print(f"[INFO] Successfully create logger for {name}")
    return logger
