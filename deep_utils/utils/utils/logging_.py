import os
import logging
import sys
from typing import Union


def get_logger(name: str, write_logs=False, log_path: Union[str, None] = None) -> logging.Logger:
    """
    Creates a logger for a given name
    :param name: The name that logger will be created for
    :param write_logs: whether write logs to the hard drive or not.
    :param log_path: save address for logger
    :return:
    """
    if write_logs and log_path is None:
        raise ValueError(f"write_logs is set to True, provide a log_path")
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if write_logs:
        log_dir, file_name = os.path.split(log_path)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    print(f"[INFO] Successfully create logger for {name}")
    return logger
