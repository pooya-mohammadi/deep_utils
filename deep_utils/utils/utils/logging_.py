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
    print(f"[INFO] Successfully created logger for {name}")
    return logger


def log_print(logger: Union[None, logging.Logger], message: str, log_type="info"):
    """
    Logs the input messages with the given log_type. In case the logger object is not provided, prints the message.
    :param logger:
    :param message:
    :param log_type:
    :return:
    """
    if log_type == 'info':
        if logger is not None and isinstance(logger, logging.Logger):
            logger.info(message)
        else:
            print(f'[INFO] {message}')
    elif log_type == 'error':
        if logger is not None and isinstance(logger, logging.Logger):
            logger.error(message)
        else:
            print(f'[ERROR] {message}')
    else:
        print(f'[ERROR] log_type: {log_type} is not supported')
        raise ValueError("[ERROR] log_type: {log_type} is not supported")


def value_error_log(logger: Union[None, logging.Logger], message: str):
    """
    generates a value error and logs it!
    :param logger:
    :param message:
    :return:
    """
    log_print(logger, message, log_type='error')
    raise ValueError(message)
