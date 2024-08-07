import inspect
import logging
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Callable, Union

LOGGING_FORMATS = {
    "basic": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "func_name": "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
}


def start_end_logger_decorator(func) -> Callable:
    """
    Logs the start and end of a func
    :param func: The func that will be measured
    :return:
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger_ = kwargs.get("logger", None)
        verbose = kwargs.get("verbose", 1)
        file_path = inspect.getfile(func)
        func_name = func.__name__
        log_print(
            logger_,
            f"{file_path} -- func: {func_name} --> message: Starting...",
            verbose=verbose,
            get_func_name=False,
        )
        results = func(*args, **kwargs)
        log_print(
            logger_,
            f"{file_path} -- func: {func_name} --> message: Done...",
            verbose=verbose,
            get_func_name=False,
        )
        return results

    return wrapper


def get_logger(
        name: str,
        log_path: Union[str, Path, None] = None,
        remove_previous_handlers=True,
        logging_format: str = "basic",
) -> logging.Logger:
    """
    Creates a logger for a given name,
    :param name: The name that logger will be created for
    :param log_path: Where logger information will be saved
    :param remove_previous_handlers: If set to true, removes previous handlers to prevent repeated prints
    :param logging_format: What format to use, default is basic. Check LOGGING_FORMATS dict for available formats.
    :return:
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(LOGGING_FORMATS[logging_format])

    if log_path:
        log_dir, file_name = os.path.split(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if remove_previous_handlers:
        logger.handlers = logger.handlers[-1:]
    logger.setLevel(logging.INFO)
    print(
        f"[INFO] Successfully created logger for {name}"
        + (f" in {log_path}" if log_path else "")
    )
    return logger


def func_log(message, roll_back=1):
    """Automatically log the current function details."""

    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    code_f_back = inspect.currentframe()
    for _ in range(roll_back):
        code_f_back = code_f_back.f_back
    func = code_f_back.f_code
    # Dump the message + the name of this function to the log.
    message = f"message: {message} --> file: {func.co_filename} -- line: {func.co_firstlineno} -- func: {func.co_name}"
    return message


def log_print(
        logger: Union[None, logging.Logger] = None,
        message: str = "",
        log_type="info",
        verbose=1,
        get_func_name=True,
        roll_back=2,
):
    """
    Logs the input messages with the given log_type. In case the logger object is not provided, prints the message.
    :param logger:
    :param message:
    :param log_type: possible log types: info, error. Default is set to info
    :param verbose: whether to print/log
    :param get_func_name: whether to include func-name in the message
    :param roll_back: roll back used for func-log. This number indicates how many functions should `func-log` trace back
    to get to the main func
    :return:
    """
    if get_func_name:
        message = func_log(message, roll_back=roll_back)
    if log_type == "info":
        if logger is not None and isinstance(logger, logging.Logger):
            logger.info(message)
        else:
            if verbose:
                print(f"[INFO] {message}")
    elif log_type == "error":
        if logger is not None and isinstance(logger, logging.Logger):
            logger.error(message)
        else:
            if verbose:
                print(f"[ERROR] {message}")
    else:
        if verbose:
            print(f"[ERROR] log_type: {log_type} is not supported")
        raise ValueError("[ERROR] log_type: {log_type} is not supported")


def value_error_log(logger: Union[None, logging.Logger], message: str):
    """
    generates a value error and logs it!
    :param logger:
    :param message:
    :return:
    """
    log_print(logger, message, log_type="error")
    raise ValueError(message)


def save_params(param_path, args, logger=None):
    """
    Save the arguments in the given path.
    Args:
        param_path:
        args:
        logger: If provided the message will be logged unless will be printed to console!

    Returns:

    """
    log_print(logger, f"Saving params!")
    with open(param_path, mode="w") as f:
        arguments = vars(args)
        for key, val in arguments.items():
            f.write(f"{key} {val}\n")
    log_print(logger, f"Params are successfully saved in {param_path}!")


def get_conf_matrix(
        class_name_map,
        y_pred,
        y_true,
        save_path=None,
        conf_csv_name="conf_matrix.csv",
        conf_jpg_name="conf_matrix.jpg",
        logger=None,
):
    """
    Computes config matrix and saves the csv and jpg file if the save_path is provided!
    Args:
        class_name_map:
        y_pred:
        y_true:
        save_path:
        conf_csv_name:
        conf_jpg_name:
        logger: If provided the message will be logged unless will be printed to console!

    Returns: configuration matrix

    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    conf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        conf_matrix,
        index=list(class_name_map.keys()),
        columns=list(class_name_map.keys()),
    )
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="g")
    plt.xlabel("")
    plt.ylabel("")
    if save_path is not None:
        df_cm.to_csv(os.path.join(save_path, conf_csv_name))
        plt.savefig(os.path.join(save_path, conf_jpg_name))
    log_print(logger, "Confusion matrix is successfully generated!")
    return conf_matrix


def get_cls_report(y_pred, y_true, save_path=None, logger=None):
    """
    Generate classification report and saves them if the save_path is provided!
    Args:
        y_pred:
        y_true:
        save_path:
        logger: If provided the message will be logged unless will be printed to console!

    Returns:

    """
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred)
    if save_path:
        with open(save_path, mode="w") as f:
            f.write(report)
    log_print(logger, "Successfully generated classification report")
    return report


if __name__ == "__main__":
    l = get_logger("p", log_path="log.log")
    log_print(l, "wow", verbose=1)
