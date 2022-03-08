import os
import logging
import sys
from pathlib import Path
from typing import Union


def get_logger(name: str, log_path: Union[str, Path, None] = None) -> logging.Logger:
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


def log_print(logger: Union[None, logging.Logger], message: str, log_type="info", verbose=1):
    """
    Logs the input messages with the given log_type. In case the logger object is not provided, prints the message.
    :param logger:
    :param message:
    :param log_type:
    :param verbose: whether to print/log or not!
    :return:
    """
    if log_type == 'info':
        if logger is not None and isinstance(logger, logging.Logger):
            logger.info(message)
        else:
            if verbose:
                print(f'[INFO] {message}')
    elif log_type == 'error':
        if logger is not None and isinstance(logger, logging.Logger):
            logger.error(message)
        else:
            if verbose:
                print(f'[ERROR] {message}')
    else:
        if verbose:
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
    with open(param_path, mode='w') as f:
        arguments = vars(args)
        for key, val in arguments.items():
            f.write(f"{key} {val}\n")
    log_print(logger, f"Params are successfully saved in {param_path}!")


def get_conf_matrix(class_name_map, y_pred, y_true,
                    save_path=None,
                    conf_csv_name="conf_matrix.csv",
                    conf_jpg_name="conf_matrix.jpg",
                    logger=None):
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
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    conf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=list(class_name_map.keys()), columns=list(class_name_map.keys()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
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
        with open(save_path, mode='w') as f:
            f.write(report)
    log_print(logger, 'Successfully generated classification report')
    return report
