from logging import Logger
from typing import Union
from deep_utils.utils.utils.logging_ import log_print, value_error_log


def minio_get(minio_client, bucket_name, object_name, logger: Union[None, Logger]):
    """
    Get object from minio_client. This function is created for compatibility otherwise no extra functionality provided compared to the main module.
    :param minio_client:
    :param bucket_name:
    :param object_name:
    :param logger:
    :return:
    """
    obj = minio_client.get_object(bucket_name, object_name)
    log_print(logger, message=f"Successfully got object: {object_name} from bucket: {bucket_name}")
    return obj


def minio_fget(minio_client, bucket_name, object_name, file_path, logger: Union[None, Logger]):
    """
    Get file object from minio_client. This function is created for compatibility otherwise no extra functionality provided compared to the main module.
    :param minio_client:
    :param bucket_name:
    :param object_name:
    :param file_path: Where to save the file
    :param logger:
    :return:
    """
    obj = minio_client.fget_object(bucket_name, object_name, file_path)
    log_print(logger, message=f"Successfully got object: {object_name} from bucket: {bucket_name}")
    return obj


def minio_put(minio_client, bucket_name, object_name, data, create=True, logger: Union[Logger, None] = None):
    """
    put an object inside a bucket
    :param minio_client:
    :param bucket_name:
    :param object_name:
    :param data: data-file to be stored in
    :param create: create te bucket in case it doesn't exist, default is True.
    :param logger: a logger instance
    :return:
    """
    found = minio_client.bucket_exists(bucket_name)
    if found:
        log_print(logger, f"Bucket {bucket_name} already exists")
    elif not found and create:
        minio_client.make_bucket(bucket_name)
        log_print(logger, f"Successfully Created bucket: {bucket_name}")
    else:
        value_error_log(logger, 'f"Error in creating minio bucket"')

    length = len(data.read())
    data.seek(0)

    result = minio_client.put_object(
        bucket_name,
        object_name,
        data,
        length,
    )
    log_print(logger, f"Successfully put object: {object_name} into bucket: {bucket_name}")
    return result


def minio_fput(minio_client, bucket_name, object_name, file_path, create=True, logger: Union[Logger, None] = None):
    """
    put a file inside a bucket
    :param minio_client:
    :param bucket_name:
    :param object_name:
    :param file_path: file to be stored in the given bucket
    :param create: create te bucket in case it doesn't exist, default is True.
    :param logger: a logger instance
    :return:
    """
    found = minio_client.bucket_exists(bucket_name)
    if found:
        log_print(logger, f"Bucket {bucket_name} already exists")
    elif not found and create:
        minio_client.make_bucket(bucket_name)
        log_print(logger, f"Successfully Created bucket: {bucket_name}")
    else:
        value_error_log(logger, 'f"Error in creating minio bucket"')

    result = minio_client.fput_object(
        bucket_name,
        object_name,
        file_path,
    )
    log_print(logger, f"Successfully put file: {object_name} into bucket: {bucket_name}")
    return result
