import os
import sys
from functools import wraps

import requests

from deep_utils.utils.logging_utils.logging_utils import log_print, value_error_log


def get_file(fname, origin, cache_dir=None):
    base_dir, fname = os.path.split(fname)

    if cache_dir is None:
        if "DEEP_UTILS_HOME" in os.environ:
            cache_dir = os.environ.get("DEEP_UTILS_HOME")
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache/deep_utils")

    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, base_dir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)
    if fpath.endswith(".zip"):
        check_path = fpath[:-4]
    else:
        check_path = fpath
    download = False if os.path.exists(check_path) else True

    if download:
        print("Downloading data from", origin)
        error_msg = "URL fetch failure on {}"
        try:
            with open(fpath, "wb") as f:
                response = requests.get(origin, stream=True)
                total = response.headers.get("content-length")

                if total is None:
                    f.write(response.content)
                else:
                    downloaded = 0
                    total = int(total)
                    for data in response.iter_content(
                            chunk_size=max(int(total / 1000), 1024 * 1024)
                    ):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total)
                        sys.stdout.write(
                            "\r[{}{}]".format("█" * done, "." * (50 - done))
                        )
                        sys.stdout.flush()
            sys.stdout.write("\n")
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise Exception(error_msg.format(origin))
        if fpath.endswith(".zip"):
            from zipfile import ZipFile

            try:
                with ZipFile(fpath, "r") as zip:
                    zip.printdir()
                    print(f"extracting {fpath}")
                    zip.extractall(fpath[:-4])
                    print(f"extracting is done!")
                os.remove(fpath)
            except (Exception, KeyboardInterrupt):
                raise Exception(f"Error in extracting {fpath}")
    if fpath.endswith(".zip"):
        fpath = fpath[:-4]
    return fpath


def download_file(
        url,
        download_dir=".",
        unzip=False,
        remove_zip=False,
        file_name=None,
        logger=None,
        remove_download=False,
):
    if url is None:
        value_error_log(logger, message="url is None. Exiting the function")

    os.makedirs(download_dir, exist_ok=True)
    log_print(logger, f"Downloading data from {url}")
    error_msg = "URL fetch failure on {}"
    download_des = None
    try:
        response = requests.get(url, stream=True)
        total = response.headers.get("content-length")
        while not total:
            response = requests.get(url, stream=True)
            total = response.headers.get("content-length")
        try:
            if file_name is None:
                file_name = response.headers.get("filename")
                if file_name is None:
                    file_name = response.headers.get("content-disposition").split("=")[
                        -1
                    ]
        except:
            file_name = os.path.split(url)[-1]

        download_des = os.path.join(download_dir, file_name)
        with open(download_des, "wb") as f:
            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)
                for data in response.iter_content(
                        chunk_size=max(int(total / 1000), 1024 * 1024)
                ):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total)
                    sys.stdout.write("\r[{}{}]".format(
                        "█" * done, "." * (50 - done)))
                    sys.stdout.flush()
        sys.stdout.write("\n")
    except (Exception, KeyboardInterrupt):
        if (
                download_des is not None
                and os.path.isfile(download_des)
                and remove_download
        ):
            log_print(logger, f"ًRemoving {download_des} due to error")
            os.remove(download_des)
        log_print(logger, error_msg.format(url), log_type="error")
        raise Exception(error_msg.format(url))
    if unzip:
        unzip_func(download_des, download_dir, remove_zip, logger)
    return download_des


def unzip_func(zip_file, res_dir, remove_zip=False, logger=None):
    """
    A file to unzip zip files!
    Args:
        zip_file:
        res_dir:
        remove_zip:
        logger:

    Returns:

    """
    from zipfile import ZipFile

    with ZipFile(zip_file, "r") as zip:
        zip.extractall(res_dir)
        log_print(logger, f"Successfully extracted {zip_file} to {res_dir}!")
    if zip_file is not None and remove_zip:
        os.remove(zip_file)
        log_print(logger, f"Successfully removed {zip_file}!")


def download_decorator(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        config = self.config
        download_variables = self.download_variables
        for var in download_variables:
            if getattr(config, var) is None:
                url = getattr(config, var + "_url")
                f_name = getattr(config, var + "_cache")
                f_path = get_file(f_name, url)
                setattr(config, var, f_path)
        return func(self, *args, **kwargs)

    return wrapper


if __name__ == "__main__":
    url = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
    download_file(url, "/home/ai/pooya", unzip=True, remove_zip=False)
    download_file(
        url, "/home/ai/pooya", unzip=True, remove_zip=False, file_name="wow.zip"
    )
    download_file(
        "https://github.com/Practical-AI/deep_utils/archive/refs/tags/0.4.2.zip",
        "/home/ai/pooya",
        unzip=True,
        remove_zip=True,
    )
    download_file(
        "https://github.com/Practical-AI/deep_utils/archive/refs/tags/0.4.2.zip",
        "/home/ai/pooya",
        unzip=False,
        remove_zip=True,
    )
