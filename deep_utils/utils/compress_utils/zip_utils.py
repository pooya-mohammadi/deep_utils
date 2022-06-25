import os
from os.path import join


def unzip(zip_path, zip_dir="."):
    from zipfile import ZipFile

    with ZipFile(zip_path, "r") as zip:
        print(f"extracting {zip_path}")
        zip.extractall(zip_dir)
        print(f"extracting is done!")


def unzip_dir_zip(dir_path, remove_zip_file=False):
    """
    Unzip a directory of zip files
    :param dir_path: path to directory that contains zip files
    :param remove_zip_file: whether to remove zip-file, default is False.
    :return:
    """
    for base, directories, file_names in os.walk(dir_path):
        for file_name in file_names:
            if file_name.endswith(".zip"):
                zip_path = join(base, file_name)
                unzip(zip_path, zip_dir=base)
                if remove_zip_file:
                    os.system(f'rm -rf "{zip_path}"')
