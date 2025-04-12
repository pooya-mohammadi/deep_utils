import os
import shutil
from os.path import join, split
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

from deep_utils.utils.logging_utils.logging_utils import log_print, value_error_log
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.shutil_utils.shutil_utils import mv_or_copy


def transfer_directory_items(
        in_dir,
        out_dir,
        transfer_list,
        mode="cp",
        remove_out_dir=False,
        skip_transfer=False,
        remove_in_dir=False,
):
    """

    Args:
        in_dir:
        out_dir:
        transfer_list:
        mode:
        remove_out_dir:
        skip_transfer: If the file does not exist, skip and do not raise Error.
        remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!

    Returns:

    """
    print(f"[INFO] Starting to copying/moving from {in_dir} to {out_dir}")
    remove_create(out_dir, remove=remove_out_dir)
    if mode == "cp":
        for name in transfer_list:
            try:
                shutil.copy(os.path.join(in_dir, name), out_dir)
            except FileNotFoundError as e:
                if skip_transfer:
                    print("[INFO] shutil.copy did not find the file, skipping...")
                else:
                    raise FileNotFoundError(e)
    elif mode == "mv":
        for name in transfer_list:
            try:
                shutil.move(os.path.join(in_dir, name), out_dir)
            except FileNotFoundError as e:
                if skip_transfer:
                    print("[INFO] shutil.move did not find the file, skipping...")
                else:
                    raise FileNotFoundError()
        if remove_in_dir:
            shutil.rmtree(in_dir)
    else:
        raise ValueError(
            f"{mode} is not supported, supported modes: mv and cp")
    print(f"finished copying/moving from {in_dir} to {out_dir}")


def dir_train_test_split(
        in_dir,
        train_dir="./train",
        val_dir="./val",
        test_size=0.1,
        mode="cp",
        remove_out_dir=False,
        skip_transfer=False,
        remove_in_dir=False,
        skip_error=True,
        ignore_list: List[str] = None,
        logger=None,
        verbose=1
):
    """

    :param in_dir:
    :param train_dir:
    :param val_dir:
    :param test_size:
    :param mode:
    :param remove_out_dir:
    :param skip_transfer: If the file does not exist, skip and do not raise Error
    :param remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!
    :param skip_error: If set to True, skips the train_test_split error and returns empty lists
    :param ignore_list: a list of names that are ignored
    :param logger:
    :param verbose:
    :return:
    """
    from sklearn.model_selection import train_test_split
    log_print(logger, f"Starting to split dir: {in_dir}", verbose=verbose)
    if ignore_list is not None:
        list_ = [n for n in os.listdir(in_dir) if n not in ignore_list]
    else:
        list_ = os.listdir(in_dir)
    try:
        train_name, val_name = train_test_split(list_, test_size=test_size)
    except ValueError as e:
        message = f"Couldn't split the data in {in_dir}: {e}"
        if skip_error:
            log_print(logger, message=message, log_type="error")
            return [], []
        else:
            value_error_log(logger, message=message)
    transfer_directory_items(
        in_dir,
        train_dir,
        train_name,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=False,
    )
    transfer_directory_items(
        in_dir,
        val_dir,
        val_name,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=remove_in_dir,
    )
    log_print(logger, f"Finished splitting dir: {in_dir}", verbose=verbose)
    return train_name, val_name


def split_dir_of_dir(
        in_dir,
        train_dir="./train",
        val_dir="./val",
        test_size=0.1,
        mode="cp",
        remove_out_dir=False,
        remove_in_dir=False,
):
    """

    Args:
        in_dir:
        train_dir:
        val_dir:
        test_size:
        mode:
        remove_out_dir:
        remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!

    Returns:

    """
    if remove_out_dir:
        remove_create(train_dir)
        remove_create(val_dir)
    for data in os.listdir(in_dir):
        dir_ = join(in_dir, data)
        if dir_ in [train_dir, val_dir]:
            print(
                f"[INFO] {dir_} is equal to {val_dir} or {train_dir}, Skipping ...")
            continue
        if not os.path.isdir(dir_):
            print(f"[INFO] {dir_} is not a directory, Skipping ...")
            continue
        if len(os.listdir(dir_)) == 0:
            print(f"[INFO] {dir_} is empty, Skipping ...")
            continue
        dir_train_test_split(
            dir_,
            train_dir=join(train_dir, data),
            val_dir=join(val_dir, data),
            mode=mode,
            test_size=test_size,
            remove_out_dir=remove_out_dir,
            remove_in_dir=remove_in_dir,
        )
    if mode == "mv" and remove_in_dir:
        shutil.rmtree(in_dir)


def crawl_directory_dataset(
        dir_: str,
        ext_filter: list = None,
        map_labels=False,
        label_map_dict: dict = None,
        logger=None,
        verbose=1,
) -> Union[Tuple[List[str], List[int]], Tuple[List[str], List[int], Dict]]:
    """
    crawls a directory of classes and returns the full path of the items paths and their class names
    :param dir_: path to directory of classes
    :param ext_filter: extensions that will be passed and others will be dropped
    :param map_labels: This map the labels to indices
    :param label_map_dict: A map which is used for filtering and also mapping the labels!
    :param logger: A logger
    :param verbose: whether print logs or not!
    :return: Tuple[List[str], List[int], Dict], x_list containing the paths, y_list containing the class_names, and
    label_map dictionary.
    """
    print(f"[INFO] beginning to crawl {dir_}")
    x, y = [], []
    label_map = dict()
    for cls_name in os.listdir(dir_):
        cls_path = join(dir_, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for item_name in os.listdir(cls_path):
            item_path = join(cls_path, item_name)
            name, ext = os.path.splitext(item_name)
            if ext_filter is not None and ext not in ext_filter:
                log_print(
                    logger,
                    f"{item_path} with {ext} is not in ext_filtering: {ext_filter}",
                    verbose=verbose,
                )
                continue

            if label_map_dict is not None:
                if cls_name in label_map_dict:
                    y.append(label_map_dict[cls_name])
                else:
                    log_print(logger, f"Skipping cls_name:{cls_name}, x: {x}")
                    continue
            elif map_labels:
                if cls_name not in label_map:
                    cls_index = len(label_map)
                    label_map[cls_name] = cls_index
                else:
                    cls_index = label_map[cls_name]
                y.append(cls_index)
            else:
                y.append(cls_name)
            x.append(item_path)
    log_print(logger, f"successfully crawled {dir_}", verbose=verbose)
    if map_labels:
        return x, y, label_map
    else:
        return x, y


def remove_create(dir_: str, remove=True, logger=None, verbose=1):
    """
    Removes and creates the input directory!
    :param dir_:
    :param remove: whether to remove the directory or not
    :param logger:
    :param verbose:
    :return:
    """
    import shutil

    if os.path.exists(dir_) and remove:
        shutil.rmtree(dir_)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
        log_print(logger, f"Successfully removed and created dir: {dir_}", verbose=verbose)
        return dir_
    raise ValueError("dir_ should be provided!")


def file_incremental(file_path: str, artifact_type="prefix", artifact_value=0, extra_punctuation="_",
                     add_artifact_value=False):
    """
    This function is used to increment a file's address with prefix or suffix values until it becomes unique
    :param file_path:
    :param artifact_type:
    :param artifact_value:
    :param extra_punctuation:
    :param add_artifact_value: If set to True, adds the artifact_value then checks its existence.
    :return:
    """
    dir_, filename = os.path.split(file_path)
    artifact_value = int(artifact_value)
    while True:
        if add_artifact_value:
            # Maybe someone requires their file to have the first artifact
            file_path = join(dir_, split_extension(filename,
                                                   artifact_type=artifact_type,
                                                   artifact_value=artifact_value,
                                                   extra_punctuation=extra_punctuation))
        if not os.path.exists(file_path):
            break

        if not add_artifact_value:
            file_path = join(dir_, split_extension(filename,
                                                   artifact_type=artifact_type,
                                                   artifact_value=artifact_value,
                                                   extra_punctuation=extra_punctuation))
        artifact_value += 1
    return file_path


def cp_mv_all(
        input_dir,
        res_dir,
        mode="cp",
        filter_ext: Union[list, tuple, str, None] = None,
        artifact_type="prefix",
        artifact_value=0,
        extra_punctuation="_",
        add_artifact_value=False,
        logger=None,
        verbose=1,

):
    """
    Move or Copy all the files in a directory to another one. In case any of the files had the same name as the target files,
     their name will be incremented using file_incremental
    :param input_dir:
    :param res_dir:
    :param mode:
    :param filter_ext:
    :return:
    """
    n = 0
    filter_ext = [filter_ext] if filter_ext is not None and isinstance(filter_ext, str) else filter_ext
    for f_name in os.listdir(input_dir):
        _, ext = split_extension(f_name)
        if filter_ext is not None and ext not in filter_ext:
            continue
        f_in_path = os.path.join(input_dir, f_name)
        f_out_path = os.path.join(res_dir, f_name)
        f_out_path = file_incremental(f_out_path, artifact_type=artifact_type, artifact_value=artifact_value,
                                      extra_punctuation=extra_punctuation, add_artifact_value=add_artifact_value)

        mv_or_copy(f_in_path, f_out_path, mode=mode)
        n += 1
    log_print(
        logger,
        f"Successfully {mode}-ed {n} items with filters: {filter_ext} from {input_dir} to {res_dir}",
        verbose=verbose,
    )


def split_segmentation_dirs(
        in_images,
        in_masks,
        out_train="./train",
        out_val="./val",
        image_dir_name="images",
        mask_dir_name="masks",
        img_ext=None,
        mask_ext=None,
        mode="cp",
        test_size=0.2,
        remove_out_dir=False,
        remove_in_dir=False,
        skip_transfer=False,
):
    from sklearn.model_selection import train_test_split

    if img_ext is None and mask_ext is None:
        in_image_list = os.listdir(in_images)
        in_mask_list = os.listdir(in_masks)
        in_mask_dict = {split_extension(i)[0]: i for i in in_mask_list}

        in_image_list_train, in_image_list_val = train_test_split(
            in_image_list, test_size=test_size
        )
        in_mask_list_train = [
            in_mask_dict[split_extension(i)[0]] for i in in_image_list_train
        ]
        in_mask_list_val = [
            in_mask_dict[split_extension(i)[0]] for i in in_image_list_val
        ]
    else:
        in_image_list = [i for i in os.listdir(
            in_images) if i.endswith(img_ext)]
        in_image_list_train, in_image_list_val = train_test_split(
            in_image_list, test_size=test_size
        )
        in_mask_list_train = [
            split_extension(i, extension=mask_ext) for i in in_image_list_train
        ]
        in_mask_list_val = [
            split_extension(i, extension=mask_ext) for i in in_image_list_val
        ]

    transfer_directory_items(
        in_images,
        join(out_train, image_dir_name),
        in_image_list_train,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=remove_in_dir,
    )
    transfer_directory_items(
        in_images,
        join(out_val, image_dir_name),
        in_image_list_val,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=remove_in_dir,
    )

    transfer_directory_items(
        in_masks,
        join(out_train, mask_dir_name),
        in_mask_list_train,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=remove_in_dir,
    )
    transfer_directory_items(
        in_masks,
        join(out_val, mask_dir_name),
        in_mask_list_val,
        mode=mode,
        remove_out_dir=remove_out_dir,
        skip_transfer=skip_transfer,
        remove_in_dir=remove_in_dir,
    )


def find_file(dir_path, name, ext=".ckpt", logger=None, verbose=1):
    """
    finds the closest file to the input name in the input dir_path
    :param dir_path:
    :param name:
    :param ext:
    :param logger:
    :param verbose:
    :return:
    """
    for file_name in os.listdir(dir_path):
        if file_name.startswith(name) and file_name.endswith(ext):
            log_print(
                logger, f"name: {file_name} found in dir: {dir_path}", verbose=verbose
            )
            return join(dir_path, file_name)
    log_print(
        logger,
        f"name: {name} not found in dir: {dir_path}",
        log_type="warning",
        verbose=verbose,
    )
    return None


def combine_directory_of_directories(dataset_dir, result_dir, remove_result_dir=True, logger=None, verbose=0):
    """
    Combine a directory of directories in which there are several samples, and they should be combined in a final
    directory
    :param dataset_dir:
    :param result_dir:
    :param remove_result_dir:
    :param logger:
    :param verbose:
    :return:
    """
    remove_create(result_dir, remove_result_dir)
    for directory_name in os.listdir(dataset_dir):
        directory = join(dataset_dir, directory_name)
        if os.path.isdir(directory):
            os.system(f'cp {directory}/* {result_dir}')
            log_print(logger, f'Successfully moved {directory} to {result_dir}!', verbose=verbose)


class DirUtils:
    @staticmethod
    def split_dir_of_dir(
            in_dir,
            train_dir="./train",
            val_dir="./val",
            test_size=0.1,
            mode="cp",
            remove_out_dir=False,
            remove_in_dir=False,
    ):
        """

        Args:
            in_dir:
            train_dir:
            val_dir:
            test_size:
            mode:
            remove_out_dir:
            remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!

        Returns:

        """
        if remove_out_dir:
            remove_create(train_dir)
            remove_create(val_dir)
        for data in os.listdir(in_dir):
            dir_ = join(in_dir, data)
            if dir_ in [train_dir, val_dir]:
                print(
                    f"[INFO] {dir_} is equal to {val_dir} or {train_dir}, Skipping ...")
                continue
            if not os.path.isdir(dir_):
                print(f"[INFO] {dir_} is not a directory, Skipping ...")
                continue
            if len(os.listdir(dir_)) == 0:
                print(f"[INFO] {dir_} is empty, Skipping ...")
                continue
            dir_train_test_split(
                dir_,
                train_dir=join(train_dir, data),
                val_dir=join(val_dir, data),
                mode=mode,
                test_size=test_size,
                remove_out_dir=remove_out_dir,
                remove_in_dir=remove_in_dir,
            )
        if mode == "mv" and remove_in_dir:
            shutil.rmtree(in_dir)

    @staticmethod
    def write_txt(path: str | Path, list_content: list, mode="w"):
        with open(path, mode=mode) as f:
            for item in list_content:
                f.write(f"{item}\n")

    @staticmethod
    def read_txt(path: str | Path, mode="r"):
        list_content = [item.strip() for item in open(path, mode=mode).readlines()]
        return list_content

    @staticmethod
    def dir_train_test_split(
            in_dir,
            train_dir="./train",
            val_dir="./val",
            test_size=0.1,
            mode="cp",
            remove_out_dir=False,
            skip_transfer=False,
            remove_in_dir=False,
            skip_error=True,
            ignore_list: List[str] = None,
            logger=None,
            verbose=1
    ):
        """
        :param in_dir:
        :param train_dir:
        :param val_dir:
        :param test_size:
        :param mode:
        :param remove_out_dir:
        :param skip_transfer: If the file does not exist, skip and do not raise Error
        :param remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!
        :param skip_error: If set to True, skips the train_test_split error and returns empty lists
        :param ignore_list: a list of names that are ignored
        :param logger:
        :param verbose:
        :return:
        """
        from sklearn.model_selection import train_test_split
        log_print(logger, f"Starting to split dir: {in_dir}", verbose=verbose)
        if ignore_list is not None:
            list_ = [n for n in os.listdir(in_dir) if n not in ignore_list]
        else:
            list_ = os.listdir(in_dir)
        try:
            train_name, val_name = train_test_split(list_, test_size=test_size)
        except ValueError as e:
            message = f"Couldn't split the data in {in_dir}: {e}"
            if skip_error:
                log_print(logger, message=message, log_type="error")
                return [], []
            else:
                value_error_log(logger, message=message)
        transfer_directory_items(
            in_dir,
            train_dir,
            train_name,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=False,
        )
        transfer_directory_items(
            in_dir,
            val_dir,
            val_name,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=remove_in_dir,
        )
        log_print(logger, f"Finished splitting dir: {in_dir}", verbose=verbose)
        return train_name, val_name

    @staticmethod
    def split_extension(path,
                        extension: Union[str, None] = None,
                        suffix: Union[str, None] = None,
                        prefix: Union[str, None] = None,
                        artifact_type: Union[str, None] = None,
                        artifact_value: Union[str, int, None] = None,
                        extra_punctuation: Union[str, None] = None,
                        current_extension: Optional[str] = None
                        ):
        """
        split extension or add new suffix, prefix, or extension
        :param path:
        :param extension:
        :param suffix:
        :param prefix:
        :param artifact_type: prefix or suffix
        :param artifact_value: value for defined prefix or suffix
        :param extra_punctuation: the punctuation to be used before suffix or after prefix
        :param current_extension: the current extension of the file.
        :return:

        >>> split_extension("image.jpg", suffix="_res")
        'image_res.jpg'
        >>> split_extension("image.jpg")
        ('image', '.jpg')
        >>> split_extension("image.jpg", extension=".png")
        'image.png'
        >>> split_extension("image.jpg", extension="png")
        'image.png'
        >>> split_extension("image.jpg", extension="png", prefix="0_")
        '0_image.png'
        >>> split_extension("image.jpg", extension="png", suffix="_res", prefix="0_")
        '0_image_res.png'
        >>> split_extension("image.nii.gz", current_extension=".nii.gz", suffix="_crop")
        'image_crop.nii.gz'
        """

        return split_extension(path=path,
                               extension=extension,
                               suffix=suffix,
                               prefix=prefix,
                               artifact_type=artifact_type,
                               artifact_value=artifact_value,
                               extra_punctuation=extra_punctuation,
                               current_extension=current_extension, )

    @staticmethod
    def list_dir_full_path(directory: str,
                           filter_directories: bool = True,
                           interest_extensions: Optional[Union[str, List[str]]] = None,
                           only_directories: bool = False,
                           get_full_path: bool = True,
                           sort: bool = True,
                           not_exists_is_ok: bool = False,
                           dir_depth: int = -1,
                           exact_depth: bool = False,
                           return_dict: bool = False,
                           ends_with:Optional[Union[str, List[str]]] = None,
                           ) -> Union[List[str], Dict[str, str]]:
        """
        Returns the full path objects in a directory
        :param directory:
        :param filter_directories: If set to True, return on objects and not directories. This won't work,
        if only_directories is set to True,
        :param interest_extensions: If provided, files that have this extension will be chosen!
        :param only_directories: If set to True, only directories are extracted and filter_directories is ignored.
        :param get_full_path: If set to False, only the name is returned
        :param sort: If set to True, the directory will be sorted first!
        :param not_exists_is_ok: If set the True, and directory does not exist just returns an empty list,
         otherwise raises error.
        :param dir_depth: How depth the code should search, default is -1 which means deactivated.
        Only works when only_directories is set to True.
        :param exact_depth: If set True, the exact depth should be matched and smaller ones are not accepted!
        :param return_dict: If return_dict is set to True, the output will be a dict like the following: {filename: filepath}
        :param ends_with: If ends with these items they will be accepted
        :return:
        """
        interest_extensions = interest_extensions or []
        interest_extensions = [interest_extensions] if isinstance(interest_extensions, str) else interest_extensions
        interest_extensions = [f".{ext}" if not ext.startswith(".") else ext for ext in
                               interest_extensions]

        output = []
        if not os.path.exists(directory):
            if not_exists_is_ok:
                return output
            else:
                raise ValueError(f"Directory: {directory} does not exist!")
        if only_directories and dir_depth > 0:
            directory = "./" if directory == "." else directory
            for root, dirs, files in os.walk(directory):
                for dir_name in dirs:
                    if not DirUtils.endswith(dir_name, ends_with):
                        continue
                    current_dir_path = join(directory, root.replace(directory, '').lstrip("//"), dir_name.lstrip("//"))
                    relative_current_dir_path = join(root.replace(directory, ''), dir_name.lstrip("//"))
                    current_depth = len(relative_current_dir_path.strip("/").split("/"))
                    if exact_depth:
                        if current_depth == dir_depth:
                            output.append(current_dir_path)
                    else:
                        if current_depth <= dir_depth:
                            output.append(current_dir_path)
            output = sorted(output) if sort else output
        else:
            for filename in sorted(os.listdir(directory)) if sort else os.listdir(directory):
                if not DirUtils.endswith(filename, ends_with):
                    continue
                file_path = join(directory, filename)
                if not only_directories:
                    if filter_directories and os.path.isdir(file_path):
                        continue
                    if interest_extensions and DirUtils.split_extension(file_path)[1] not in interest_extensions:
                        continue
                else:
                    if not os.path.isdir(file_path):
                        continue

                output.append(file_path if get_full_path else filename)
            output = sorted(output) if sort else output
            if return_dict:
                if interest_extensions:
                    output = {DirUtils.remove_extension_with_replace(split(filepath)[-1], interest_extensions): filepath
                              for filepath in output}
                else:
                    output = {DirUtils.split_extension(split(filepath)[-1])[0]: filepath for filepath in output}
        return output

    @staticmethod
    def remove_extension_with_replace(filename: str, extensions: list[str]) -> str:
        """
        Removes extensions from the input filename
        :param filename:
        :param extensions:
        :return:
        """
        for extension in extensions:
            rev_ext = extension[::-1]
            filename = filename[::-1].replace(rev_ext, "", 1)[::-1]
        return filename

    @staticmethod
    def remove_create(dir_: str, remove=True, logger=None, verbose=0) -> str:
        """
    Removes and creates the input directory!
    :param dir_:
    :param remove: whether to remove the directory or not
    :param logger:
    :param verbose:
    :return:
    """
        return remove_create(dir_=dir_, remove=remove, logger=logger, verbose=verbose)

    @staticmethod
    def get_filename(file_path: str, remove_extension: Union[bool, str] = False) -> str:
        name: str = os.path.split(file_path)[-1]
        if remove_extension:
            if isinstance(remove_extension, str):
                name = name.rstrip(remove_extension)
            else:
                name, _ = DirUtils.split_extension(name)

        return name

    @staticmethod
    def crawl_directory_dataset(
            dir_: str,
            ext_filter: list | str = None,
            map_labels=False,
            label_map_dict: dict = None,
            logger=None,
            verbose=1,
    ) -> Union[Tuple[List[str], List[int]], Tuple[List[str], List[int], Dict]]:
        """
        crawls a directory of classes and returns the full path of the items paths and their class names
        :param dir_: path to directory of classes
        :param ext_filter: extensions that will be passed and others will be dropped
        :param map_labels: This map the labels to indices
        :param label_map_dict: A map which is used for filtering and also mapping the labels!
        :param logger: A logger
        :param verbose: whether print logs or not!
        :return: Tuple[List[str], List[int], Dict], x_list containing the paths, y_list containing the class_names, and
        label_map dictionary.
        """
        print(f"[INFO] beginning to crawl {dir_}")
        x, y = [], []
        label_map = dict()
        ext_filter = [ext_filter] if isinstance(ext_filter, str) else ext_filter
        for cls_name in os.listdir(dir_):
            cls_path = join(dir_, cls_name)
            if not os.path.isdir(cls_path):
                continue
            for item_name in os.listdir(cls_path):
                item_path = join(cls_path, item_name)
                name, ext = os.path.splitext(item_name)
                if ext_filter is not None and ext not in ext_filter:
                    log_print(
                        logger,
                        f"{item_path} with {ext} is not in ext_filtering: {ext_filter}",
                        verbose=verbose,
                    )
                    continue

                if label_map_dict is not None:
                    if cls_name in label_map_dict:
                        y.append(label_map_dict[cls_name])
                    else:
                        log_print(logger, f"Skipping cls_name:{cls_name}, x: {x}")
                        continue
                elif map_labels:
                    if cls_name not in label_map:
                        cls_index = len(label_map)
                        label_map[cls_name] = cls_index
                    else:
                        cls_index = label_map[cls_name]
                    y.append(cls_index)
                else:
                    y.append(cls_name)
                x.append(item_path)
        log_print(logger, f"successfully crawled {dir_}", verbose=verbose)
        if map_labels:
            return x, y, label_map
        else:
            return x, y

    @staticmethod
    def split_xy_dir(
            x_in_dir,
            y_in_dir,
            x_train_dir="train/samples",
            y_train_dir="train/targets",
            x_val_dir="val/samples",
            y_val_dir="val/targets",
            mode="cp",
            val_size=0.1,
            label_extension: str = None,
            img_suffix: str = None,
            lbl_suffix: str = None,
            skip_transfer=False,
            remove_out_dir=False,
            remove_in_dir=False,
    ):
        train_names, val_names = dir_train_test_split(
            x_in_dir,
            train_dir=x_train_dir,
            val_dir=x_val_dir,
            mode=mode,
            remove_out_dir=remove_out_dir,
            test_size=val_size,
        )
        if not img_suffix or not lbl_suffix:
            img_suffix, lbl_suffix = "", ""

        if label_extension is None:
            train_labels = [name.replace(img_suffix, lbl_suffix) for name in train_names]
            val_labels = [name.replace(img_suffix, lbl_suffix) for name in val_names]
        else:
            train_labels = [os.path.splitext(name.replace(img_suffix, lbl_suffix))[0] + label_extension for name in
                            train_names]
            val_labels = [os.path.splitext(name.replace(img_suffix, lbl_suffix))[0] + label_extension for name in
                          val_names]

        transfer_directory_items(
            y_in_dir,
            y_train_dir,
            train_labels,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=remove_in_dir,
        )
        transfer_directory_items(
            y_in_dir,
            y_val_dir,
            val_labels,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=remove_in_dir,
        )

    @staticmethod
    def endswith(filepath: str, ext: list[str] | str):
        """
        Checks whether a file ends with a list of strings! If ext is None it will return True!
        :param filepath:
        :param ext:
        :return:
        """
        if isinstance(ext, list):
            for ext_ in ext:
                if filepath.endswith(ext_):
                    return True
            return False
        elif isinstance(ext, str):
            return filepath.endswith(ext)
        elif ext is None:
            return True
        else:
            raise ValueError(f"ext: {ext} is not supported!")
    @staticmethod
    def file_incremental(file_path: str | None, artifact_type="prefix", artifact_value=0, extra_punctuation="_",
                         add_artifact_value=False, dir_items: list[str] | None = None):
        """
        This function is used to increment a file's address with prefix or suffix values until it becomes unique
        :param file_path:
        :param artifact_type:
        :param artifact_value:
        :param extra_punctuation:
        :param add_artifact_value: If set to True, adds the artifact_value then checks its existence.
        :param dir_items: list of items
        :return:
        """
        dir_, filename = os.path.split(file_path)
        artifact_value = int(artifact_value)
        while True:
            if add_artifact_value:
                # Maybe someone requires their file to have the first artifact

                file_path = split_extension(filename,
                                            artifact_type=artifact_type,
                                            artifact_value=artifact_value,
                                            extra_punctuation=extra_punctuation)
                if not dir_items:
                    file_path = join(dir_, file_path)
            if (dir_items and not (file_path in dir_items)) or (not dir_items and not os.path.exists(file_path)):
                break

            if not add_artifact_value:
                file_path = split_extension(filename,
                                            artifact_type=artifact_type,
                                            artifact_value=artifact_value,
                                            extra_punctuation=extra_punctuation)
                if not dir_items:
                    file_path = join(dir_, file_path)
            artifact_value += 1
        return file_path

    @staticmethod
    def mkdir_incremental(dir_path: str | list[str], base_name="exp", fix_name=None, overwrite=False) -> Path:
        """
        makes new directories, if it exists increment it and makes another one. Good for hyperparameter tuning!
        Args:
            dir_path:
            base_name:
            fix_name: If provided this will be created!
            overwrite: If True, it will overwrite the existing directory!

        Returns:

        """
        os.makedirs(dir_path, exist_ok=True)
        if overwrite:
            return Path(dir_path)

        if fix_name is not None:
            final_path = os.path.join(dir_path, fix_name)
            os.makedirs(final_path, exist_ok=True)
        else:
            folders = []
            for dir_ in (os.listdir(dir_path) if isinstance(dir_path, str) else dir_path):
                if base_name in dir_:
                    counter = dir_.split(base_name + "_")[-1]
                    if counter.isdigit():
                        folders.append(int(counter))
            if len(folders) == 0:
                final_path = os.path.join(dir_path, base_name + f"_1")
            else:
                max_counter = max(folders)
                final_path = os.path.join(
                    dir_path, base_name + f"_{max_counter + 1}")
            os.makedirs(final_path)

        return Path(final_path)

    @staticmethod
    def execute_command(command: str):
        from subprocess import Popen, PIPE
        process = Popen(command, stdout=PIPE, stderr=None, shell=True)
        output = process.communicate()[0]
        return output.decode()

    @staticmethod
    def is_windows():
        if os.name != "posix":
            return True
        else:
            return False

    def split(path: str, depth: int = 1, continuous: bool = False, list_it: bool = False):
        """

        :param path:
        :param depth:
        :param continuous:
        :return:
        >>> DirUtils.split("/pooya/ali/saeed/wow.txt", 3)
        'ali'
        >>> DirUtils.split("/pooya/ali/saeed/wow.txt", 2)
        'saeed'
        >>> DirUtils.split("/pooya/ali/saeed/wow.txt", 1)
        'wow.txt'
        >>> DirUtils.split("/pooya/ali/saeed/wow.txt", 2, continuous=True)
        'saeed/wow.txt'
        >>> DirUtils.split("/pooya/ali/saeed", 2, continuous=True)
        'ali/saeed'
        >>> DirUtils.split("/pooya/ali/saeed", 2, continuous=True, list_it=True)
        ['ali', 'saeed']
        >>> DirUtils.split("/pooya/ali/saeed", 0)
        ['pooya', 'ali', 'saeed']
        """
        if depth == 0:
            outputs = []
            while path:
                path, p = split(path)
                if not p:
                    break
                outputs.insert(0, p)
            return outputs
        elif depth == 1:
            import warnings
            warnings.warn("Use os.path.split(path)[-1] for depth=1 :)")
            return split(path)[-1]
        elif depth < 1:
            raise ValueError("depth should not be lower than 1")
        else:
            if not continuous:
                p = path
                for _ in range(depth - 1):
                    p = split(p)[0]

                output = split(p)[-1]
            else:
                outputs = []
                for _ in range(depth):
                    path, p = split(path)
                    outputs.insert(0, p)
                if list_it:
                    output = outputs
                else:
                    output = os.path.join(*outputs)
            return output


mkdir_incremental = DirUtils.mkdir_incremental

if __name__ == '__main__':
    print() #
