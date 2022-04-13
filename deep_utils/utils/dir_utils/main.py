import os
from pathlib import Path
import shutil
from os.path import join
from typing import Tuple, List, Dict, Union
from deep_utils.utils.utils.logging_ import value_error_log, log_print
from deep_utils.utils.os_utils.os_path import split_extension


def transfer_directory_items(in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False, skip_transfer=False,
                             remove_in_dir=False):
    """

    Args:
        in_dir:
        out_dir:
        transfer_list:
        mode:
        remove_out_dir:
        skip_transfer:
        remove_in_dir: if mode is mv and this is set to true the in_dir will be removed!

    Returns:

    """
    print(f'starting to copying/moving from {in_dir} to {out_dir}')
    if remove_out_dir or os.path.isdir(out_dir):
        remove_create(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    if mode == 'cp':
        for name in transfer_list:
            try:
                shutil.copy(os.path.join(in_dir, name), out_dir)
            except FileNotFoundError as e:
                if skip_transfer:
                    print('[INFO] shutil.copy did not find the file, skipping...')
                else:
                    raise FileNotFoundError()
    elif mode == 'mv':
        for name in transfer_list:
            try:
                shutil.move(os.path.join(in_dir, name), out_dir)
            except FileNotFoundError as e:
                if skip_transfer:
                    print('[INFO] shutil.move did not find the file, skipping...')
                else:
                    raise FileNotFoundError()
        if remove_in_dir:
            shutil.rmtree(in_dir)
    else:
        raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
    print(f'finished copying/moving from {in_dir} to {out_dir}')


def dir_train_test_split(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False,
                         skip_transfer=False, remove_in_dir=False):
    from sklearn.model_selection import train_test_split
    list_ = os.listdir(in_dir)
    train_name, val_name = train_test_split(list_, test_size=test_size)
    transfer_directory_items(in_dir, train_dir, train_name, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=False)
    transfer_directory_items(in_dir, val_dir, val_name, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=remove_in_dir)
    return train_name, val_name


def split_dir_of_dir(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False,
                     remove_in_dir=False):
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
            print(f"[INFO] {dir_} is equal to {val_dir} or {train_dir}, Skipping ...")
            continue
        if not os.path.isdir(dir_):
            print(f"[INFO] {dir_} is not a directory, Skipping ...")
            continue
        if len(os.listdir(dir_)) == 0:
            print(f"[INFO] {dir_} is empty, Skipping ...")
            continue
        dir_train_test_split(dir_, train_dir=join(train_dir, data), val_dir=join(val_dir, data), mode=mode,
                             test_size=test_size, remove_out_dir=remove_out_dir, remove_in_dir=remove_in_dir)
    if mode == 'mv' and remove_in_dir:
        shutil.rmtree(in_dir)


def split_xy_dir(x_in_dir,
                 y_in_dir,
                 x_train_dir='train/samples',
                 y_train_dir='train/targets',
                 x_val_dir='val/samples',
                 y_val_dir='val/targets',
                 mode='cp',
                 val_size=0.1,
                 skip_transfer=False,
                 remove_out_dir=False,
                 remove_in_dir=False):
    train_names, val_names = dir_train_test_split(x_in_dir,
                                                  train_dir=x_train_dir,
                                                  val_dir=x_val_dir,
                                                  mode=mode,
                                                  remove_out_dir=remove_out_dir,
                                                  test_size=val_size)
    train_labels = [os.path.splitext(name)[0] + '.txt' for name in train_names]
    val_labels = [os.path.splitext(name)[0] + '.txt' for name in val_names]

    transfer_directory_items(y_in_dir, y_train_dir,
                             train_labels, mode=mode, remove_out_dir=remove_out_dir, skip_transfer=skip_transfer,
                             remove_in_dir=remove_in_dir)
    transfer_directory_items(y_in_dir, y_val_dir, val_labels,
                             mode=mode, remove_out_dir=remove_out_dir, skip_transfer=skip_transfer,
                             remove_in_dir=remove_in_dir)


def crawl_directory_dataset(dir_: str, ext_filter: list = None, map_labels=False, label_map_dict: dict = None,
                            logger=None,
                            verbose=1) -> Union[
    Tuple[List[str], List[int]], Tuple[List[str], List[int], Dict]]:
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
        for item_name in os.listdir(cls_path):
            item_path = join(cls_path, item_name)
            name, ext = os.path.splitext(item_name)
            if ext_filter is not None and ext not in ext_filter:
                log_print(logger, f"{item_path} with {ext} is not in ext_filtering: {ext_filter}", verbose=verbose)
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


def remove_create(dir_):
    import os
    import shutil
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)


def mkdir_incremental(dir_path: str, base_name='exp', fix_name=None) -> Path:
    """
    makes new directories, if it exists increment it and makes another one. Good for hyperparameter tuning!
    Args:
        dir_path:
        base_name:
        fix_name: If provided this will be created!

    Returns:

    """
    os.makedirs(dir_path, exist_ok=True)
    if fix_name is not None:
        final_path = os.path.join(dir_path, fix_name)
        os.makedirs(final_path, exist_ok=True)
    else:
        folders = []
        for dir_ in os.listdir(dir_path):
            if base_name in dir_:
                counter = dir_.split(base_name + '_')[-1]
                if counter.isdigit():
                    folders.append(int(counter))
        if len(folders) == 0:
            final_path = os.path.join(dir_path, base_name + f"_1")
        else:
            max_counter = max(folders)
            final_path = os.path.join(dir_path, base_name + f"_{max_counter + 1}")
        os.makedirs(final_path)

    return Path(final_path)


def cp_mv_all(input_dir, res_dir, mode="cp", filter_ext: Union[list, None] = None, logger=None, verbose=1):
    """
    Using shutil library all the move/copy all the files from one directory to another one!
    :param input_dir:
    :param res_dir:
    :param mode:
    :param filter_ext:
    :return:
    """
    n = 0
    for f_name in os.listdir(input_dir):
        _, ext = split_extension(f_name)
        if filter_ext is not None and ext in filter_ext:
            continue
        f_in_path = os.path.join(input_dir, f_name)
        f_out_path = os.path.join(res_dir, f_name)
        if mode == "cp":
            shutil.copy(f_in_path, f_out_path)
            n += 1
        elif mode == "mv":
            shutil.move(f_in_path, f_out_path)
            n += 1
        else:
            raise value_error_log(logger, f"mode {mode} is not supported!")
    log_print(logger, f"Successfully moved {n} items with filters: {filter_ext} from {input_dir} to {res_dir}")


def split_segmentation_dirs(in_images, in_masks, out_train="./train", out_val="./val", image_dir_name="images",
                            mask_dir_name="masks", img_ext=None, mask_ext=None,
                            mode='cp', test_size=0.2, remove_out_dir=False,
                            remove_in_dir=False, skip_transfer=False,
                            ):
    from sklearn.model_selection import train_test_split
    if img_ext is None and mask_ext is None:
        in_image_list = os.listdir(in_images)
        in_mask_list = os.listdir(in_masks)
        in_mask_dict = {split_extension(i)[0]: i for i in in_mask_list}

        in_image_list_train, in_image_list_val = train_test_split(in_image_list, test_size=test_size)
        in_mask_list_train = [in_mask_dict[split_extension(i)[0]] for i in in_image_list_train]
        in_mask_list_val = [in_mask_dict[split_extension(i)[0]] for i in in_image_list_val]
    else:
        in_image_list = [i for i in os.listdir(in_images) if i.endswith(img_ext)]
        in_image_list_train, in_image_list_val = train_test_split(in_image_list, test_size=test_size)
        in_mask_list_train = [split_extension(i, extension=mask_ext) for i in in_image_list_train]
        in_mask_list_val = [split_extension(i, extension=mask_ext) for i in in_image_list_val]

    transfer_directory_items(in_images, join(out_train, image_dir_name),
                             in_image_list_train, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=remove_in_dir)
    transfer_directory_items(in_images, join(out_val, image_dir_name),
                             in_image_list_val, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=remove_in_dir)

    transfer_directory_items(in_masks, join(out_train, mask_dir_name),
                             in_mask_list_train, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=remove_in_dir)
    transfer_directory_items(in_masks, join(out_val, mask_dir_name),
                             in_mask_list_val, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer, remove_in_dir=remove_in_dir)
