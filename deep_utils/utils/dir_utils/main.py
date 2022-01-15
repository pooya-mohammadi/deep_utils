import os
import shutil
from os.path import join
from typing import Tuple, List, Dict


def transfer_directory_items(in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False, skip_transfer=False):
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
    else:
        raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
    print(f'finished copying/moving from {in_dir} to {out_dir}')


def dir_train_test_split(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False,
                         skip_transfer=False):
    from sklearn.model_selection import train_test_split
    list_ = os.listdir(in_dir)
    train_name, val_name = train_test_split(list_, test_size=test_size)
    transfer_directory_items(in_dir, train_dir, train_name, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer)
    transfer_directory_items(in_dir, val_dir, val_name, mode=mode, remove_out_dir=remove_out_dir,
                             skip_transfer=skip_transfer)
    return train_name, val_name


def split_dir_of_dir(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False):
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
                             test_size=test_size, remove_out_dir=remove_out_dir)


def split_xy_dir(x_in_dir,
                 y_in_dir,
                 x_train_dir='train/samples',
                 y_train_dir='train/targets',
                 x_val_dir='val/samples',
                 y_val_dir='val/targets',
                 mode='cp',
                 val_size=0.1,
                 skip_transfer=False,
                 remove_out_dir=False):
    train_names, val_names = dir_train_test_split(x_in_dir,
                                                  train_dir=x_train_dir,
                                                  val_dir=x_val_dir,
                                                  mode=mode,
                                                  remove_out_dir=remove_out_dir,
                                                  test_size=val_size)
    train_labels = [os.path.splitext(name)[0] + '.txt' for name in train_names]
    val_labels = [os.path.splitext(name)[0] + '.txt' for name in val_names]

    transfer_directory_items(y_in_dir, y_train_dir,
                             train_labels, mode=mode, remove_out_dir=remove_out_dir, skip_transfer=skip_transfer)
    transfer_directory_items(y_in_dir, y_val_dir, val_labels,
                             mode=mode, remove_out_dir=remove_out_dir, skip_transfer=skip_transfer)


def crawl_directory_dataset(dir_: str, ext_filter: list = None) -> Tuple[List[str], List[int], Dict]:
    """
    crawls a directory of classes and returns the full path of the items paths and their class names
    :param dir_: path to directory of classes
    :param ext_filter: extensions that will be passed and others will be dropped
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
                print(f"[INFO] {item_path} with {ext} is not in ext_filtering: {ext_filter}")
                continue
            x.append(item_path)
            if cls_name not in label_map:
                cls_index = len(label_map)
                label_map[cls_name] = cls_index
            else:
                cls_index = label_map[cls_name]
            y.append(cls_index)
    print(f"[INFO] successfully crawled {dir_}")
    return x, y, label_map


def remove_create(dir_):
    import os
    import shutil
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)
