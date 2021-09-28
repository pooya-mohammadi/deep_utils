import os
import shutil
from os.path import join


def transfer_directory_items(in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False):
    print(f'starting to copying/moving from {in_dir} to {out_dir}')
    if remove_out_dir or os.path.isdir(out_dir):
        remove_create(out_dir)
    else:
        os.makedirs(out_dir, exist_ok=True)
    if mode == 'cp':
        for name in transfer_list:
            shutil.copy(os.path.join(in_dir, name), out_dir)
    elif mode == 'mv':
        for name in transfer_list:
            shutil.move(os.path.join(in_dir, name), out_dir)
    else:
        raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
    print(f'finished copying/moving from {in_dir} to {out_dir}')


def dir_train_test_split(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False):
    from sklearn.model_selection import train_test_split
    list_ = os.listdir(in_dir)
    train_name, val_name = train_test_split(list_, test_size=test_size)
    transfer_directory_items(in_dir, train_dir, train_name, mode=mode, remove_out_dir=remove_out_dir)
    transfer_directory_items(in_dir, val_dir, val_name, mode=mode, remove_out_dir=remove_out_dir)
    return train_name, val_name


def split_dir_of_dir(in_dir, train_dir='./train', val_dir='./val', test_size=0.1, mode='cp', remove_out_dir=False):
    for data in os.listdir(in_dir):
        dir_ = join(in_dir, data)
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
                             train_labels, mode=mode, remove_out_dir=remove_out_dir)
    transfer_directory_items(y_in_dir, y_val_dir, val_labels,
                             mode=mode, remove_out_dir=remove_out_dir)


def remove_create(dir_):
    import os
    import shutil
    if os.path.exists(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)
