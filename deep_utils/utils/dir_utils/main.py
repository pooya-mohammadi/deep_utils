import os
import shutil


def transfer_directory_items(in_dir, out_dir, transfer_list, mode='cp', remove_out_dir=False):
    print(f'starting to copying/moving from {in_dir} to {out_dir}')
    if remove_out_dir or os.path.isdir(out_dir):
        os.system(f'rm -rf {out_dir}; mkdir -p {out_dir}')
    if mode == 'cp':
        for name in transfer_list:
            shutil.copy(os.path.join(in_dir, name), out_dir)
    elif mode == 'mv':
        for name in transfer_list:
            shutil.move(os.path.join(in_dir, name), out_dir)
    else:
        raise ValueError(f'{mode} is not supported, supported modes: mv and cp')
    print(f'finished copying/moving from {in_dir} to {out_dir}')


def dir_train_test_split(in_dir, out_dir, test_size=0.1, result_names=('train', 'val'), mode='cp',
                         remove_out_dir=False):
    from sklearn.model_selection import train_test_split
    list_ = os.listdir(in_dir)
    train_name, val_name = train_test_split(list_, test_size=test_size)
    transfer_directory_items(in_dir,
                             os.path.join(out_dir, result_names[0]),
                             train_name, mode=mode,
                             remove_out_dir=remove_out_dir)
    transfer_directory_items(in_dir,
                             os.path.join(out_dir, result_names[1]),
                             train_name, mode=mode,
                             remove_out_dir=remove_out_dir)
