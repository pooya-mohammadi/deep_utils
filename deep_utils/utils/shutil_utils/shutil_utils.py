import shutil
from os.path import join, split
from typing import List


def mv_or_copy(src, dst, mode):
    if mode == "cp":
        shutil.copy(src, dst)
    elif mode == "mv":
        shutil.move(src, dst)


def mv_or_copy_list(src_lst: List[str], dst: str, mode):
    if mode == "cp":
        for src in src_lst:
            src_name = split(src)[1]
            shutil.copy(src, join(dst, src_name))
    elif mode == "mv":
        for src in src_lst:
            src_name = split(src)[1]
            shutil.move(src, join(dst, src_name))
