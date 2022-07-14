import shutil


def mv_or_copy(src, dst, mode):
    if mode == "cp":
        shutil.copy(src, dst)
    elif mode == "mv":
        shutil.move(src, dst)
