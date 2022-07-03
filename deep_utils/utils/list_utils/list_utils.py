def shift_lst(lst: list, move_forward):
    return lst[-move_forward:] + lst[:-move_forward]
