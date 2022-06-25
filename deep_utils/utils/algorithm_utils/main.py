import sys


def subset_sum(
    input_numbers,
    target_number,
    result=(None, None, sys.maxsize),
    partial=None,
    indices=None,
    index=0,
):
    """
    Returns a subset of the inputs that has the closest value to the target_number input.
    Output: indices of chosen samples, values of chosen samples, difference from the input target_number.
    :param input_numbers:
    :param target_number:
    :param result:
    :param partial:
    :param indices:
    :param index:
    :return:
    >>> subset_sum([3, 9, 8, 4, 5, 7, 10], 150)
    [(0, 1, 2, 3, 4, 5, 6), (3, 9, 8, 4, 5, 7, 10), 104]
    >>> subset_sum([3, 9, 8, 4, 5, 7, 10], 15)
    [(0, 2, 3), (3, 8, 4), 0]
    >>> subset_sum([0, 0, 2], 1)
    [(2,), (2,), 1]
    """
    indices = [] if indices is None else indices
    partial = [] if partial is None else partial
    result = list(result) if not isinstance(result, list) else result
    result[0] = [] if result[0] is None else result[0]
    result[1] = [] if result[1] is None else result[1]

    s = sum(partial)

    # check if the partial sum is equals to target
    diff = abs(s - target_number)

    if diff == 0:
        result[0] = tuple(indices)
        result[1] = tuple(partial)
        result[-1] = abs(target_number - sum(partial))
        return
    elif diff <= result[-1]:
        result[0] = tuple(indices)
        result[1] = tuple(partial)
        result[-1] = abs(target_number - sum(partial))

    for i in range(len(input_numbers)):
        current_index = i + index
        n = input_numbers[i]
        remaining = input_numbers[i + 1:]
        subset_sum(
            remaining,
            target_number,
            result,
            partial + [n],
            indices + [current_index],
            current_index + 1,
        )
        if result[-1] == 0 or sum(partial) - target_number > result[-1]:
            break
    return result
