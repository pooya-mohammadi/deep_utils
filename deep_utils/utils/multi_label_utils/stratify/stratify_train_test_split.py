from typing import Union
import time
from deep_utils.utils.algorithm_utils.main import subset_sum
import numpy as np


def stratify_train_test_split_multi_label(x: Union[list, tuple, np.ndarray], y: np.ndarray, test_size=0.2,
                                          closest_ratio=False):
    """

    :param x:
    :param y: A 2D array that represents number of labels in each class. Each column is representative of a class.
    :param test_size:
    :param closest_ratio: For huge arrays extracting the closest ratio requires an intensive recursive function to work
     which could result in maximum recursion error. If set to True which choose samples from the smallest till passes
     the target number. Set this variable to True if you are sure. by default is set to False.
    :return:
    """
    assert len(y.shape) == 2, "y should be 2D"
    assert test_size > 0.0, "test_size cannot be a zero or negative value!"
    x = np.array(x, dtype=np.object_) if not isinstance(x, np.ndarray) else x
    available_samples = np.ones((y.shape[0]), dtype=np.bool8)
    test_samples = np.zeros((y.shape[0]), dtype=np.bool8)
    train_samples = np.zeros((y.shape[0]), dtype=np.bool8)
    class_sample_counts = y.sum(axis=0)
    # stratify starts from a class with the lowest number of samples
    class_indices = np.argsort(class_sample_counts)
    for class_index in class_indices:
        test_number_samples = y[:, class_index][test_samples].sum()
        n_test = np.ceil(class_sample_counts[class_index] * test_size)
        n_test = max(0, n_test - test_number_samples)
        input_labels = y[:, class_index].copy()
        input_labels[np.invert(available_samples)] = 0
        if n_test == 0 or len(input_labels) == 0:
            continue
        if closest_ratio:
            chosen_indices, *_ = subset_sum(input_numbers=input_labels, target_number=n_test)
        else:
            sorted_indices = np.argsort(input_labels)
            cum_sum_values = np.cumsum(input_labels[sorted_indices])
            chosen_indices = sorted_indices[cum_sum_values < n_test].tolist()
            if len(chosen_indices) < len(sorted_indices):
                chosen_indices.append(sorted_indices[len(chosen_indices)])
            # ratio = input_labels[sorted_indices[cum_sum_values < n_test]].sum()/ cum_sum_values[-1]
            # v = 1
        # Update available_samples, train_samples, test_samples
        for update_index, n_label in enumerate(input_labels):
            if n_label == 0:
                # samples that have no elements are ignored ...
                continue
            if update_index in chosen_indices:
                test_samples[update_index] = True
                train_samples[update_index] = False
            else:
                test_samples[update_index] = False
                train_samples[update_index] = True
            available_samples[update_index] = False
    # those that are not
    train_samples = np.bitwise_or(train_samples, np.bitwise_not(test_samples))
    return x[train_samples], x[test_samples], y[train_samples], y[test_samples]