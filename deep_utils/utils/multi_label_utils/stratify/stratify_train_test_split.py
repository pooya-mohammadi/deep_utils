from typing import Union
from deep_utils.utils.algorithm_utils.main import subset_sum
import numpy as np


def stratify_train_test_split_multi_label(x: Union[list, tuple, np.ndarray], y: np.ndarray, test_size=0.2,
                                          closest_ratio=False):
    """
    A handy function for splitting multi-label samples based on their number of classes. This is mainly useful for
    object detection and ner-like tasks that each sample may contain several objects/tags from different classes! The
    process of splitting starts from classes with the smallest number of samples to make sure their ratio is saved
    because they have small number of samples and retaining the ratio for them is challenging compared to those classes
    with more samples.
    :param x: A list, Tuple or ndarray that contains the samples
    :param y: A 2D array that represents number of labels in each class. Each column is representative of a class. As an
    example: y = np.array([[2, 3], [1, 1]]) says that sample one has
    two objects/tags for class 0 and 3 objects/tags for class 1 and so on.
    :param test_size:
    :param closest_ratio: For huge arrays extracting the closest ratio requires an intensive recursive function to work
     which could result in maximum recursion error. If set to True which choose samples from the smallest till passes
     the target number. Set this variable to True if you are sure. by default is set to False.
    :return:
    >>> y = np.array([[1, 2, 0], [1, 0, 0], [1, 2, 0]])
    >>> x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    >>> stratify_train_test_split_multi_label(x, y, test_size=0.3)
    (array([[2, 2, 2],
           [3, 3, 3]]), array([[1, 1, 1]]), array([[1, 0, 0],
           [1, 2, 0]]), array([[1, 2, 0]]))

    """
    assert len(y.shape) == 2, "y should be 2D"
    assert test_size > 0.0, "test_size cannot be a zero or negative value!"
    from sklearn.model_selection import train_test_split
    x = np.array(x, dtype=np.object_) if not isinstance(x, np.ndarray) else x

    # excluding samples with no objects/tags
    y_no_objects = y[y.sum(1) == 0]
    x_no_objects = x[y.sum(1) == 0]
    x = x[y.sum(1) > 0]
    y = y[y.sum(1) > 0]

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
    # Allocating all the remaining samples to train because the code structure ensures the ratio of test
    # samples to the whole dataset.
    train_samples = np.bitwise_or(train_samples, np.bitwise_not(test_samples))

    # splitting samples with no objects
    x_no_objects_train, x_no_objects_test, y_no_objects_train, y_no_objects_test = train_test_split(x_no_objects,
                                                                                                    y_no_objects,
                                                                                                    test_size=test_size)

    return np.concatenate([x[train_samples], x_no_objects_train]), \
           np.concatenate([x[test_samples], x_no_objects_test]), \
           np.concatenate([y[train_samples], y_no_objects_train]), \
           np.concatenate([y[test_samples], y_no_objects_test])


if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1], [1, 0], [1, 0]])
    x_train, x_test, y_train, y_test = stratify_train_test_split_multi_label(X, y, test_size=0.5, closest_ratio=False)
    print("x_train", x_train)
    print("x_test", x_test)
    print("y_train", y_train)
    print("y_test", y_test)
    print("ratio:", y_test.sum(0) / y.sum(0))
