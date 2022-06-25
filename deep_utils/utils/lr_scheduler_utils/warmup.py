import math
from typing import Callable


def cosine_reduce(x, total_steps):
    return math.cos((x / total_steps) * (math.pi / 2))


def warmup_cosine(
    warmup_steps, max_lr, total_steps, optimizer_lr=None, initial_lr=1e-6, min_lr=0
) -> Callable:
    """

    How to use it:
    import torch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(...))
    :param warmup_steps:
    :param max_lr:
    :param total_steps:
    :param optimizer_lr: While using huggingface trainer, the output lr is multiplied to the main lr of the optimizer
    which is a bit confusing but if using huggingface and want to make sure to have exact values pass in the optimizer_lr
    value as well so that it will become uneffected.
    :param initial_lr:
    :param min_lr:
    :return:
    >>> epochs = 25
    >>> warmup_steps = epochs//10
    >>> max_lr = 5e-5
    >>> lambda_func = warmup_cosine(warmup_steps, max_lr, epochs)
    >>> for step in range(1, epochs + 1):
    ...     print("lr:", round(lambda_func(step), 6))
    lr: 1.5e-05
    lr: 5e-05
    lr: 5e-05
    lr: 5e-05
    lr: 4.9e-05
    lr: 4.8e-05
    lr: 4.7e-05
    lr: 4.6e-05
    lr: 4.4e-05
    lr: 4.3e-05
    lr: 4.1e-05
    lr: 3.9e-05
    lr: 3.7e-05
    lr: 3.4e-05
    lr: 3.2e-05
    lr: 2.9e-05
    lr: 2.6e-05
    lr: 2.3e-05
    lr: 2e-05
    lr: 1.7e-05
    lr: 1.3e-05
    lr: 1e-05
    lr: 7e-06
    lr: 3e-06
    lr: 0.0
    """
    assert initial_lr < max_lr, "max_lr can't be equal to or less than initial lr"
    assert warmup_steps < total_steps, "warmup_steps can't be larger than total_steps"

    def lambda_func(x):
        if x <= warmup_steps:
            lr = initial_lr + (1 - cosine_reduce(x, warmup_steps)) * (
                max_lr - initial_lr
            )
        else:
            x = x - warmup_steps
            lr = min_lr + cosine_reduce(x, total_steps - warmup_steps) * (
                max_lr - min_lr
            )
            lr = max(min_lr, lr)
        if optimizer_lr is not None and isinstance(optimizer_lr, float):
            return lr / optimizer_lr
        return lr

    return lambda_func
