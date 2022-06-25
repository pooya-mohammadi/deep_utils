def load_tf_opt(opt_name: str, lr: float, **args):
    """
    This module is used for loading tensorflow/keras optimizers
    Args:
        opt_name:
        lr:
        **args:

    Returns:

    """
    from tensorflow.keras.optimizers import SGD, Adadelta, Adam, RMSprop

    opt_dict = {"adam": Adam, "rmsprop": RMSprop,
                "sgd": SGD, "adadelta": Adadelta}
    opt = opt_dict.get(opt_name.lower(), None)
    if opt is None:
        raise ValueError(
            f"[ERROR] optimization: {opt_name} is not supported, supported optimization are {opt_dict.keys()}"
        )
    return opt(learning_rate=lr, **args)
