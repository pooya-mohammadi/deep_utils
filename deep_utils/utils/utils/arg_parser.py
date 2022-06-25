def easy_argparse(*args):
    """
    This is a simple implementation of arg_parser. Each argument should be passed as dictionaries
    Args:
        *args:

    Returns:

    """
    import argparse

    parser = argparse.ArgumentParser()

    for arg in args:
        tp = type(arg)
        if tp == dict:
            name = arg.pop("name")
            parser.add_argument(name, **arg)
        else:
            raise TypeError(f"easy_argparse just supports dictionaries")
    return parser.parse_args()
