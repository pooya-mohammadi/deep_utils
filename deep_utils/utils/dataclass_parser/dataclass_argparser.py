import argparse
import dataclasses


__all__ = ('DataClassArgParser', )

class Arg:
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class DataClassArgParser:
    Arg = Arg
    class Int(Arg):
        def __init__(self, **kwargs):
            super().__init__(type=int, **kwargs)


    class Float(Arg):
        def __init__(self, **kwargs):
            super().__init__(type=float, **kwargs)


    class Str(Arg):
        def __init__(self, **kwargs):
            super().__init__(type=str, **kwargs)


    class _MetaChoice(type):
        def __getitem__(self, item):
            return self(choices=list(item), type=item)


    class Choice(Arg, metaclass=_MetaChoice):
        def __init__(self, choices, **kwargs):
            super().__init__(choices=choices, **kwargs)

    @staticmethod
    def parse_to(container_class, **kwargs):
        def mangle_name(name):
            return '--' + name

        parser = argparse.ArgumentParser(
            description=container_class.__doc__)
        for field in dataclasses.fields(container_class):
            name = field.name
            default = field.default
            value_or_class = field.type
            if isinstance(value_or_class, type):
                try:
                    value = value_or_class(default=default)
                    kwargs_ = value.kwargs
                except TypeError:
                    value = value_or_class(default)
                    kwargs_ = dict(default=default, type=value_or_class)
            else:
                value = value_or_class
                value.kwargs['default'] = default
                kwargs_ = value.kwargs
            parser.add_argument(
                mangle_name(name), **kwargs_)

        arg_dict = parser.parse_args(**kwargs)
        return container_class(**vars(arg_dict))
