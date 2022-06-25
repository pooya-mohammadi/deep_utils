_nt_itemgetters = {}


def dictnamedtuple(typename, field_names, *, rename=False, defaults=None, module=None):
    import sys as _sys
    from collections import OrderedDict
    from keyword import iskeyword as _iskeyword
    from operator import itemgetter as _itemgetter

    if isinstance(field_names, str):
        field_names = field_names.replace(",", " ").split()
    field_names = list(map(str, field_names))
    typename = _sys.intern(str(typename))

    if rename:
        seen = set()
        for index, name in enumerate(field_names):
            if (
                not name.isidentifier()
                or _iskeyword(name)
                or name.startswith("_")
                or name in seen
            ):
                field_names[index] = f"_{index}"
            seen.add(name)

    for name in [typename] + field_names:
        if not isinstance(name, str):
            raise TypeError("Type names and field names must be strings")
        if not name.isidentifier():
            raise ValueError(
                "Type names and field names must be valid " f"identifiers: {name!r}"
            )
        if _iskeyword(name):
            raise ValueError(
                "Type names and field names cannot be a " f"keyword: {name!r}"
            )

    seen = set()
    for name in field_names:
        if name.startswith("_") and not rename:
            raise ValueError(
                "Field names cannot start with an underscore: " f"{name!r}"
            )
        if name in seen:
            raise ValueError(f"Encountered duplicate field name: {name!r}")
        seen.add(name)

    field_defaults = {}
    if defaults is not None:
        defaults = tuple(defaults)
        if len(defaults) > len(field_names):
            raise TypeError("Got more default values than field names")
        field_defaults = dict(
            reversed(list(zip(reversed(field_names), reversed(defaults))))
        )

    # Variables used in the methods and docstrings
    field_names = tuple(map(_sys.intern, field_names))
    num_fields = len(field_names)
    arg_list = repr(field_names).replace("'", "")[1:-1]
    repr_fmt = "(" + ", ".join(f"{name}=%r" for name in field_names) + ")"
    tuple_new = tuple.__new__
    _len = len

    # Create all the named tuple methods to be added to the class namespace

    s = f"def __new__(_cls, {arg_list}): return _tuple_new(_cls, ({arg_list}))"
    namespace = {"_tuple_new": tuple_new, "__name__": f"namedtuple_{typename}"}
    # Note: exec() has the side-effect of interning the field names
    exec(s, namespace)
    __new__ = namespace["__new__"]
    __new__.__doc__ = f"Create new instance of {typename}({arg_list})"
    if defaults is not None:
        __new__.__defaults__ = defaults

    @classmethod
    def _make(cls, iterable):
        result = tuple_new(cls, iterable)
        if _len(result) != num_fields:
            raise TypeError(
                f"Expected {num_fields} arguments, got {len(result)}")
        return result

    _make.__func__.__doc__ = (
        f"Make a new {typename} object from a sequence " "or iterable"
    )

    def _replace(_self, **kwds):
        result = _self._make(map(kwds.pop, field_names, _self))
        if kwds:
            raise ValueError(f"Got unexpected field names: {list(kwds)!r}")
        return result

    _replace.__doc__ = (
        f"Return a new {typename} object replacing specified " "fields with new values"
    )

    def __repr__(self):
        "Return a nicely formatted representation string"
        return self.__class__.__name__ + repr_fmt % self

    def _asdict(self):
        "Return a new OrderedDict which maps field names to their values."
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        "Return self as a plain tuple.  Used by copy and pickle."
        return tuple(self)

    # Modify function metadata to help with introspection and debugging

    for method in (
        __new__,
        _make.__func__,
        _replace,
        __repr__,
        _asdict,
        __getnewargs__,
    ):
        method.__qualname__ = f"{typename}.{method.__name__}"

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        "__doc__": f"{typename}({arg_list})",
        "__slots__": (),
        "_fields": field_names,
        "_fields_defaults": field_defaults,
        "__new__": __new__,
        "_make": _make,
        "_replace": _replace,
        "__repr__": __repr__,
        "_asdict": _asdict,
        "__getnewargs__": __getnewargs__,
    }
    cache = _nt_itemgetters
    for index, name in enumerate(field_names):
        try:
            itemgetter_object, doc = cache[index]
        except KeyError:
            itemgetter_object = _itemgetter(index)
            doc = f"Alias for field number {index}"
            cache[index] = itemgetter_object, doc
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (tuple,), class_namespace)

    # For pickling to work, the __module__ variable needs to be set to the frame
    # where the named tuple is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython), or where the user has
    # specified a particular module.
    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module

    class DictNamedTuple(result):

        DictNamedTuple = True
        TypeName = typename

        def keys(self):
            keys = []
            for key in self._fields:
                keys.append(key)
            return keys

        def items(self):
            found = []
            for key in self._fields:
                value = getattr(self, key)
                found.append((key, value))
            return found

        def values(self):
            values = []
            for key in self._fields:
                values.append(getattr(self, key))
            return values

        def get(self, key):
            value = getattr(self, key)
            return value

        def __getitem__(self, item):
            if isinstance(item, str):
                item = self._fields.index(item)
            val = super(DictNamedTuple, self).__getitem__(item)
            return val

    return DictNamedTuple
