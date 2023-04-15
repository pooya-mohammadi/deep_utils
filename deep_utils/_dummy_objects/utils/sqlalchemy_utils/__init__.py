from deep_utils._dummy_objects.dummy_framework import DummyObject, requires_backends


class SQLAlchemyChecks(metaclass=DummyObject):
    _backend = ["sqlalchemy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class SQLAlchemyInserts(metaclass=DummyObject):
    _backend = ["sqlalchemy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class SQLAlchemyUtils(metaclass=DummyObject):
    _backend = ["sqlalchemy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)
