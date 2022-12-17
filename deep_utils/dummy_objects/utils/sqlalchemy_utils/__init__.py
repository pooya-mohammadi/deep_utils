from deep_utils.main_abs.dummy_framework.dummy_framework import DummyObject, requires_backends


class SQLAlchemyChecks(metaclass=DummyObject):
    _backend = ["sqlalchemy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)


class SQLAlchemyInserts(metaclass=DummyObject):
    _backend = ["sqlalchemy"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, self._backend)