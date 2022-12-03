try:
    from deep_utils.dummy_objects.utils.sqlalchemy_utils import SQLAlchemyChecks
    from .checks import SQLAlchemyChecks
except ModuleNotFoundError:
    pass
