from typing import TypeVar, Union

from sqlalchemy import select

SQLAlchemyUtilsType = TypeVar("SQLAlchemyUtilsType", bound="SQLAlchemyUtils")


class SQLAlchemyUtils:

    @staticmethod
    def select_id(cls, id_, session, class_id_feature="id") -> Union[SQLAlchemyUtilsType, None]:
        try:
            selected = select(cls).where(getattr(cls, class_id_feature) == id_)
            instance = session.scalars(selected).one()
        except:
            return None
        return instance

    @staticmethod
    def update(instance, session):
        try:
            session.add(instance)
            session.commit()
        except:
            session.rollback()
        return instance
