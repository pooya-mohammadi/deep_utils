from sqlalchemy import select, and_, or_
from sqlalchemy.exc import NoResultFound
from typing import Dict, Any, TypeVar

InstanceType = TypeVar("InstanceType")


class SQLAlchemyInserts:

    @staticmethod
    def insert(session, instance: InstanceType) -> InstanceType:
        try:
            session.add(instance)
            session.commit()
        except:
            session.rollback()
        return instance
