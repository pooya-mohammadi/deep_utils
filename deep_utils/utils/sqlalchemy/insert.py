from typing import TypeVar

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
