from sqlalchemy import select, and_, or_
from sqlalchemy.exc import NoResultFound
from typing import Dict, Any


class SQLAlchemyChecks:

    @staticmethod
    def check(cls, key_value: Dict[str, Any], combine="and", session=None) -> bool:

        conditions = [getattr(cls, key) == val for key, val in key_value.items()]
        condition = and_(*conditions) if combine.lower() == "and" else or_(*conditions)

        selected = select(cls).where(condition)
        try:
            session.scalars(selected).one()
        except NoResultFound:
            return True
        return False
