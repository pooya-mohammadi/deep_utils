from dataclasses import dataclass
from typing import Union


@dataclass
class PointDataClass:
    x: Union[int, float]
    y: Union[int, float]


@dataclass
class BoxDataClass:
    p1: PointDataClass
    p2: PointDataClass
