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

    @classmethod
    def from_list(cls, box_list):
        return cls(PointDataClass(box_list[0], box_list[1]), PointDataClass(box_list[2], box_list[3]))

    def to_list(self):
        return [self.p1.x, self.p1.y, self.p2.x, self.p2.y]
