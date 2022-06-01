import re
from typing import List



class Shape():
    def __init__(self) -> None:
        pass

    def DecidePoint(self,point):
        raise NotImplementedError()

class Rectangle():
    def __init__(self,point1,point2) -> None:

        self._point_1 = point1
        self._point_2 = point2

        if self._point_1[0] > self._point_2[0]:
            temp = self._point_2[0]
            self._point_2[0] = self._point_1[0]
            self._point_1[0] = temp

        if self._point_1[1] > self._point_2[1]:
            temp = self._point_2[1]
            self._point_2[1] = self._point_1[1]
            self._point_1[1] = temp


    def DecidePoint(self,point):
        isOk = True

        isOk = isOk and point[0] >= self._point_1[0] and point[0] <= self._point_2[0]
        isOk = isOk and point[1] >= self._point_1[1] and point[1] <= self._point_2[1]

        return isOk


class ShapeGen():
    def __init__(self,shapes: List[Shape]) -> None:
        self._shapes = shapes


    def DecidePoint(self,point):

        pointIsOk: bool = False

        for shape in self._shapes:
            pointIsOk = pointIsOk or shape.DecidePoint(point)

        return pointIsOk

    def FilterPoints(self, points):
        outpoints = []

        for iterPoint in points:
            if self.DecidePoint(iterPoint): outpoints.append(iterPoint)

        return outpoints
