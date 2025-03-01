#Classes for running dissertation code
from matplotlib.dviread import Box
from matplotlib.pylab import rand
from sympy import randMatrix
from Point import Point
from typing import List
from Functions import Function

class box(Point):
    def __init__(self, vector, loc, land: bool, turbine: bool, wind: bool):
        #Initialises the box with a position vector and defined as not-stopped
        super().__init__(vector)
        self.loc = loc
        self.land = land
        self.turbine = turbine
        self.wind = wind


###################################################

list_points = []
n = 4
for i in range(0 ,n):
    point = box(randMatrix(2,1),(1,2), True, False, True)
    list_points.append(point)

def test(val:List[box]):
    value = 0
    for i in val:
        if i.land == True:
            value += i.wind
    return value

print(test(list_points))

