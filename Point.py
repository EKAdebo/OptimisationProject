from typing import List
from Functions import Function

class Point:

    def __init__(self, vector):
        #Initialises the point with a position vector and defined as not-stopped
        self.stopped = False
        self.vector = vector
        self.f = []
        self.eval_f = []
        self.eval_d = []

    def evaluate(self, functions:List[Function]):
        #Evaluates the point with a set of functions
        for func in functions:
            self.f.append(func.name)
            self.eval_f.append(func(self.vector))
            self.eval_d.append(func.evaluate_gradient(self.vector))


            
