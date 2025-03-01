from typing import List
from Functions import Function

class Point:

    def __init__(self, vector):
        #Initialises the point with a position vector and defined as not-stopped
        self.stopped = False
        self.vector = vector
        self.f = []                         #Names of the functions evaluated
        self.eval_f = []                    #Evaluation of the functions at the point
        self.eval_d = []                    #Evaluation of the gradient of the functions at the point

    def evaluate(self, functions:List[Function]):
        #Evaluates the point with a set of functions
        for func in functions:
            self.f.append(func.name)                                #Appends the name of the function
            self.eval_f.append(func(self.vector))                   #Appends the evaluation of the function
            self.eval_d.append(func.evaluate_gradient(self.vector)) #Appends the evaluation of the gradient of the function