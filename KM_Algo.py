from math import floor
from typing import List
import numpy as np
from Point import Point

def compare(s:Point, R:List[Point]) -> bool:
    '''
    Checks one point for non-dominance against a list of points

    INPUTS
    R - list of vectors
    s - one vector

    OUPUT
    bool - True if s is not dominated by any vector in R, False otherwise
    '''
    #The results of the functions evaluated at the point s
    eval_s = s.eval_f

    for r in R:
        eval_r = r.eval_f
        result =any([eval_s[i] < eval_r[i] for i in range(len(eval_s))])

        if  (not result):
            return False

    return True

def Front(V:List[Point]) -> List[Point]:
    '''
    A recursive function that finds the non-dominated vectors in V

    INPUTS
    V -  Set of points, assumed to be evaluated at the objective functions and sorted by the

    OUPUT
    M - The set of non-dominated vectors in V
    '''
    #Dividing the list into two halves
    vSize = len(V)

    #If there is only one point in the list, return it
    if vSize == 1:
        return V

    half = np.floor(vSize/2).astype(int)

    #Formation of the two lists
    R = Front(V[0:half])
    S = Front(V[half:vSize])

    #Finding the points in S that are non-dominated by R
    T = np.array([s for s in S if compare(s,R)] )

    if T.size == 0:
        return R

    R.extend(T)

    return R

def KungMethod(V:List[Point]) -> List[Point]:
    '''
    THis method takes in a set of points and returns the non-dominated ones.
    It is an implementation of Kung's method.
    NOTE: This algorithm assumes a minimisation problem

    INPUT
    V - A set of points assumed to be evaluated for the necessary objective functions

    OUPUT
    set of non-dominated points
    '''

    #Sort of Points corresponding to the first objective function
    V.sort(key=lambda point: point.eval_f[0])

    nonDom = Front(V)

    return nonDom