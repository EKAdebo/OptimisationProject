from math import floor
from typing import List
import numpy as np
#from Functions import Function
from Point import Point


def compare(s:Point, R:List[Point]) -> bool:
    '''
    Checks one vector for non-dominance against a list of vectors

    INPUTS
    R - set of vectors
    s - one vector

    OUPUT
    bool - True if s is not dominated by any vector in R, False otherwise
    '''

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
    V -  vectors

    OUPUT
    M - The set of non-dominated vectors in V
    '''

    vSize = len(V)
    half = np.floor(vSize/2).astype(int)


    if vSize == 1:
        "dead end"
        return V
    
    else:
        R = Front(V[0:half])
        S = Front(V[half:vSize])

    R_val = [r.eval_f[0] for r in R]
    S_val = [s.eval_f[0] for s in S]


    T = np.array([s for s in S if compare(s,R)] )

    T_val = [t.eval_f[0] for t in T]
    
    if T.size == 0:
        return R
    
    R.extend(T)

    return R



def KungMethod(V:List[Point]) -> List[Point]:
    '''
    An implementation of Kung's method, it returns the set of non-dominated values in V
    NOTE: This algorithm assumes a minimisation problem
    
    INPUT
    V - set of vectors
    
    OUPUT
    set of non-dominated vectors
    '''

    V.sort(key=lambda point: point.eval_f[0])


    nonDom = Front(V)

    return nonDom
