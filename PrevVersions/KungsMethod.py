import numpy as np

##Sort by first item of each vector - decreasing
def Sort(V):
    '''
    Sort by first item of each vector in V
    Inputs
    V - set of vectors
    '''
    arrT = V.T
    sorted_ind = np.lexsort(arrT[::-1])
    sorted_arr = V[sorted_ind]
    return sorted_arr

def compare(s, R):
    '''
    Checks one vector for non-dominance against a list of vectors

    INPUTS
    R - set of vectors
    s - one vector

    OUPUT
    bool - True if s is not dominated by any vector in R, False otherwise
    '''
    for r in R:
        if (s >= r).all():
            return False 
    return True

def Front(V):
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
        return V
    
    else:
        R = Front(V[0:half])
        S = Front(V[half:vSize])

    T = np.array([s for s in S if compare(s,R)] )
    
    if T.size == 0:
        return R
    
    M = np.concatenate((R,T))

    return M



def KungMethod(V):
    '''
    An implementation of Kung's method, it returns the set of non-dominated values in V
    NOTE: This algorithm assumes a minimisation problem
    
    INPUT
    V - set of vectors
    
    OUPUT
    set of non-dominated vectors
    '''
    
    VSorted = Sort(V)
    nonDom = Front(VSorted)
    return nonDom
