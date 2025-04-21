import numpy as np
from typing import List
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
import json

from sympy import Li

from Point import Point
from Functions import Function, Constraint
from KM_Algo import KungMethod

###############################################

###############################################

def save_to_json(points:List[Point], filename="nonDominated.json"):
    """
    Save the points to a JSON file.
    Each point is represented as a dictionary with keys "vector", "eval_f", and "eval_d".
    Input:
    points: List of Point objects to be saved.
    filename: Name of the JSON file to save the points to.
    """
    data = []
    for p in points:
        data.append({
            "vector": p.vector.tolist(),
            "eval_f": p.eval_f,
            "eval_d": [d.tolist() for d in p.eval_d]
        })
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

###############################################

###############################################

def main(objFuncs: List[Function], G: List[Constraint], H: List[Constraint], X0:List[Point], iter: int = 50, mu: float = 0.25, beta: float = 0.5, tol: float = 1e-4):

    """
    Generates a set of points approximating the pareto front, using the algorithm described in
    A METHOD FOR CONSTRAINED MULTIOBJECTIVE OPTIMIZATION BASED ON SQPTECHNIQUES

    INPUTS
    objFuncs: List of functions to be minimised
    G: List of inequality constraints
    H: List of equality constraints
    X0: List of starting points
    iter: Number of iterations to run the algorithm
    mu: Penalty parameter
    beta: Step size parameter
    tol: Tolerance for stopping criteria

    OUTPUTS
    X0: List of starting points
    points: List of points estimated to be on the pareto front
    """

    #Stage 3 of the algorithm
    points = Stage3(X0.copy(), objFuncs, G, H, iter, tol, mu, beta)

    #Stage 4 of the algorithm
    points = Stage4(points, objFuncs, G, H, tol, mu, beta)


    return X0, points

###############################################

###############################################

def Stage3(points:List[Point], objFuncs:List[Function], G:List[Constraint], H:List[Constraint], iter:int, tol:float, mu:float, beta:float):
    """
    Stage 3 of the algorithm, generates a set of points approximating the pareto front.
    Over each iteration, and each point, a search direction is calculated for each objective function.
    The search direction is then used to generate a new point, which is added to the set of points.

    INPUTS
    points: List of starting points
    objFuncs: List of functions to be minimised
    G: List of inequality constraints
    H: List of equality constraints
    iter: Number of iterations to run the algorithm
    tol: Tolerance for stopping criteria
    mu: Penalty parameter
    beta: Step size parameter

    OUTPUTS
    points: List of points estimated to be on the pareto front
    """

    #Number of objective functions
    numFuncs = len(objFuncs)

    for i in range(iter):
        print("------------------------------")
        print(f"Iter {i}, X length {len(points)}")

        T = []

        #For each point in the set of points
        for p in (points):

            #For points p that are not stopped
            if p.stopped != True:

                #for each onde of the objective functions
                for j in range(0,numFuncs):

                    #Compute the search direction and lagrange multipliers
                    d,L = search_dir(p,G,H,objFuncs[j],j)

                    #If the search direction is too small, continue to either next point or next objective function
                    if np.linalg.norm(d) < np.power(tol, 1/4):
                        continue

                    #Calculate the penalty parameter based on the lagrange multipliers
                    pen_val = penalty(L)

                    #Calculate the step size alpha using the merit function
                    alpha = alpha_val_singular(p, np.array(d), G,H, objFuncs[j],beta,pen_val, mu,tol)

                    #If the step size is too small, continue to either next point or next objective function
                    if alpha < np.sqrt(tol):
                        continue

                    #Generate a new point using the step size and search direction
                    p_new = Point(p.vector + alpha * np.array(d))
                    p_new.evaluate(objFuncs)

                    T.append(p_new)

                #"Stop" the point p, as it has been evaluated for all objective functions
                p.stopped = True

            #If no new points were generated, end stage 3
            if len(T) == 0:
                break


            print(f"Points generated: {len(T)}")

            #Joim the new points with the existing points
            points.extend(T)

            #Find the set of non-dominated points using the Kung method
            points = KungMethod(points)
            print(f"Non-dominated points remaining: {len(points)}")

            #Save each iteration to a json file
            save_to_json(points, f"/iterations/iter_{i}.json")

    return points

###############################################

###############################################

def Stage4(points:List[Point], objFuncs:List[Function], G:List[Constraint], H:List[Constraint], tol:float, mu:float, beta:float):
    """
    Stage 4 of the algorithm, takes the set of points generated by Stage 3 and refines them
    Over each iteration, and each point, a search direction is calculated for each objective function.
    The search direction is then used to generate a new point, which is added to the set of points.

    INPUTS
    points: List of starting points
    objFuncs: List of functions to be minimised
    G: List of inequality constraints
    H: List of equality constraints
    tol: Tolerance for stopping criteria
    mu: Penalty parameter
    beta: Step size parameter

    OUTPUTS
    points: List of points estimated to be on the pareto front

    """
    numNonStopped = len(points)
    print("------------------------------")
    print(" ")
    print(f"Stage 4, X length {len(points)}")

    #Set all the points to be not stopped and define the current points as their own reference points
    for p in points:
        temp_p = Point(p.vector)
        temp_p.evaluate(objFuncs)
        p.reference = temp_p
        p.stopped = False

    #Create a list of the "stopped" status of the points
    stoppedCheck = [p.stopped for p in points]

    #While not all the points are stopped, continue to iterate
    while not all(stoppedCheck):
        print("------------------------------")
        T = []
        print(f"Iter {iter}, X length {len(points)}")

        stopCount = len([j == False for j in stoppedCheck])
        print(f"Points not stopped: {stopCount}")

        #for each point
        for i,p in enumerate(points):

            #If the point is not stopped, calculate the search direction and step size
            if  p.stopped == False:
                #Calculate search direction v according to (4.2)
                v,L = search_dir_two(p,G,H,objFuncs)

                #If search direction is too small, stop the point
                if np.linalg.norm(v) < np.power(tol, 1/4):
                    p.stopped = True
                    stoppedCheck[i] = True
                    T.append(p)
                    continue

                #Compute penalty parameter
                pen_val = penalty(L)

                #Find alpha
                alpha = alpha_val_multi(p, np.array(v), G,H, objFuncs,beta,pen_val, mu,tol)

                #If alpha is too small, stop the point
                if alpha < np.sqrt(tol):
                    p.stopped = True
                    stoppedCheck[i] = True
                    T.append(p)
                    continue

                #Generate a new point using the step size and search direction
                p_new = Point(p.vector + alpha * np.array(v))
                p_new.evaluate(objFuncs)
                p_new.reference = p.reference
                T.append(p_new)

            else:
                T.append(p)

        if len(T) == 0:
            break

        #Set of non-dominated points
        points = KungMethod(T)
        #print(f"Non-dom points: {len(points)}")
        stoppedCheck = [p.stopped for p in points]

    return points

###############################################

###############################################

def alpha_val_singular(x:Point, d, g:List[Constraint], h:List[Constraint], f1:Function, beta, penalty_value, mu,tol = 1e-4):
    """
    Calculate the step size alpha using the merit function where
    merit(x + alpha * d) <= merit(x) + mu * alpha * merit_d(x,d)

    INPUTS
    x: Current point
    d: Search direction
    g: List of inequality constraints
    h: List of equality constraints
    f1: Function to be minimised
    beta: Step size parameter
    penalty_value: Penalty parameter
    mu: Penalty parameter
    tol: Tolerance for stopping criteria

    OUTPUTS
    alpha: Step size
    """
    #Get the position vector from the point object
    x_vector = x.vector


    #Starting value of alpha - We find largest alpha in {1, beta, beta^2, ...} such the inequality holds TODO: Check if this is correct
    alpha = 1
    while True:
        #Evaluate the terms of the inequality
        merit1 = merit(x_vector + alpha*d,g,h,f1,penalty_value)
        merit2 = merit(x_vector,g,h,f1,penalty_value) + mu * alpha * merit_d(x_vector,g,h,f1,penalty_value,d,tol)

        if (merit1 <= merit2):# and check:
            break
        alpha = alpha * beta

    return alpha

###############################################

###############################################

def alpha_val_multi(x:Point, d, g:List[Constraint], h:List[Constraint], objFuncs:List[Function],beta,penalty_value, mu,tol = 1e-4):
    """
    Calculate the step size alpha using the merit function where
    merit(x + alpha * d) <= merit(x) + mu * alpha * merit_d(x,d)

    INPUTS
    x: Current point
    d: Search direction
    g: List of inequality constraints
    h: List of equality constraints
    objFuncs: List of Functions to be minimised
    beta: Step size parameter
    penalty_value: Penalty parameter
    mu: Penalty parameter
    tol: Tolerance for stopping criteria

    OUTPUTS
    alpha: Step size
    """
    #Get the position vector from the point object
    x_vector = x.vector

    #Starting value of alpha - We find largest alpha in {1, beta, beta^2, ...} such the inequality holds TODO: Check if this is correct
    alpha = 1
    while True:
        #Evaluate the terms of the inequality
        merit1 = merit_multi(x_vector + alpha*d,g,h,objFuncs,penalty_value)
        merit2 = merit_multi(x_vector,g,h,objFuncs,penalty_value) + mu * alpha * merit_d_multi(x_vector,g,h,objFuncs,penalty_value,d,tol)

        if (merit1 <= merit2):# and check:
            break
        alpha = alpha * beta

    return alpha

###############################################

###############################################

def merit(x,g:List[Constraint],h:List[Constraint],f1:Function, p:float):
    """
    Calculate merit function for a given point x, and a positve penalty
    parameter defined as follows:

    merit(x) = f(x) + penalty * (sum(g_i(x)^+) + sum(|h_i(x)|))

    INPUTS
    x: Current point
    g: List of inequality constraints
    h: List of equality constraints
    f1: Function to be minimised
    p: Penalty parameter

    OUTPUTS
    m: Merit function value
    """


    g_plus = [max(g[i](x),0) for i in range(len(g))]

    m = f1(x) + p* (sum([g_plus[i] for i in range(len(g))])) + p * (sum([np.abs(h[i].func(x)) for i in range(len(h))]))
    return m

###############################################

###############################################

def merit_multi(x,g:List[Constraint],h:List[Constraint],objFuncs:List[Function], p):
    """
    Calculate merit function for a given point x, and a positve penalty
    parameter defined as follows:

    merit(x) = sum(f_i(x)) + penalty * (sum(g_i(x)^+) + sum(|h_i(x)|))

    INPUTS
    x: Current point
    g: List of inequality constraints
    h: List of equality constraints
    objFuncs: List of objective functions to be considered
    p: Penalty parameter

    OUTPUTS
    m: Merit function value
    """
    g_plus = np.array([max(g[i](x),0) for i in range(len(g))])

    sum_f = np.sum(np.array([f(x) for f in objFuncs]))

    m = sum_f + p* (np.sum(g_plus)) + p * (sum([np.abs(h[i](x)) for i in range(len(h))]))

    return m

###############################################

###############################################

def merit_d(x: List,g:List[Constraint],h:List[Constraint],f1:Function, penalty_val,d,tol = 1e-4):
    """
    Evaluate the derivative of the merit function for a given point x as follows:
    merit'(x) = f'(x;d) + penalty_val * (sum(g_i'(x)^+) + sum(|h_i'(x)|))

    INPUTS
    x: Current point
    g: List of inequality constraints
    h: List of equality constraints
    f1: Function to be minimised
    penalty_val: Penalty parameter
    d: Search direction
    tol: Tolerance for stopping criteria

    OUTPUTS
    m_d: Directional derivative of the merit function at x
    """

    G_deriv = [max(np.dot(g[i].evaluate_gradient(x), d),0) if abs(g[i](x)) <= tol else 0 for i in range(len(g))]
    H_deriv = [np.dot(h[i].evaluate_gradient(x) ,d) for i in range(len(h))]

    #Join the above two lists
    dir_deriv = G_deriv + H_deriv

    #Calculate the direction derivative of the objective function
    m_d = np.dot(f1.evaluate_gradient(x),d)

    #TODO:currently the euclidean norm, however may be best to change to sqrt(x^TJJ^t x) where J is the jacobian of the obj function?
    m_d = m_d + penalty_val * (np.linalg.norm(dir_deriv))

    return m_d

###############################################

###############################################

def merit_d_multi(x: List,g:List[Constraint],h:List[Constraint],objFuncs:List[Function], penalty_val,d,tol = 1e-4):
    """
    Evaluate the derivative of the merit function for a given point x as follows:
    merit'(x) = sum(f_i'(x;d)) + penalty_val * (sum(g_i'(x)^+) + sum(|h_i'(x)|))

    INPUTS
    x: Current point
    g: List of inequality constraints
    h: List of equality constraints
    objFuncs: Objective Functions to be minimised
    penalty_val: Penalty parameter
    d: Search direction
    tol: Tolerance for stopping criteria

    OUTPUTS
    m_d: Directional derivative of the merit function at x
    """
    #Evaluate the direcitonal derivative of each constraint function at x

    G_deriv = [max(np.dot(g[i].evaluate_gradient(x), d),0) if abs(g[i](x)) <= tol else 0 for i in range(len(g))]
    H_deriv = [np.dot(h[i].evaluate_gradient(x) ,d) for i in range(len(h))]


    #Join the above two lists
    dir_deriv = G_deriv + H_deriv


    #Calculate the direction derivative of the objective function
    f_d = np.array([np.dot(f.evaluate_gradient(x),d)for f in objFuncs])

    m_d = np.sum(f_d) + penalty_val * (np.linalg.norm(dir_deriv))

    return m_d

###############################################

###############################################
#CALCULATING PENALTY PARAMETER
def penalty(L_Multi:List[float], k:float = 1.1):
    """
    Calculate the penalty parameter, given by the sum of the lagrange multipliers
    for each constraint function found during the search direction step.
    """
    p =  k * np.sum(np.linalg.norm(L_Multi, ord=2))
    return p

###############################################

###############################################
def search_dir(x,g:List[Constraint],h:List[Constraint],f1:Function,n:int):
    """
    Solve the quadratic optimisation problem
    min delta(f)^T * d + 1/2 * d^T * H * d

    note: H is currently the Identity matrix

    INPUTS
    x - current point
    g: List of inequality constraints
    h: List of equality constraints
    f1: Function to be minimised
    n: Index of the function to be minimised

    OUTPUTS
    d: Search direction
    L: Lagrange multipliers for the constraints
    """

    #Value of x
    x_v = x.vector
    #Value of the gradient of the function f1 at x
    df_x = x.eval_d[n]

    # Define the objective function for scipy minimize
    def objective(d):
        return np.dot(df_x, d) + 0.5 * np.dot(d, d)

    # Define the constraints for scipy minimize
    constraints = []
    for f in g:

        constraints.append({'type': 'ineq', 'fun': lambda d, f=f: -f(x_v) - np.dot(f.evaluate_gradient(x_v), d)})

    for f in h:
        constraints.append({'type': 'eq', 'fun': lambda d, f=f: f(x_v) + np.dot(f.evaluate_gradient(x_v), d)})


    # Initial guess for d
    d0 = np.zeros(len(x_v))

    # Solve the optimization problem
    result = minimize(objective, d0,method='trust-constr',constraints=constraints)

    if result.success == False:
        print("-----------SEARCH 1------------")
        print("Optimization failed. Debugging information:")
        print("x:", x_v)

    d_values = result.x
    lagrange = result.v

    return d_values, lagrange

###############################################

###############################################

def search_dir_two(x,g:List[Constraint],h:List[Constraint],func:List[Function]):
    """
    Solve the quadratic optimisation problem
    min delta(f)^T * d + 1/2 * d^T * H * d

    note: H is currently the Identity matrix

    INPUTS
    x - current point
    g: List of inequality constraints
    h: List of equality constraints
    func: List of functions to be minimised

    OUTPUTS
    d: Search direction
    L: Lagrange multipliers for the constraints
    """

    n = len(x.eval_d)
    #Value of x
    x_v = x.vector
    #Reference point


    #Functions at x
    f_x = x.eval_f
    f_x_ref = x.reference.eval_f

    #Gradient at x
    #df_x = x.eval_d

    # Define the objective function for scipy minimize
    def objective(v):
        a = np.array([np.dot(x.eval_d[i],v) for i in range(n)])

        return np.sum(a)+ n* 0.5 * np.dot(v,v)

    # Define the constraints for scipy minimize
    constraints = []

    for f_num in range(n):
        #TODO:think this through is this correct? requirement of - signs
        a = x.eval_d[f_num]

        constraints.append({'type': 'ineq', 'fun': lambda v: - f_x[f_num] + f_x_ref[f_num] -  np.dot(a,v)})

    for f in g:

        constraints.append({'type': 'ineq', 'fun': lambda v, f=f: - f(x_v) - np.dot(f.evaluate_gradient(x_v), v)})

    for f in h:
        constraints.append({'type': 'eq', 'fun': lambda v, f=f: f(x_v) + np.dot(f.evaluate_gradient(x_v), v)})


    # Initial guess for d
    d0 = np.zeros(len(x_v))

    # Solve the optimization problem
    result = minimize(objective, d0,method='trust-constr',constraints=constraints)
    #TODO:CHECK WHY SO MANY ERRORS

    d_values = result.x
    lagrange = result.v

    return d_values, lagrange