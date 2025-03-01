import numpy as np
from typing import List
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy import concat, concatenate

from Point import Point
from Functions import Function, Constraint
from KM_Algo import KungMethod


###############################################

###############################################

#Number of iterations
def main(objFuncs:List[Function], G:List[Constraint], H:List[Constraint],dimPoints: int,npoints: int = 100,
         iter:int = 50, mu:float = 0.25, beta:float = 0.5, tol:float = 1e-4):
    """
    Runs an algortihm estimating a set of points on the pareto front
    points: List of starting points
    objFuncs: List of objective functions
    G: List of inequality constraints
    H: List of equality constraints
    iter: Number of iterations
    tol: Tolerance
    """
    #Generates a set of original points
    X0 = generatePoints(npoints, dimPoints, objFuncs)

    #Number of objective functions
    numFuncs = len(objFuncs)

    #Creates a copy of the original set of poitns for comaprison at the end
    points = X0.copy()

    for i in range(iter):
        T = []
        #FOR EACH POINT IN X0 THE SET OF RANDOM POINTS
        for p in points:

            #CHEK IF POINT IS STOPPED
            if p.stopped != True:

                #FOR EACH OBJECTIVE FUNCTION
                for j in range(0,numFuncs):

                    #COMPUTE SEARCH DIRECTION
                    d = search_dir(p,G,H,objFuncs[j],j)

                    #IF SEARCH DIRECTION IS TOO SMALL,CONTINUE TO EVAL NEXT OBJ FUNC
                    if np.linalg.norm(d) < np.power(tol, 1/4):
                        continue

                    c,psi = penalty(G,H,p,objFuncs[j],mu)

                    alpha = alpha_val(p, np.array(d), beta, c, psi, G,H, objFuncs[j], mu)

                    if np.linalg.norm(d) < np.sqrt(tol):
                        continue

                    p_new = Point(p.vector + alpha * np.array(d))
                    p_new.evaluate(objFuncs)

                    T.append(p_new)

                p.stopped = True

        if len(T) == 0:
            break

        points.extend(T)

        print(f"Iter {i}, X length {len(points)}")

        points = KungMethod(points)

    if dimPoints == 2:
        displayResults(X0, points)

    return X0, points

###############################################

###############################################

def displayResults(X0:List[Point], points:List[Point]):

    #Displays a plot of the original evaluations of the points vs the points that are found by the algorithm.

    plt.subplot(2,2,1)
    for p in X0:
        plt.scatter(p.eval_f[0], p.eval_f[1], color='red',label = 'Original Points' if 'Original Points' not in plt.gca().get_legend_handles_labels()[1] else "")
    for p in points:
        plt.scatter(p.eval_f[0], p.eval_f[1], color='blue',label = 'Non-Dominated Points' if 'Non-Dominated Points' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('ParetoFront')
    plt.legend()

    plt.subplot(2,2,2)
    for p in points:
        plt.scatter(p.eval_f[0], p.eval_f[1], color='blue')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title('ParetoFront')


    plt.subplot(2,2,3)
    for p in X0:
        plt.scatter(p.vector[0], p.vector[1], color='red',label = 'Original Points' if 'Original Points' not in plt.gca().get_legend_handles_labels()[1] else "")
    for p in points:
        plt.scatter(p.vector[0], p.vector[1], color='blue',label = 'Non-Dominated Points' if 'Non-Dominated Points' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('ParetoFront')
    plt.legend()

    plt.subplot(2,2,4)
    for p in points:
        plt.scatter(p.vector[0], p.vector[1], color='blue')

    plt.show()
###############################################

###############################################

def generatePoints(npoints:int, dim:int, objFuncs:List[Function]):
    """
    This function generates a set of random points
    npoints: Number of points to generate
    dim: Dimension of the points
    objFuncs: List of objective functions
    """

    Xorg = []
    for i in range(npoints):
        x = Point(10*np.random.rand(dim))
        x.evaluate(objFuncs)
        Xorg.append(x)
    return Xorg

###############################################

###############################################

def alpha_val(x:Point, d, beta,c,psi, g:List[Constraint], h:List[Constraint], f1:Function, mu):
    """
    Calculate the step size for a given point x
    as described in:
    A SUPERLINEARLY CONVERGENT ALGORITHM FOR CONSTRAINED OPTIMIZATION PROBLEMS*

    Args: x - current point
          beta - step size reduction factor
          g - list of functions that define inequality <= constraints
          h - list of functions that define equality == constraints
          f1 - function to be minimised

    Returns: alpha
    """
    x_vector = x.vector
    #Calculate the step size
    pen1 = f1(x_vector)
    pen = pen1  + c * psi
    alpha = 1
    while True:
        merit1 = merit(x_vector + alpha*d,g,h,f1,pen)
        merit2 = merit(x_vector,g,h,f1,pen) + mu * alpha * merit_d(x_vector,g,h,f1,psi,d)

        if merit1 <= merit2:
            break
        alpha = alpha * beta

    return alpha

###############################################

###############################################

#merit function derivative
def merit_d(x: List,g:List[Constraint],h:List[Constraint],f1:Function, psi,d):
      """
      Calculate the merit function for a given point x
      as described in:
      A SUPERLINEARLY CONVERGENT ALGORITHM FOR CONSTRAINED OPTIMIZATION PROBLEMS*

      Args: x - current point
            g - list of functions that define inequality <= constraints
            h - list of functions that define equality == constraints
            f1 - function to be minimised
            mu - penalty parameter

      Returns: merit function value
      """

      G_deriv = [g[i].evaluate_gradient(x) * d for i in range(len(g))]
      H_deriv = [h[i].evaluate_gradient(x) * d if h[i](x) == 0 else 0 for i in range(len(h))]


      dir_deriv = G_deriv + H_deriv


      m_d = np.dot(f1.evaluate_gradient(x),d)
      m_d = m_d + psi * (np.linalg.norm(dir_deriv))

      return m_d

###############################################

###############################################
# #merit function
def merit(x,g:List[Constraint],h:List[Constraint],f1:Function, p):
      """
      Calculate the merit function for a given point x
      as described in:
      A SUPERLINEARLY CONVERGENT ALGORITHM FOR CONSTRAINED OPTIMIZATION PROBLEMS*

      Args: x - current point
            g - list of functions that define inequality <= constraints
            h - list of functions that define equality == constraints
            f1 - function to be minimised
            mu - penalty parameter



      Returns: merit function value
      """

      m = f1(x) + p* (sum([g[i].func(x) for i in range(len(g))])) + p * (sum([np.abs(h[i].func(x)) for i in range(len(h))]))
      return m

###############################################

###############################################
#CALCULATING PENALTY PARAMETER
def penalty(g: List[Constraint], h: List[Constraint], x: Point,f1:Function,beta, b: float = 0.001):
      """
      Calculate the penalty parameter for a given point x
      as described in:
      A SUPERLINEARLY CONVERGENT ALGORITHM FOR CONSTRAINED OPTIMIZATION PROBLEMS*

      Args: g - list of functions that define inequality <= constraints
            h - list of functions that define equality == constraints
            x - current point
            mu - penalty parameter

      Returns: mu
      """

      x_vector = x.vector


      #TODO maybe calc this as input to avoid repeated calculations
      #Matrix of deriv calculations
      G_d = [g[i].evaluate_gradient(x_vector) for i in range(len(g))]
      H_d = [h[i].evaluate_gradient(x) for i in range(len(h))]

      #TODO move next calc out of function and pass as input , place before main loop in order to avoid repeated calculations
      #Max value of g and h func evaled at x
      G = [g[i](x_vector) for i in range(len(g))]
      H = [np.abs(h[i].func(x_vector)) for i in range(len(h))]
      psi = max(max(G, default=float('-inf')), max(H, default=float('-inf')))


      def minFunc(l_mu):
            l = l_mu[:len(G)]
            mu = l_mu[len(G):]
            term1 = np.linalg.norm(f1.evaluate_gradient(x_vector) + np.dot(np.array(G_d).T,l) + np.dot(np.array(H_d).T,mu))**2
            term2 = np.sum([((psi -G[i])**2)*(l[i]**2) for i in range(len(G))])
            term3 = np.sum([((psi -np.abs(H[i]))**2)*(mu[i]**2) for i in range(len(H))])
            return term1 + term2 + term3

      initial_guess = np.zeros(len(G) + len(H))
      result = minimize(minFunc, initial_guess)

      l_opt = result.x[:len(G)]
      mu_opt = result.x[len(G):]
      #Calculate the penalty parameter

      c = np.sum(l_opt) + sum(np.linalg.norm(vector) for vector in mu_opt)
      c = max(c + b, b)

      #Penalty func = f(x) + c * psi(x)
      f1_x = f1(x_vector)

      return c,psi

###############################################

###############################################

def search_dir(x,g:List[Constraint],h:List[Constraint],f1:Function,n:int):
    """
    Solve the quadratic optimisation problem
    min delta(f)^T * d + 1/2 * d^T * H * d

    Args: x - current point
          g - list of functions that define inequality <= constraints
          h - list of functions that define equality == constraints
          f1_d - gradient of the function f1 at x


    Returns: d
    Note: H is currently the Identity matrix

    """

    x_v = x.vector
    df_x = x.eval_d[n]
    # Define the objective function for scipy minimize
    def objective(d):
        return np.dot(df_x, d) + 0.5 * np.dot(d, d)

    # Define the constraints for scipy minimize
    constraints = []
    for f in g:
        constraints.append({'type': 'ineq', 'fun': lambda d, f=f: f(x_v) + np.dot(f.evaluate_gradient(x_v), d)})

    for f in h:
        constraints.append({'type': 'eq', 'fun': lambda d, f=f: f(x_v) + np.dot(f.evaluate_gradient(x_v), d)})

    # Initial guess for d
    d0 = np.zeros(len(x_v))

    # Solve the optimization problem
    result = minimize(objective, d0, constraints=constraints)

    d_values = result.x

    return d_values
