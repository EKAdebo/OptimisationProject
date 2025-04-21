class Function:
    #Defines a function, and defined functions to call it on different points.
    def __init__(self, func, func_g, name):
        #Initialises the function with a function, its gradient and a name
        self.func = func
        self.func_d = func_g
        self.name = name

    def __call__(self, vector):
        #When calling the function, it evaluates the function at a point. EG func(x)
        return self.func(vector)

    def evaluate_gradient(self, vector):
        #Evaluates the gradient of the function at a point. E.G. func.evaluate_gradient(x)
        return self.func_d(vector)

class Constraint(Function):
    #A child class of Function, that defines a constraint function
    def __init__(self, func, func_g, name, equality=False):
        super().__init__(func, func_g, name)

        #Defines if the constraint is an inequality or equality constraint
        self.equality = equality