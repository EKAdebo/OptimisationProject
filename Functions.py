class Function:
    def __init__(self, func, func2, name):
        self.func = func
        self.func_d = func2
        self.name = name

    def __call__(self, vector):
        return self.func(vector)
    
    def evaluate_gradient(self, vector):
        return self.func_d(vector)
    
class Constraint(Function):
    def __init__(self, func, func2, name, equality=False):
        super().__init__(func, func2, name)
        self.equality = equality