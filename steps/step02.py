import numpy as np
from numpy import ndarray


class Variable:
    def __init__(self, data: ndarray):
        self.data = data


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = x**2
        output = Variable(y)
        return output


x = Variable(np.array(10))
f = Function()
y = f(x)

print(type(y))
print(y.data)
