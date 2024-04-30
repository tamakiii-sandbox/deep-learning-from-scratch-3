import numpy as np
from numpy import ndarray

class Variable:
    def __init__(self, data: ndarray):
        self.data = data

data = np.array(1.0)
x = Variable(data)
print("x.data:", x.data)

x = np.array(1)
print("x.ndim:", x.ndim)

y = np.array([1, 2, 3])
print("y.ndim:", y.ndim)

z = np.array([
 [1, 2, 3],
 [4, 5, 6],
])
print("z.ndim:", z.ndim)
