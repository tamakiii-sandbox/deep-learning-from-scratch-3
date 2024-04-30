import numpy as np
from numpy import ndarray

class Variable:
    def __init__(self, data: ndarray):
        self.data = data

data = np.array(1.0)
x = Variable(data)
print(x.data)

y = Variable(1.0)
print(y.data)

