import numpy as np
from numpy import ndarray


class Variable:
    def __init__(self, data: ndarray):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: callable):
        self.creator = func

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

class Square(Function):
    def forward(self, x: float):
        y = x ** 2
        return y

    def backward(self, gy: float):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x: float):
        y = np.exp(x)
        return y

    def backward(self, gy: float):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def main():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C

if __name__ == "__main__":
    main()
