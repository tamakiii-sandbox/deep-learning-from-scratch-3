from steps.step08 import Square, Exp, Variable

Variable = Variable


def square(x):
    f = Square()
    return Square()(x)


def exp(x):
    f = Exp()
    return f(x)
