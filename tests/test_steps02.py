from typing import Any
import numpy as np
from steps.step01 import Variable
from steps.step02 import Function

def test_simple_function() -> None:
    x: Variable = Variable(np.array(10))
    f: Function = Function()
    y: Any = f(x)
    assert isinstance(y, Variable)
    assert y.data == np.array(100)
