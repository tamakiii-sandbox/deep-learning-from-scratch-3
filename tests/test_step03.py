import pytest
from typing import Any
import numpy as np

from steps.step03 import Exp
from steps.step02 import Variable


def test_exp_function():
    # Create a Variable instance with a scalar value
    x = Variable(np.array(2.0))  # np.exp(2) is about 7.389056
    exp_function = Exp()

    # Apply the Exp function
    result = exp_function(x)

    # Check that the result is correct
    expected = np.exp(2.0)
    np.testing.assert_almost_equal(
        result.data,
        expected,
        decimal=5,
        err_msg="Exp function did not compute correctly",
    )


def test_exp_function_with_negative_input():
    # Test with negative input
    x = Variable(np.array(-1.0))  # np.exp(-1) is about 0.367879
    exp_function = Exp()

    # Apply the Exp function
    result = exp_function(x)

    # Check that the result is correct
    expected = np.exp(-1.0)
    np.testing.assert_almost_equal(
        result.data,
        expected,
        decimal=5,
        err_msg="Exp function did not handle negative input correctly",
    )
