import math
import numpy as np
import pytest
import unittest
from newton import deriv, optimize

def test_success():
    assert math.isclose(optimize(1, np.sin)[0], math.pi / 2), "Newton's method failed for finding the local maximum of sine at (pi / 2, 1)"

def test_failure():
    assert optimize(1, np.exp)[0] == "Newton's method failed to converge", "Newton's method behaves unexpectedly for e^x starting at 1"