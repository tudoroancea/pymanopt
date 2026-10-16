import numpy as np

from pymanopt.backends import _numba, _numpy_autodiff
from pymanopt.backends.numpy_backend import NumpyBackend


class NumbaBackend(NumpyBackend):
    """NumPy backend whose functions and derivatives are compiled with numba.

    Experimental. Programs numba cannot compile fall back to plain NumPy with
    a warning.
    """

    _compile = staticmethod(_numba.compile)

    def __init__(self, dtype: type = np.float64):
        super().__init__(dtype)

    def __repr__(self):
        return f"NumbaBackend(dtype={self.dtype})"

    def compile_function(self, function):
        return _numpy_autodiff.compiled(function, self._compile)
