import warnings

import numpy as np
import pytest

from .test_numpy_autodiff import COMPLEX_CASES, REAL_CASES, _check

numba_backend = pytest.importorskip("pymanopt.backends.numba_backend")

# Cases using operations the numba compiler does not lower yet, which fall
# back to the NumPy programs with a warning.
FALLBACKS = {"reductions", "indexing", "matmul"}


def _check_numba(case, dtype):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check(*case, dtype, backend=numba_backend.NumbaBackend)
    return [str(w.message) for w in caught if "numba" in str(w.message)]


@pytest.mark.parametrize("name", REAL_CASES)
def test_real(name):
    assert bool(_check_numba(REAL_CASES[name], np.float64)) == (
        name in FALLBACKS
    )


@pytest.mark.parametrize("name", COMPLEX_CASES)
def test_complex(name):
    assert not _check_numba(COMPLEX_CASES[name], np.complex128)


def test_functions_are_compiled():
    backend = numba_backend.NumbaBackend()
    cost = backend.compile_function(lambda x, y: np.sum(x**2 * y))
    gradient = backend.compile_function(lambda x, y: [2 * x * y, x**2])
    x, y = np.arange(3.0), np.ones(3)
    np.testing.assert_allclose(cost(x, y), 5.0)
    g_x, g_y = gradient(x, y)
    np.testing.assert_allclose(g_x, 2 * x)
    np.testing.assert_allclose(g_y, x**2)
