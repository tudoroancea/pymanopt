from typing import Any, Sequence

import numpy as np
import numpy.testing as np_testing
import packaging.version as pv
import scipy
import scipy.linalg

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend

np_array_t = np.ndarray


class NumpyNumericsBackend(NumericsBackend):
    _dtype: np.dtype

    def __init__(self, dtype=np.float64):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"NumpyNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: np_array_t) -> np_array_t:
        return np.abs(array)

    def all(self, array: np_array_t) -> bool:
        return np.all(array).item()

    def any(self, array: np_array_t) -> bool:
        return np.any(array).item()

    def arange(self, start: int, stop: int, step: int) -> np_array_t:
        return np.arange(start, stop, step)

    def arccos(self, array: np_array_t) -> np_array_t:
        return np.arccos(array)

    def arccosh(self, array: np_array_t) -> np_array_t:
        return np.arccosh(array)

    def arctan(self, array: np_array_t) -> np_array_t:
        return np.arctan(array)

    def arctanh(self, array: np_array_t) -> np_array_t:
        return np.arctanh(array)

    def argmin(self, array: np_array_t):
        return np.argmin(array)

    def argsort(self, array: np_array_t):
        return np.argsort(array)

    def array(self, array: array_t) -> np_array_t:  # type: ignore
        return (
            array
            if isinstance(array, np.ndarray) and array.dtype == self.dtype
            else np.array(array, dtype=self.dtype)
        )

    def assert_allclose(self, array_a: np_array_t, array_b: np_array_t) -> None:
        return np_testing.assert_allclose(array_a, array_b)

    def assert_almost_equal(self, array_a: np_array_t, array_b: np_array_t) -> None:
        return np_testing.assert_almost_equal(array_a, array_b)

    def assert_array_almost_equal(
        self, array_a: np_array_t, array_b: np_array_t
    ) -> None:
        return np_testing.assert_array_almost_equal(array_a, array_b)

    def block(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.block(arrays)

    def conjugate(self, array: np_array_t) -> np_array_t:
        return np.conjugate(array)

    def cos(self, array: np_array_t) -> np_array_t:
        return np.cos(array)

    def diag(self, array: np_array_t) -> np_array_t:
        return np.diag(array)

    def diagonal(self, array: np_array_t, axis1: int, axis2: int) -> np_array_t:
        return np.diagonal(array, axis1, axis2)

    def exp(self, array: np_array_t) -> np_array_t:
        return np.exp(array)

    def expand_dims(self, array: np_array_t, axis: int) -> np_array_t:
        return np.expand_dims(array, axis)

    def eye(self, size: int) -> np_array_t:
        return np.eye(size)

    def hstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.hstack(arrays)

    def iscomplexobj(self, array: np_array_t) -> bool:
        return np.iscomplexobj(array)

    def isnan(self, array: np_array_t) -> np_array_t:
        return np.isnan(array)

    def isrealobj(self, array: np_array_t) -> bool:
        return np.isrealobj(array)

    def linalg_cholesky(self, array: np_array_t) -> np_array_t:
        return np.linalg.cholesky(array)

    def linalg_det(self, array: np_array_t) -> np_array_t:
        return np.linalg.det(array)

    def linalg_eigh(self, array: np_array_t) -> tuple[np_array_t, np_array_t]:
        return np.linalg.eigh(array)

    def linalg_expm(self, array: np_array_t) -> np_array_t:
        # Scipy 1.9.0 added support for calling scipy.linalg.expm on stacked matrices.
        if pv.parse(scipy.version.version) >= pv.parse("1.9.0"):
            scipy_expm = scipy.linalg.expm
        else:
            scipy_expm = np.vectorize(scipy.linalg.expm, signature="(m,m)->(m,m)")
        return scipy_expm(array)

    def linalg_inv(self, array: np_array_t) -> np_array_t:
        return np.linalg.inv(array)

    def linalg_logm(self, array: np_array_t) -> np_array_t:
        return np.vectorize(scipy.linalg.logm, signature="(m,m)->(m,m)")(array)

    def linalg_matrix_rank(self, array: np_array_t) -> int:
        return np.linalg.matrix_rank(array)

    def linalg_norm(self, array: np_array_t, *args: Any, **kwargs: Any) -> np_array_t:
        return np.linalg.norm(array)  # type: ignore

    def linalg_qr(self, array: np_array_t) -> tuple[np_array_t, np_array_t]:
        return np.linalg.qr(array)

    def linalg_solve(self, array_a: np_array_t, array_b: np_array_t) -> np_array_t:
        return np.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: np_array_t, array_q: np_array_t
    ) -> np_array_t:
        return scipy.linalg.solve_continuous_lyapunov(array_a, array_q)

    def linalg_svd(
        self, array: np_array_t, *args: Any, **kwargs: Any
    ) -> tuple[np_array_t, np_array_t, np_array_t]:
        return scipy.linalg.svd(array)

    def log(self, array: np_array_t) -> np_array_t:
        return np.log(array)

    def logspace(self, start: int, stop: int, num: int) -> np_array_t:
        return np.logspace(start, stop, num)

    def ndim(self, array: np_array_t) -> int:
        return array.ndim

    def ones(self, shape: Sequence[int]) -> np_array_t:
        return np.ones(shape)

    def prod(self, array: np_array_t) -> float:
        return np.prod(array)  # type: ignore

    def random_normal(
        self, loc: float, scale: float, size: Sequence[int]
    ) -> np_array_t:
        return np.random.normal(loc=loc, scale=scale, size=size)

    def random_randn(self, *dims: int) -> np_array_t:
        return np.random.randn(*dims)

    def random_uniform(self, size: int) -> np_array_t:
        return np.random.uniform(size=size)

    def real(self, array: np_array_t) -> np_array_t:
        return np.real(array)

    def sin(self, array: np_array_t) -> np_array_t:
        return np.sin(array)

    def sinc(self, array: np_array_t) -> np_array_t:
        return np.sinc(array)

    def sort(self, array: np_array_t) -> np_array_t:
        return np.sort(array)

    def spacing(self, array: np_array_t) -> np_array_t:
        return np.spacing(array)  # type: ignore

    def sqrt(self, array: np_array_t) -> np_array_t:
        return np.sqrt(array)

    def squeeze(self, array: np_array_t) -> np_array_t:
        return np.squeeze(array)

    def sum(self, array: np_array_t, *args: Any, **kwargs: Any) -> np_array_t:
        return np.sum(array, *args, **kwargs)  # type: ignore

    def tan(self, array: np_array_t) -> np_array_t:
        return np.tan(array)

    def tanh(self, array: np_array_t) -> np_array_t:
        return np.tanh(array)

    def tensordot(self, a: np_array_t, b: np_array_t, axes: int) -> np_array_t:
        return np.tensordot(a, b, axes=axes)

    def tile(self, array: np_array_t, reps: int | Sequence[int]) -> np_array_t:
        return np.tile(array, reps)

    def trace(self, array: np_array_t, *args: tuple, **kwargs: dict) -> np_array_t:
        return np.trace(array, *args, **kwargs)  # type: ignore

    def transpose(
        self, array: np_array_t, axes: Sequence[int] | None = None
    ) -> np_array_t:
        return np.transpose(array, axes)

    def vstack(self, arrays: Sequence[np_array_t]) -> np_array_t:
        return np.vstack(arrays)

    def where(self, condition: np_array_t) -> np_array_t:
        return np.where(condition)  # type: ignore

    def zeros(self, shape: Sequence[int]) -> np_array_t:
        return np.zeros(shape)
