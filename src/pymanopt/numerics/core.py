from functools import wraps
from typing import Optional, Sequence

import numpy as np
import scipy.special

from pymanopt.numerics.array_t import array_t


def not_implemented(function):
    @wraps(function)
    def inner(*arguments):
        if isinstance(arguments[0], Sequence):
            type_str = f"Sequence[{type(arguments[0][0])}]"
        else:
            type_str = str(type(arguments[0]))
        raise TypeError(
            f"Function '{function.__name__}' not implemented for arguments of "
            f"type '{type(arguments[0])}'"
            f"type '{type_str}'."
        )

    return inner


class NumericsBackend:
    #########################################

    @not_implemented
    def abs(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def all(self, array: array_t) -> Optional[bool]:  # type: ignore
        pass

    @not_implemented
    def any(self, array: array_t) -> Optional[bool]:  # type: ignore
        pass

    @not_implemented
    def arange(self, start: int, stop: int, step: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arccos(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arccosh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arctan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def arctanh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def argmin(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def argsort(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def array(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def assert_allclose(self, array_a: array_t, array_b: array_t) -> None:  # type: ignore
        pass

    @not_implemented
    def assert_almost_equal(self, array_a: array_t, array_b: array_t) -> None:  # type: ignore
        pass

    @not_implemented
    def assert_array_almost_equal(self, array_a: array_t, array_b: array_t) -> None:  # type: ignore
        pass

    @not_implemented
    def block(self, arrays: Sequence[array_t]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def conjugate(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def cos(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def diag(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def diagonal(self, array: array_t, axis1: int, axis2: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def exp(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def expand_dims(self, array: array_t, axis: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def eye(self, size: int) -> array_t:  # type: ignore
        pass

    # finfo
    # float64

    @not_implemented
    def hstack(self, arrays: Sequence[array_t]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def iscomplexobj(self, array: array_t) -> bool:  # type: ignore
        pass

    @not_implemented
    def isnan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def isrealobj(self, array: array_t) -> bool:  # type: ignore
        pass

    @not_implemented
    def linalg_cholesky(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_det(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_eigh(self, array: array_t) -> tuple[array_t, array_t]:  # type: ignore
        pass

    @not_implemented
    def linalg_expm(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_inv(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_matrix_rank(self, array: array_t) -> int:  # type: ignore
        pass

    @not_implemented
    def linalg_logm(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_norm(self, array: array_t, *args: tuple, **kwargs: dict) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_qr(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_solve(self, array_a: array_t, array_b: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_solve_continuous_lyapunov(
        self,
        array_a: array_t,  # type: ignore
        array_q: array_t,  # type: ignore
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def linalg_svd(
        self,
        array: array_t,  # type: ignore
        *args: tuple,
        **kwargs: dict,
    ) -> tuple[array_t, array_t, array_t]:  # type: ignore
        pass

    @not_implemented
    def log(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def logspace(self, start: int, stop: int, num: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def ndim(self, array: array_t) -> int:  # type: ignore
        pass

    newaxis = None

    @not_implemented
    def ones(self, shape: Sequence[int]) -> array_t:  # type: ignore
        pass

    pi = np.pi

    #   - np.polyfit
    #   - np.polyval

    @not_implemented
    def prod(self, array: array_t) -> float:  # type: ignore
        pass

    @not_implemented
    def random_normal(self, size: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def random_randn(self, size: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def random_uniform(self, size: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def real(self, array: array_t) -> array_t:  # type: ignore
        pass

    # TODO: seterr
    # def seterr(all=None):
    #     np.seterr(all=all)
    #     if all == 'raise':
    #         try:
    #             import torch
    #             torch.autograd.set_detect_anomaly(True)
    #         except ImportError:
    #             pass
    #         try:
    #             import jax
    #             jax.config.update("jax_debug_nans", True)
    #         except ImportError:
    #             pass
    #         try:
    #             import tensorflow as tf
    #             tf.debugging.enable_check_numerics()
    #         except ImportError:
    #             pass

    @not_implemented
    def sin(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def sinc(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def sort(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def spacing(self, array: array_t) -> array_t:  # type: ignore
        pass

    def special_comb(self, n: int, k: int):
        return scipy.special.comb(n, k)

    @not_implemented
    def sqrt(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def squeeze(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def sum(
        self,
        array: array_t,  # type: ignore
        *args: tuple,
        **kwargs: dict,
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tan(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tanh(self, array: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tensordot(self, a: array_t, b: array_t, axes: int) -> array_t:  # type: ignore
        pass

    @not_implemented
    def tile(self, array: array_t, reps: int | Sequence[int]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def trace(
        self,
        array: array_t,  # type:ignore
        *args: tuple,
        **kwargs: dict,
    ) -> array_t:  # type: ignore
        pass

    @not_implemented
    def transpose(self, array: array_t, axes: Sequence[int] | None) -> array_t:  # type: ignore
        pass

    #   - np.vectorize

    @not_implemented
    def vstack(self, arrays: Sequence[array_t]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def where(self, condition: array_t) -> array_t:  # type: ignore
        pass

    @not_implemented
    def zeros(self, shape: Sequence[int]) -> array_t:  # type: ignore
        pass

    @not_implemented
    def zeros_like(self, array: array_t) -> array_t:  # type: ignore
        pass
