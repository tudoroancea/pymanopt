from typing import Sequence
from numbers import Number

import numpy as np
import numpy.testing as np_testing
import scipy.linalg
import torch

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class PytorchNumericsBackend(NumericsBackend):
    _dtype: torch.dtype

    def __init__(self, dtype=torch.float64):
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"PytorchNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################

    def abs(self, array: torch.Tensor) -> torch.Tensor:
        return torch.abs(array)

    def all(self, array: torch.Tensor) -> bool:
        return bool(torch.all(array).item())

    def any(self, array: torch.Tensor) -> bool:
        return bool(torch.any(array).item())

    def arange(self, start: int, stop: int, step: int) -> torch.Tensor:
        return torch.arange(start, stop, step)

    def arccos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccos(array)

    def arccosh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arccosh(array)

    def arctan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctan(array)

    def arctanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.arctanh(array)

    def argmin(self, array: torch.Tensor):
        return torch.argmin(array)

    def argsort(self, array: torch.Tensor):
        return torch.argsort(array)

    def array(self, array: array_t) -> torch.Tensor:  # type: ignore
        return (
            array
            if isinstance(array, torch.Tensor) and array.dtype == self.dtype
            else torch.as_tensor(array, dtype=self.dtype)
        )

    def assert_allclose(self, array_a: torch.Tensor, array_b: torch.Tensor) -> None:
        np_testing.assert_allclose(
            array_a.cpu().detach().numpy(), array_b.cpu().detach().numpy()
        )

    def assert_almost_equal(
        self, array_a: torch.Tensor | Number, array_b: torch.Tensor | Number
    ) -> None:
        # TODO: check if the intputs are tensors or floats
        np_testing.assert_almost_equal(
            array_a if isinstance(array_a, Number) else array_a.item(),
            array_b if isinstance(array_b, Number) else array_b.item(),
        )

    def assert_array_almost_equal(
        self, array_a: torch.Tensor, array_b: torch.Tensor
    ) -> None:
        np_testing.assert_array_almost_equal(
            array_a.cpu().detach().numpy(), array_b.cpu().detach().numpy()
        )

    def block(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(arrays)

    def conjugate(self, array: torch.Tensor) -> torch.Tensor:
        return torch.conj(array)

    def cos(self, array: torch.Tensor) -> torch.Tensor:
        return torch.cos(array)

    def diag(self, array: torch.Tensor) -> torch.Tensor:
        return torch.diag(array)

    def diagonal(self, array: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        return torch.diagonal(array, dim1=axis1, dim2=axis2)

    def exp(self, array: torch.Tensor) -> torch.Tensor:
        return torch.exp(array)

    def expand_dims(self, array: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.unsqueeze(array, dim=axis)
        # if isinstance(axis, int):
        #     axis = [axis]
        #
        # # Normalize axis values for negative indices
        # positive_axis = list()
        # for ax in axis:
        #     if ax >= 0:
        #         positive_axis.append(ax)
        # negative_axis = list()
        # for ax in axis:
        #     if ax < 0:
        #         negative_axis.append(ax)
        #
        # # Sort the axis list
        # positive_axis.sort()
        # negative_axis.sort()
        # negative_axis = negative_axis[::-1]
        #
        # for i in range(len(positive_axis)):
        #     tensor = torch.unsqueeze(tensor, dim=positive_axis[i])
        #
        # for i in range(len(negative_axis)):
        #     dim = tensor.ndim + negative_axis[i] + 1
        #     tensor = torch.unsqueeze(tensor, dim=dim)
        #
        # return tensor

    def eye(self, size: int) -> torch.Tensor:
        return torch.eye(size)

    def hstack(self, arrays: list[torch.Tensor]) -> torch.Tensor:
        return torch.hstack(arrays)

    def iscomlexobj(self, array: torch.Tensor) -> bool:
        return torch.is_complex(array)

    def isnan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.isnan(array)

    def isrealobj(self, array: torch.Tensor) -> bool:
        return bool(torch.all(torch.isreal(array)).item())

    def linalg_cholesky(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(array)

    def linalg_det(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.det(array)

    def linalg_eigh(self, array: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(array)

    def linalg_expm(self, array: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(array)

    def linalg_inv(self, array: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(array)

    def linalg_matrix_rank(self, array: torch.Tensor) -> int:
        return int(torch.linalg.matrix_rank(array))

    def linalg_logm(self, array: torch.Tensor) -> torch.Tensor:
        # logm is not implemented in PyTorch
        # see: https://github.com/pytorch/pytorch/issues/9983
        # hence we use the SciPy implementation
        return self.array(scipy.linalg.logm(array.cpu().detach().numpy()))

    def linalg_norm(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> torch.Tensor:
        return torch.linalg.norm(array, *args, **kwargs)

    def linalg_qr(self, array: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.qr(array)

    def linalg_solve(
        self, array_a: torch.Tensor, array_b: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.solve(array_a, array_b)

    def linalg_solve_continuous_lyapunov(
        self, array_a: torch.Tensor, array_q: torch.Tensor
    ) -> torch.Tensor:
        # solve_continuous_lyapunov is not implemented in PyTorch so we use the
        # SciPy implementation
        return torch.tensor(
            scipy.linalg.solve_continuous_lyapunov(
                array_a.cpu().detach().numpy(), array_q.cpu().detach().numpy()
            )
        )

    def linalg_svd(
        self, array: torch.Tensor, *args: tuple, **kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.linalg.svd(array, *args, **kwargs)

    def log(self, array: torch.Tensor) -> torch.Tensor:
        return torch.log(array)

    def logspace(self, start: int, stop: int, num: int) -> torch.Tensor:
        return torch.logspace(start, stop, num)

    def ndim(self, array: torch.Tensor) -> int:
        return array.ndim

    def ones(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.ones(shape)

    def prod(self, array: torch.Tensor) -> float:
        return torch.prod(array).item()

    def random_normal(
        self, loc: float, scale: float, size: Sequence[int]
    ) -> torch.Tensor:
        return torch.normal(mean=loc, std=scale, size=size)

    def random_randn(self, *dims: int) -> torch.Tensor:
        return torch.randn(dims)

    def random_uniform(self, size: int) -> torch.Tensor:
        return torch.rand(size)

    def real(self, array: torch.Tensor) -> torch.Tensor:
        return torch.real(array)

    def sin(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sin(array)

    def sinc(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sinc(array)

    def sort(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sort(array).values

    def spacing(self, array: torch.Tensor) -> torch.Tensor:
        # spacing is not implemented in PyTorch so we use the NumPy implementation
        return self.array(np.spacing(array.cpu().detach().numpy()))

    def sqrt(self, array: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(array)

    def squeeze(self, array: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(array)

    def sum(self, array: torch.Tensor, *args: tuple, **kwargs: dict) -> torch.Tensor:
        return torch.sum(array, *args, **kwargs)

    def tan(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tan(array)

    def tanh(self, array: torch.Tensor) -> torch.Tensor:
        return torch.tanh(array)

    def tensordot(self, a: torch.Tensor, b: torch.Tensor, axes: int) -> torch.Tensor:
        return torch.tensordot(a, b, dims=axes)

    def tile(self, array: torch.Tensor, reps: int | Sequence[int]) -> torch.Tensor:
        return torch.tile(array, [reps] if isinstance(reps, int) else reps)

    def trace(self, array: torch.Tensor, *args: tuple, **kwargs: dict) -> torch.Tensor:
        return torch.trace(array, *args, **kwargs)

    def transpose(
        self, array: torch.Tensor, axes: tuple[int] | list[int] | None = None
    ) -> torch.Tensor:
        if axes is None:
            return array
        if len(axes) != 2:
            raise ValueError("Expected axes to be a tuple of length 2.")
        return torch.transpose(array, dim0=axes[-2], dim1=axes[-1])

    def vstack(self, arrays: tuple[torch.Tensor] | list[torch.Tensor]) -> torch.Tensor:
        return torch.vstack(arrays)

    def where(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(condition)

    def zeros(self, shape: list[int]) -> torch.Tensor:
        return torch.zeros(shape)

    def zeros_like(self, array: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(array)
