import numpy as np
import pytest
import torch
from pymanopt.numerics import NumpyNumericsBackend, PytorchNumericsBackend

backend_np64 = NumpyNumericsBackend(dtype=np.float64)
backend_np32 = NumpyNumericsBackend(dtype=np.float32)
backend_pt64 = PytorchNumericsBackend(dtype=torch.float64)
backend_pt32 = PytorchNumericsBackend(dtype=torch.float32)
all_backends = [backend_np64, backend_np32, backend_pt64, backend_pt32]


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
        ([1, 0, 1], [1, 0, 1]),
        ([-127, -4], [127, 4]),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_abs(input, expected_output, backend):
    backend.assert_allclose(
        backend.abs(backend.array(input)), backend.array(expected_output)
    )


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([True, False, True], False),
        ([True, True, True], True),
        ([], True),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_all(input, expected_output, backend):
    assert backend.all(backend.array(input)) == expected_output


@pytest.mark.parametrize(
    "input, expected_output",
    [
        ([False, False, False], False),
        ([True, False, True], True),
        ([True, True, False], True),
        ([], False),
    ],
)
@pytest.mark.parametrize("backend", all_backends)
def test_any(input, expected_output, backend):
    assert backend.any(backend.array(input)) == expected_output
