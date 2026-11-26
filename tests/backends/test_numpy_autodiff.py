import autograd.numpy as anp
import numpy as np
import pytest
from numpy import testing as np_testing

from pymanopt.backends import _numpy_autodiff as autodiff
from pymanopt.backends.autograd_backend import AutogradBackend
from pymanopt.backends.numpy_backend import NumpyBackend


def _positive(x):
    return np.abs(x) + 0.5


def _bounded(x):
    return np.tanh(x) * 0.9


def _spd(x):
    return x @ x.T + 3 * np.eye(x.shape[0])


A = np.random.default_rng(0).normal(size=(4, 4))

# Each case is (cost, argument shapes, argument transformations). Costs are
# written against a NumPy-like module so that the same code runs on autograd,
# which serves as reference.
REAL_CASES = {
    "arithmetic": (
        lambda xp, x, y: xp.sum(x * y + x / (y**2 + 1) - y + 2.0 - x**3),
        [(3, 4), (4,)],
        None,
    ),
    "exp_log": (
        lambda xp, x: (
            xp.sum(xp.exp(x) + xp.log(x) + xp.log1p(x))
            + xp.sum(xp.expm1(x) + xp.log2(x) + xp.log10(x))
        ),
        [(5,)],
        [_positive],
    ),
    "roots_powers": (
        lambda xp, x, y: (
            xp.sum(xp.sqrt(x) + xp.square(x) + 1 / x)
            + xp.sum(x**y + 2.0**x + x**0.5)
        ),
        [(5,), (5,)],
        [_positive, None],
    ),
    "trigonometric": (
        lambda xp, x: (
            xp.sum(xp.sin(x) * xp.cos(x) + xp.tan(x))
            + xp.sum(xp.sinh(x) + xp.cosh(x) + xp.tanh(x))
        ),
        [(5,)],
        [_bounded],
    ),
    "inverse_trigonometric": (
        lambda xp, x: xp.sum(xp.arcsin(x) + xp.arccos(x) * xp.arctan(x)),
        [(5,)],
        [_bounded],
    ),
    "abs_maximum_where_clip": (
        lambda xp, x, y: (
            xp.sum(xp.abs(x) * xp.maximum(x, y))
            + xp.sum(xp.minimum(x, 0.3) + xp.where(x > 0, x**2, -y))
            + xp.sum(xp.clip(x, -0.5, 0.5) ** 2)
        ),
        [(6,), (6,)],
        None,
    ),
    "reductions": (
        lambda xp, x: (
            xp.sum(xp.sum(x, axis=0) ** 2)
            + xp.sum(xp.sum(x, axis=1, keepdims=True) ** 3)
            + xp.sum(xp.mean(x, axis=(0, 2)) ** 2)
            + xp.max(x, axis=1).sum()
            + xp.min(x)
            + x.max()
            + xp.mean(x)
        ),
        [(3, 4, 2)],
        None,
    ),
    "shapes": (
        lambda xp, x: (
            xp.sum(xp.reshape(x, (4, 3)) @ x)
            + xp.sum(xp.swapaxes(x[None], 0, 2) ** 2)
            + xp.sum(xp.expand_dims(x, 1) * x)
            + xp.sum(xp.squeeze(x[:, :1]) ** 3)
            + xp.sum(xp.ravel(x) ** 2)
            + xp.sum(x.T.reshape(2, 6) ** 2)
            + xp.sum(xp.transpose(x[None], (2, 0, 1)) ** 3)
        ),
        [(3, 4)],
        None,
    ),
    "indexing": (
        lambda xp, x: (
            xp.sum(x[1:, ::2] ** 2)
            + x[0, 1] * x[2, 3]
            + xp.sum(x[[0, 0, 2]] ** 3)
            + xp.sum(x[..., -1])
            + xp.sum(x[:, None, 1] ** 2)
            + x[xp.argmax(x[:, 0]), 0] ** 2
        ),
        [(3, 4)],
        None,
    ),
    "joining": (
        lambda xp, x, y: (
            xp.sum(xp.concatenate([x, 2 * x], axis=1) ** 2)
            + xp.sum(xp.stack([x, y], axis=1) ** 3)
            + xp.sum(xp.append(x, y) ** 2)
        ),
        [(3, 3), (3, 3)],
        None,
    ),
    "diagonals": (
        lambda xp, x, v: (
            xp.sum(xp.diag(x) ** 2)
            + xp.sum(xp.diag(v, 1) @ x)
            + xp.trace(x @ x)
            + xp.sum(xp.triu(x, 1) ** 2 + xp.tril(x) ** 3)
        ),
        [(4, 4), (3,)],
        None,
    ),
    "matmul": (
        lambda xp, x, y, v: (
            xp.sum(x @ y)
            + xp.sum(v[:4] @ x[0] @ y)
            + xp.sum(x @ v[:4])
            + v[:4] @ v[1:]
            + xp.sum((x @ y) @ (x[0] @ y).T)
            + xp.sum(A @ x[1] @ x[0].T)
        ),
        [(2, 4, 4), (4, 5), (5,)],
        None,
    ),
    "products": (
        lambda xp, x, y: (
            xp.tensordot(x, y, axes=2)
            + xp.sum(xp.tensordot(x, y, axes=([0], [1])) ** 2)
            + xp.sum(xp.tensordot(x, y, axes=0))
            + xp.sum(xp.dot(x, y) + xp.dot(x[0], y) + xp.dot(2.0, x))
            + xp.sum(xp.inner(x, y) ** 2)
            + xp.sum(xp.outer(x[0], y[1]) ** 2)
        ),
        [(3, 3), (3, 3)],
        None,
    ),
    "norms": (
        lambda xp, x: (
            xp.linalg.norm(x) ** 2
            + xp.linalg.norm(x, "fro")
            + xp.sum(xp.linalg.norm(x, axis=0))
            + xp.linalg.norm(x[0], 2)
        ),
        [(3, 4)],
        None,
    ),
    "linear_algebra": (
        lambda xp, s, b: (
            xp.sum(xp.linalg.solve(s, b))
            + xp.sum(xp.linalg.solve(s, b[:, 0]) ** 2)
            + xp.sum(xp.linalg.inv(s) ** 2)
            + xp.linalg.det(s)
            + xp.linalg.slogdet(s)[1]
        ),
        [(4, 4), (4, 2)],
        [_spd, None],
    ),
}

C = A + 1j * np.random.default_rng(1).normal(size=(4, 4))
C = C + C.conj().T

COMPLEX_CASES = {
    "squares": (lambda xp, x: xp.real(xp.sum(x**2)), [(5,)], None),
    "rayleigh": (
        lambda xp, x: -xp.real(xp.trace(xp.conj(x.T) @ C @ x)),
        [(4, 2)],
        None,
    ),
    "abs_norm": (
        lambda xp, x: (
            xp.sum(xp.abs(x) ** 2) + xp.linalg.norm(x) ** 3 + xp.sum(xp.abs(x))
        ),
        [(4,)],
        None,
    ),
    "holomorphic": (
        lambda xp, x: (
            xp.real(xp.sum(xp.exp(x) * xp.conj(x)))
            + xp.imag(xp.sum(x**3))
            + xp.real(xp.sum(x / (1 + x * xp.conj(x))))
        ),
        [(4,)],
        None,
    ),
    "vdot_solve": (
        lambda xp, x: (
            xp.real(xp.sum(xp.conj(x) * (C @ x)))
            + xp.real(xp.sum(xp.linalg.solve(C + 6 * np.eye(4), x)))
        ),
        [(4,)],
        None,
    ),
}

# Cases autograd cannot differentiate, checked with finite differences.
FINITE_DIFFERENCE_CASES = {
    "vdot_norms_concatenate_diagonals": (
        lambda xp, x, y: (
            xp.vdot(x, y) * xp.sum(x)
            + xp.linalg.norm(x[1], 1)
            + xp.linalg.norm(x[2], np.inf)
            + xp.sum(xp.linalg.norm(x, axis=1, keepdims=True) ** 3)
            + xp.sum(xp.concatenate([x, y], axis=None) ** 3)
            + xp.trace(x, 1)
            + xp.sum(xp.diagonal(x, -1) ** 3)
        ),
        [(3, 3), (3, 3)],
        None,
    ),
    "orders_and_repeated_indices": (
        lambda xp, x, y: (
            xp.sum(xp.ravel(x, "F") * xp.ravel(y))
            + xp.sum(xp.reshape(x, (9,), order="F") ** 3)
            + xp.sum(x[(0, 0, 2),] ** 2)
        ),
        [(3, 3), (3, 3)],
        None,
    ),
    "batched_linear_algebra": (
        lambda xp, s, b: (
            xp.sum(xp.linalg.solve(s, b) * b)
            + xp.sum(xp.linalg.solve(s, b[0, :, :3]) ** 2)
            + xp.sum(xp.linalg.solve(s, b[0, :, 0]) ** 2)
            + xp.sum(xp.linalg.slogdet(s)[1])
            + xp.real(xp.sum(xp.linalg.slogdet(s)[0]))
        ),
        [(2, 3, 3), (1, 3, 5)],
        [lambda s: s @ np.swapaxes(s, -1, -2) + 3 * np.eye(3), None],
    ),
}


def _arguments(rng, shapes, transformations, is_complex):
    transformations = transformations or [None] * len(shapes)
    args = []
    for shape, transform in zip(shapes, transformations):
        x = rng.normal(size=shape)
        if is_complex:
            x = x + 1j * rng.normal(size=shape)
        args.append(x if transform is None else transform(x))
    return args


def _check(cost, shapes, transformations, dtype):
    is_complex = np.issubdtype(dtype, np.complexfloating)
    rng = np.random.default_rng(42)
    n = len(shapes)
    ours = NumpyBackend(dtype, autodiff=True)
    reference = AutogradBackend(dtype)

    def numpy_cost(*args):
        return cost(np, *args)

    def autograd_cost(*args):
        return cost(anp, *args)

    gradient = ours.generate_gradient_operator(numpy_cost, n)
    hvp = ours.generate_hessian_operator(numpy_cost, n)
    reference_gradient = reference.generate_gradient_operator(autograd_cost, n)
    reference_hvp = reference.generate_hessian_operator(autograd_cost, n)
    # Evaluate twice at different points to exercise the compiled programs.
    for _ in range(2):
        args = _arguments(rng, shapes, transformations, is_complex)
        vectors = _arguments(rng, shapes, None, is_complex)
        if n == 1:
            pairs = [
                (gradient(*args), reference_gradient(*args)),
                (hvp(*args, *vectors), reference_hvp(*args, *vectors)),
            ]
        else:
            pairs = list(zip(gradient(*args), reference_gradient(*args)))
            pairs += zip(hvp(*args, *vectors), reference_hvp(*args, *vectors))
        for actual, expected in pairs:
            np_testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("name", REAL_CASES)
def test_real(name):
    _check(*REAL_CASES[name], np.float64)


@pytest.mark.parametrize("name", COMPLEX_CASES)
def test_complex(name):
    _check(*COMPLEX_CASES[name], np.complex128)


def _numerical_gradient(cost, args, step=1e-6):
    gradients = []
    for i, x in enumerate(args):
        gradient = np.zeros_like(x)
        units = (1, 1j) if np.iscomplexobj(x) else (1,)
        for index in np.ndindex(x.shape):
            for unit in units:
                e = np.zeros_like(x)
                e[index] = unit * step
                plus = cost(*args[:i], x + e, *args[i + 1 :])
                minus = cost(*args[:i], x - e, *args[i + 1 :])
                gradient[index] += unit * (plus - minus) / (2 * step)
        gradients.append(gradient)
    return gradients


@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("name", FINITE_DIFFERENCE_CASES)
def test_finite_differences(name, is_complex):
    cost, shapes, transformations = FINITE_DIFFERENCE_CASES[name]
    if is_complex:
        cost = (lambda f: lambda xp, *args: xp.real(f(xp, *args)))(cost)
    rng = np.random.default_rng(3)
    args = _arguments(rng, shapes, transformations, is_complex)
    vectors = _arguments(rng, shapes, None, is_complex)
    gradient = autodiff.gradient(lambda *args: cost(np, *args))
    hvp = autodiff.hessian_vector_product(lambda *args: cost(np, *args))
    numerical = _numerical_gradient(lambda *a: cost(np, *a), args)
    for actual, expected in zip(gradient(*args), numerical):
        np_testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    t = 1e-6
    plus = gradient(*(x + t * v for x, v in zip(args, vectors)))
    minus = gradient(*(x - t * v for x, v in zip(args, vectors)))
    for actual, p, m in zip(hvp(*args, *vectors), plus, minus):
        np_testing.assert_allclose(
            actual, (p - m) / (2 * t), rtol=1e-5, atol=1e-5
        )


def test_single_precision_gradient_keeps_dtype():
    gradient = autodiff.gradient(lambda x: np.sum(A[:3, :3] @ x**2))
    (g,) = gradient(np.ones(3, dtype=np.float32))
    assert g.dtype == np.float32


def test_real_argument_with_complex_constant():
    gradient = autodiff.gradient(lambda x: np.real(np.sum((1 + 2j) * x**2)))
    x = np.arange(3.0)
    (g,) = gradient(x)
    assert g.dtype == np.float64
    np_testing.assert_allclose(g, 2 * x)


def test_control_flow():
    def cost(x):
        if np.sum(x) > 0:
            return np.sum(x**2)
        return np.sum(x**3)

    gradient = autodiff.gradient(cost)
    x = np.array([1.0, 2.0])
    np_testing.assert_allclose(gradient(x)[0], 2 * x)
    np_testing.assert_allclose(gradient(-x)[0], 3 * x**2)
    np_testing.assert_allclose(gradient(x)[0], 2 * x)


def test_traced_index_follows_values():
    gradient = autodiff.gradient(lambda x: x[np.argmax(x)] ** 2)
    np_testing.assert_allclose(gradient(np.array([1.0, 3.0]))[0], [0, 6])
    np_testing.assert_allclose(gradient(np.array([4.0, 3.0]))[0], [8, 0])
    assert gradient._compiled is not None


def test_boolean_mask():
    gradient = autodiff.gradient(lambda x: np.sum(x[x > 0] ** 2))
    np_testing.assert_allclose(
        gradient(np.array([1.0, -1.0, 2.0]))[0], [2, 0, 4]
    )
    np_testing.assert_allclose(
        gradient(np.array([1.0, 3.0, 2.0]))[0], [2, 6, 4]
    )


def test_program_is_reused_for_same_trace():
    gradient = autodiff.gradient(lambda x: np.sum(x**2))
    np_testing.assert_allclose(gradient(np.ones(2))[0], [2, 2])
    assert gradient._compiled is None
    np_testing.assert_allclose(gradient(np.arange(2.0))[0], [0, 2])
    compiled = gradient._compiled
    np_testing.assert_allclose(gradient(np.ones(2))[0], [2, 2])
    assert gradient._compiled is compiled
    np_testing.assert_allclose(gradient(np.arange(3.0))[0], [0, 2, 4])
    assert gradient._compiled is None


def test_values_read_outside_arguments():
    A = np.eye(3)
    state = {"scale": 1.0}

    def cost(x):
        return np.sum((2 * A) @ x) * state["scale"]

    gradient = autodiff.gradient(cost)
    hvp = autodiff.hessian_vector_product(lambda x: cost(x) * np.sum(x))
    x = np.ones(3)
    np_testing.assert_allclose(gradient(x)[0], [2, 2, 2])
    np_testing.assert_allclose(hvp(x, x)[0], [12, 12, 12])
    A[:] = 5 * np.eye(3)
    np_testing.assert_allclose(gradient(x)[0], [10, 10, 10])
    np_testing.assert_allclose(hvp(x, x)[0], [60, 60, 60])
    # Arrays are inputs of the compiled programs, so they are reused.
    assert gradient._compiled is not None and hvp._compiled is not None
    state["scale"] = 10.0
    np_testing.assert_allclose(gradient(x)[0], [100, 100, 100])
    np_testing.assert_allclose(hvp(x, x)[0], [600, 600, 600])
    A = np.zeros((3, 3))
    np_testing.assert_allclose(gradient(x)[0], [0, 0, 0])


def test_constant_used_only_in_backward_pass():
    # The backward pass of a product with a complex constant uses its
    # conjugate.
    C = np.array([1 + 1j, 2j])
    gradient = autodiff.gradient(lambda x: np.real(np.sum(x * C)))
    x = np.ones(2, complex)
    for _ in range(2):
        np_testing.assert_allclose(gradient(x)[0], [1 - 1j, -2j])
    compiled = gradient._compiled
    C *= 1j
    np_testing.assert_allclose(gradient(x)[0], [-1 - 1j, -2])
    assert gradient._compiled is compiled


def test_constant_and_unused_arguments_get_zero_gradients():
    g_x, g_y = autodiff.gradient(lambda x, y: np.sum(x))(
        np.ones(2), np.ones(3)
    )
    np_testing.assert_allclose(g_x, np.ones(2))
    np_testing.assert_allclose(g_y, np.zeros(3))
    assert g_x.flags.writeable
    (g,) = autodiff.gradient(lambda x: 3.0)(np.ones(2))
    np_testing.assert_allclose(g, np.zeros(2))


def test_unsupported_function():
    gradient = autodiff.gradient(lambda x: np.sum(np.sort(x)))
    with pytest.raises(NotImplementedError, match="sort"):
        gradient(np.ones(3))


@pytest.mark.parametrize(
    "cost", [lambda x: x**2, lambda x: np.ones(2), lambda x: 1 + 2j]
)
def test_non_scalar_cost(cost):
    with pytest.raises(ValueError, match="real scalar"):
        autodiff.gradient(cost)(np.ones(3))


def test_in_place_update():
    def cost(x):
        y = np.zeros(3)
        y += x
        return np.sum(y)

    with pytest.raises(NotImplementedError, match="in-place"):
        autodiff.gradient(cost)(np.ones(3))


def test_conversion_to_python_number():
    gradient = autodiff.gradient(lambda x: float(np.sum(x)) * np.sum(x))
    with pytest.raises(TypeError, match="losing its derivative"):
        gradient(np.ones(3))


def test_storing_traced_value_in_array():
    def cost(x):
        y = np.zeros(2)
        y[0] = x[1]
        return np.sum(y)

    with pytest.raises(ValueError):
        autodiff.gradient(cost)(np.ones(2))


def test_reductions_with_where_dtype_and_scalars():
    mask = np.array([True, False, True])

    def cost(x):
        return (
            np.sum(x**2, where=mask)
            + np.max(x, where=mask, initial=-10.0) ** 2
            + np.mean(x[1] ** 2, axis=0, dtype=np.float64)
        )

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    (g,) = autodiff.gradient(cost)(x)
    np_testing.assert_allclose(g, [2, 4, 12])
    assert g.dtype == np.float32
    (h,) = autodiff.hessian_vector_product(cost)(x, np.ones(3, np.float32))
    np_testing.assert_allclose(h, [2, 2, 4])


def test_traced_slice_bounds():
    gradient = autodiff.gradient(lambda x: np.sum(x[: np.argmax(x)] ** 2))
    x = np.array([1.0, 4.0, 2.0])
    for _ in range(3):
        np_testing.assert_allclose(gradient(x)[0], [2, 0, 0])
    assert gradient._compiled is not None


def test_hessian_vector_product_with_vectors_of_other_dtype():
    hvp = autodiff.hessian_vector_product(lambda x: np.real(np.sum(x**2)))
    x = np.array([1 + 2j])
    np_testing.assert_allclose(hvp(x, np.ones(1))[0], [2])
    np_testing.assert_allclose(hvp(x, np.array([1 + 2j]))[0], [2 - 4j])


def test_complex_constant_with_signed_zero():
    c = np.array(-1 + 0j)
    gradient = autodiff.gradient(lambda x: np.imag(np.sum(c**x)))
    x = np.ones(1)
    np_testing.assert_allclose(gradient(x)[0], [-np.pi])
    c[...] = complex(-1, -0.0)
    np_testing.assert_allclose(gradient(x)[0], [np.pi])


def test_complex_absolute_value_at_zero():
    gradient = autodiff.gradient(lambda x: np.sum(np.abs(x) ** 2))
    np_testing.assert_allclose(gradient(np.zeros(2, complex))[0], [0, 0])


def test_traced_keyword_operand():
    gradient = autodiff.gradient(lambda x: np.sum(a=x**2))
    with pytest.raises(NotImplementedError, match="keyword"):
        gradient(np.ones(2))


def test_array_methods_and_helpers():
    def cost(x, y):
        return (
            np.sum(x.flatten() * y.ravel())
            + np.sum(x.reshape(9) ** 3)
            + x[:1].squeeze().sum()
            + np.sum(x.clip(-0.5, 0.5) ** 2)
            + np.sum(x.clip(0) ** 2 + np.clip(x, max=0.2) ** 3)
            + x.diagonal().sum() * x.trace()
            + x.mean() * x.min() * x.max(axis=0).sum()
            + np.sum(x.dot(y) * x.T)
            + np.sum(x.swapaxes(0, 1) * x.transpose() * x.conj())
            + np.sum(np.moveaxis(x[None], 0, 2) ** 3)
            + np.sum(x.astype(np.float64) * np.copy(y) * np.ones_like(x))
            + np.sum(np.zeros_like(x)) * x.size * np.ndim(x)
            + (np.sum(x) if np.isrealobj(x) and not np.iscomplexobj(y) else 0)
        )

    rng = np.random.default_rng(4)
    args = [rng.normal(size=(3, 3)), rng.normal(size=(3, 3))]
    for actual, expected in zip(
        autodiff.gradient(cost)(*args), _numerical_gradient(cost, args)
    ):
        np_testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
