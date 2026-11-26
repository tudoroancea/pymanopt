import numpy as np
import pytest
from numpy import testing as np_testing

from examples import (
    closest_unit_norm_column_approximation,
    dominant_eigenvector,
    dominant_invariant_complex_subspace,
    dominant_invariant_subspace,
    low_rank_matrix_approximation,
    low_rank_psd_matrix_approximation,
    multiple_linear_regression,
    optimal_rotations,
    packing_on_the_sphere,
    pca,
    rank_k_correlation_matrix_approximation,
)
from examples.advanced import check_gradient, check_hessian, check_retraction
from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import (
    ComplexGrassmann,
    Euclidean,
    FixedRankEmbedded,
    Grassmann,
    Oblique,
    PSDFixedRank,
    SpecialOrthogonalGroup,
    Sphere,
    Stiefel,
)

SUPPORTED_BACKENDS = {"numpy", "autograd", "jax", "pytorch", "tensorflow"}


class TestExamples:
    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                closest_unit_norm_column_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_closest_unit_norm_column_approximation(self, backend):
        closest_unit_norm_column_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_eigenvector.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_eigenvector(self, backend):
        dominant_eigenvector.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_invariant_subspace.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_invariant_subspace(self, backend):
        dominant_invariant_subspace.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(dominant_invariant_subspace.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_dominant_invariant_complex_subspace(self, backend):
        dominant_invariant_complex_subspace.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(low_rank_matrix_approximation.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_low_rank_matrix_approximation(self, backend):
        low_rank_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                low_rank_psd_matrix_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_low_rank_psd_matrix_approximation(self, backend):
        low_rank_psd_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(multiple_linear_regression.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_multiple_linear_regression(self, backend):
        multiple_linear_regression.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(optimal_rotations.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_optimal_rotations(self, backend):
        optimal_rotations.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(packing_on_the_sphere.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_packing_on_the_sphere(self, backend):
        packing_on_the_sphere.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(set(pca.SUPPORTED_BACKENDS).intersection(SUPPORTED_BACKENDS)),
    )
    def test_pca(self, backend):
        pca.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(
                rank_k_correlation_matrix_approximation.SUPPORTED_BACKENDS
            ).intersection(SUPPORTED_BACKENDS)
        ),
    )
    def test_rank_k_correlation_matrix_approximation(self, backend):
        rank_k_correlation_matrix_approximation.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_gradient.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_gradient(self, backend):
        check_gradient.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_gradient.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_hessian(self, backend):
        check_hessian.run(backend)

    @pytest.mark.parametrize(
        "backend",
        list(
            set(check_retraction.SUPPORTED_BACKENDS).intersection(
                SUPPORTED_BACKENDS
            )
        ),
    )
    def test_check_retraction(self, backend):
        check_retraction.run(backend)


rng = np.random.default_rng(0)
real, complex_ = NumpyBackend(), NumpyBackend(np.complex128)
symmetric = rng.normal(size=(6, 6))
symmetric += symmetric.T
hermitian = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
hermitian += hermitian.conj().T


WRONG_DERIVATIVES = {
    (pca, 2): "The handwritten Hessian of the example is wrong",
    (rank_k_correlation_matrix_approximation, 1): (
        "The handwritten gradient of the example is halved"
    ),
}


@pytest.mark.parametrize("order", [1, 2], ids=["gradient", "hessian"])
@pytest.mark.parametrize(
    "example, manifold, data",
    [
        (
            closest_unit_norm_column_approximation,
            Oblique(5, 8, backend=real),
            [rng.normal(size=(5, 8))],
        ),
        (dominant_eigenvector, Sphere(6, backend=real), [symmetric]),
        (
            dominant_invariant_complex_subspace,
            ComplexGrassmann(6, 2, backend=complex_),
            [hermitian],
        ),
        (
            dominant_invariant_subspace,
            Grassmann(6, 2, backend=real),
            [symmetric],
        ),
        (
            low_rank_matrix_approximation,
            FixedRankEmbedded(5, 4, 2, backend=real),
            [rng.normal(size=(5, 4))],
        ),
        (
            low_rank_psd_matrix_approximation,
            PSDFixedRank(6, 2, backend=real),
            [symmetric],
        ),
        (
            multiple_linear_regression,
            Euclidean(3, backend=real),
            [rng.normal(size=(20, 3)), rng.normal(size=20)],
        ),
        (
            optimal_rotations,
            SpecialOrthogonalGroup(3, k=4, backend=real),
            [rng.normal(size=(4, 3, 3))],
        ),
        (pca, Stiefel(3, 2, backend=real), [rng.normal(size=(20, 3))]),
        (
            rank_k_correlation_matrix_approximation,
            Oblique(3, 6, backend=real),
            [symmetric],
        ),
    ],
    ids=lambda value: getattr(value, "__name__", None),
)
def test_numpy_autodiff_matches_handwritten_derivatives(
    example, manifold, data, order, request
):
    cost, *derivatives = example.create_cost_and_derivates(
        manifold, *data, "numpy"
    )
    if order > len(derivatives):
        pytest.skip("The example has no handwritten Hessian")
    if (example, order) in WRONG_DERIVATIVES:
        request.applymarker(
            pytest.mark.xfail(
                reason=WRONG_DERIVATIVES[example, order], strict=True
            )
        )
    n = manifold.num_values
    point = manifold.random_point()
    args = list(point) if n > 1 else [point]
    backend = NumpyBackend(autodiff=True)
    if order == 1:
        actual = backend.generate_gradient_operator(cost, n)(*args)
    else:
        args.append(manifold.random_tangent_vector(point))
        actual = backend.generate_hessian_operator(cost, n)(*args)
    expected = derivatives[order - 1](*args)
    for a, e in zip(actual, expected) if n > 1 else [(actual, expected)]:
        np_testing.assert_allclose(a, e)
