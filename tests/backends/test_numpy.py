import numpy as np
import pytest
from numpy import testing as np_testing

import pymanopt
from pymanopt.backends.numpy_backend import NumpyBackend

from . import _backend_tests


class TestNumpyBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=3, backend=NumpyBackend()
        )
        self.n = 10

        @pymanopt.function.numpy(self.manifold)
        def nary_cost(x, y):
            return np.sum(x * y)

        self.cost = self.nary_cost = nary_cost

        @pymanopt.function.numpy(self.manifold)
        def nested_nary_cost(x, y, z):
            return np.sum(x**2 * y + 3 * z)

        self.nested_nary_cost = nested_nary_cost

    def test_nary_cost_function(self):
        n = self.n
        x, y = [np.random.normal(size=n) for _ in range(2)]
        np_testing.assert_allclose(np.sum(x * y), self.nary_cost(x, y))

    def test_nested_nary_cost_function(self):
        n = self.n
        x, y, z = [np.random.normal(size=n) for _ in range(3)]
        np_testing.assert_allclose(
            np.sum(x**2 * y + 3 * z), self.nested_nary_cost(x, y, z)
        )


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(x):
            return np.sum(x**2)

        self.cost = cost


class TestUnaryComplexFunction(_backend_tests.TestUnaryComplexFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout,
            backend=NumpyBackend(np.complex128),
            is_complex=True,
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(x):
            return np.real(np.sum(x**2))

        self.cost = cost


class TestUnaryVarargFunction(_backend_tests.TestUnaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(*x):
            (x,) = x
            return np.sum(x**2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(x, y):
            return x @ y

        self.cost = cost


class TestNaryVarargFunction(_backend_tests.TestNaryFunction):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(*args):
            return np.dot(*args)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(x, y, z):
            return np.sum(x**2 + y + z**3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestMatrix(_backend_tests.TestMatrix):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestTensor3(_backend_tests.TestTensor3):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(X):
            return np.exp(np.sum(X**2))

        self.cost = cost


class TestMixed(_backend_tests.TestMixed):
    @pytest.fixture(autouse=True)
    def setup(self, pre_setup):
        self.manifold = _backend_tests.manifold_factory(
            point_layout=self.point_layout, backend=NumpyBackend()
        )

        @pymanopt.function.numpy(self.manifold)
        def cost(x, y, z):
            return (
                np.exp(np.sum(x**2))
                + np.exp(np.sum(y**2))
                + np.exp(np.sum(z**2))
            )

        self.cost = cost
