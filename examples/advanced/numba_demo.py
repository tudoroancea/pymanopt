"""Compare the NumPy and numba backends on packing points on a sphere.

Both backends differentiate the same plain NumPy cost function. The numba
backend compiles the gradient and Hessian-vector product with numba, which
takes a moment on the first call and then makes every call faster.

Run with ``python -m examples.advanced.numba_demo`` from the repository root.
"""

import time
import timeit

import numpy as np

import pymanopt
from pymanopt.manifolds import Elliptope
from pymanopt.optimizers import ConjugateGradient


def create_cost(manifold, epsilon, decorator):
    @decorator(manifold)
    def cost(X):
        Y = X @ X.T
        s = np.triu(Y, 1).max()
        expY = np.exp((Y - s) / epsilon)
        expY -= np.diag(np.diag(expY))
        u = np.triu(expY, 1).sum()
        return s + epsilon * np.log(u)

    return cost


def _microseconds(function, number=200):
    return min(timeit.repeat(function, number=number, repeat=5)) / number * 1e6


def benchmark(backend, num_points, dimension, epsilon):
    manifold = Elliptope(num_points, dimension)
    decorator = getattr(pymanopt.function, backend)
    problem = pymanopt.Problem(
        manifold, create_cost(manifold, epsilon, decorator)
    )
    np.random.seed(0)
    point = manifold.random_point()
    vector = manifold.random_tangent_vector(point)

    start = time.perf_counter()
    problem.cost(point)
    problem.euclidean_gradient(point)
    problem.euclidean_hessian(point, vector)
    first_calls = time.perf_counter() - start

    cost = _microseconds(lambda: problem.cost(point))
    gradient = _microseconds(lambda: problem.euclidean_gradient(point))
    hvp = _microseconds(lambda: problem.euclidean_hessian(point, vector))

    optimizer = ConjugateGradient(max_iterations=1000, verbosity=0)
    result = optimizer.run(problem, initial_point=point)
    print(
        f"{backend:8s} {first_calls:16.2f} {cost:10.1f} {gradient:14.1f} "
        f"{hvp:9.1f} {result.time:17.3f}"
    )


def run(num_points=24, dimension=3, epsilon=0.005):
    print(f"Packing {num_points} points on the sphere in R^{dimension}")
    print(
        f"{'backend':8s} {'first calls (s)':>16s} {'cost (us)':>10s} "
        f"{'gradient (us)':>14s} {'hvp (us)':>9s} {'optimization (s)':>17s}"
    )
    for backend in ("numpy", "numba"):
        benchmark(backend, num_points, dimension, epsilon)


if __name__ == "__main__":
    run()
