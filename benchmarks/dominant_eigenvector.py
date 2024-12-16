import os
from typing import Any

from template import run_benchmark


def init(backend: str):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Sphere

    modules = {"np": np, "pymanopt": pymanopt}

    n = 128
    matrix = np.random.normal(size=(n, n))
    matrix = 0.5 * (matrix + matrix.T)
    manifold = Sphere(n)
    vars = {"matrix": matrix, "manifold": manifold}
    euclidean_gradient = None
    if backend == "autograd":

        @pymanopt.function.autograd(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "jax":

        @pymanopt.function.jax(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(x):
            return -x.T @ matrix @ x

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(x):
            return -2 * matrix @ x

    elif backend == "pytorch":
        import torch

        modules["torch"] = torch

        matrix_ = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(x):
            return -x.reshape(1, -1) @ matrix_ @ x.reshape(-1, 1)

    elif backend == "tensorflow":
        import tensorflow as tf

        modules["tf"] = tf

        matrix = tf.constant(matrix)

        @pymanopt.function.tensorflow(manifold)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(matrix, x, axes=1), axes=1)

    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )
    vars["problem"] = problem
    return modules, vars


def autodiff(backend: str, modules: dict, vars: dict):
    if backend != "numpy":
        point = vars["manifold"].random_point()
        vars["problem"].euclidean_gradient(point)


def optim(modules: dict, vars: dict):
    optimizer = modules["pymanopt"].optimizers.SteepestDescent(verbosity=0)
    res = optimizer.run(vars["problem"]).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    estimated_dominant_eigenvector = res
    if backend == "pytorch":
        estimated_dominant_eigenvector = (
            estimated_dominant_eigenvector.cpu().detach().numpy()
        )
    elif backend == "tensorflow":
        estimated_dominant_eigenvector = estimated_dominant_eigenvector.numpy()

    # Calculate the actual solution by a conventional eigenvalue decomposition.
    np = modules["np"]
    eigenvalues, eigenvectors = np.linalg.eig(vars["matrix"])
    dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, but for comparison we need to get rid of the sign
    # ambiguity.
    if np.sign(dominant_eigenvector[0]) != np.sign(
        estimated_dominant_eigenvector[0]
    ):
        estimated_dominant_eigenvector = -estimated_dominant_eigenvector

    # Check norm between the two vectors is close to zero.
    assert np.allclose(
        dominant_eigenvector, estimated_dominant_eigenvector, atol=1e-6
    )


if __name__ == "__main__":
    run_benchmark(
        f"benchmark_{os.path.basename(__file__)}",
        init,
        autodiff,
        optim,
        check_res,
    )
