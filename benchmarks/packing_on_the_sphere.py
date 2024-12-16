import os
from typing import Any

from template import run_benchmark


def init(backend: str):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Elliptope

    modules = {"np": np, "pymanopt": pymanopt}

    dimension = 3  # Dimension of the embedding space, i.e. R^k
    num_points = 24  # Points on the sphere
    epsilon = 0.005

    manifold = Elliptope(num_points, dimension)
    if backend == "numpy":
        raise NotImplementedError(
            "numpy backend not implemented for this benchmark"
        )
    elif backend == "autograd":
        import autograd.numpy as anp

        @pymanopt.function.autograd(manifold)
        def cost(X):
            Y = X @ X.T
            # Shift the exponentials by the maximum value to reduce numerical
            # trouble due to possible overflows.
            s = anp.triu(Y, 1).max()
            expY = anp.exp((Y - s) / epsilon)
            # Zero out the diagonal
            expY -= anp.diag(anp.diag(expY))
            u = anp.triu(expY, 1).sum()
            return s + epsilon * anp.log(u)

    elif backend == "jax":
        import jax.numpy as jnp

        modules["jnp"] = jnp

        @pymanopt.function.jax(manifold)
        def cost(X):
            Y = X @ X.T
            s = jnp.triu(Y, 1).max()
            expY = jnp.exp((Y - s) / epsilon)
            expY -= jnp.diag(jnp.diag(expY))
            u = jnp.triu(expY, 1).sum()
            return s + epsilon * jnp.log(u)

    elif backend == "pytorch":
        import torch

        modules["torch"] = torch

        @pymanopt.function.pytorch(manifold)
        def cost(X):
            Y = X @ torch.transpose(X, 1, 0)
            s = torch.triu(Y, 1).max()
            expY = torch.exp((Y - s) / epsilon)
            expY = expY - torch.diag(torch.diag(expY))
            u = torch.triu(expY, 1).sum()
            return s + epsilon * torch.log(u)

    elif backend == "tensorflow":
        import tensorflow as tf

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            Y = X @ tf.transpose(X)
            s = tf.reduce_max(tf.linalg.band_part(Y, 0, -1))
            expY = tf.exp((Y - s) / epsilon)
            expY = expY - tf.linalg.diag(tf.linalg.diag_part(expY))
            u = tf.reduce_sum(tf.linalg.band_part(Y, 0, -1))
            return s + epsilon * tf.math.log(u)

    problem = pymanopt.Problem(manifold, cost)
    vars = {"manifold": manifold, "problem": problem}
    return modules, vars


def autodiff(backend: str, modules: dict, vars: dict):
    point = vars["manifold"].random_point()
    vars["problem"].euclidean_gradient(point)


def optim(modules: dict, vars: dict):
    optimizer = modules["pymanopt"].optimizers.ConjugateGradient(
        verbosity=0, min_gradient_norm=1e-8, max_iterations=1e5
    )
    res = optimizer.run(vars["problem"]).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    Y = res

    if not isinstance(Y, np.ndarray):
        if backend == "pytorch":
            Y = Y.cpu().detach().numpy()
        elif backend == "tensorflow":
            Y = Y.numpy()

    X = Y @ Y.T
    maxdot = np.triu(X, 1).max()
    print("Maximum angle between any two points:", maxdot)


if __name__ == "__main__":
    run_benchmark(
        f"benchmark_{os.path.basename(__file__)}",
        init,
        autodiff,
        optim,
        check_res,
    )
