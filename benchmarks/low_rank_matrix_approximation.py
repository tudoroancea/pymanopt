import os
from typing import Any


def init(backend: str):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import FixedRankEmbedded

    modules = {"np": np, "pymanopt": pymanopt}

    m, n, rank = 5, 4, 2
    matrix = np.random.normal(size=(m, n))
    manifold = FixedRankEmbedded(m, n, rank)
    vars = {"matrix": matrix, "rank": rank, "manifold": manifold}
    euclidean_gradient = None
    if backend == "autograd":
        import autograd.numpy as anp

        @pymanopt.function.autograd(manifold)
        def cost(u, s, vt):
            X = u @ anp.diag(s) @ vt
            return anp.linalg.norm(X - matrix) ** 2

    elif backend == "jax":
        import jax.numpy as jnp

        modules["jnp"] = jnp

        @pymanopt.function.jax(manifold)
        def cost(u, s, vt):
            X = u @ jnp.diag(s) @ vt
            return jnp.linalg.norm(X - matrix) ** 2

    elif backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(u, s, vt):
            X = u @ np.diag(s) @ vt
            return np.linalg.norm(X - matrix) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(u, s, vt):
            X = u @ np.diag(s) @ vt
            S = np.diag(s)
            gu = 2 * (X - matrix) @ (S @ vt).T
            gs = 2 * np.diag(u.T @ (X - matrix) @ vt.T)
            gvt = 2 * (u @ S).T @ (X - matrix)
            return gu, gs, gvt

    elif backend == "pytorch":
        import torch

        modules["torch"] = torch

        matrix = torch.from_numpy(matrix)

        @pymanopt.function.pytorch(manifold)
        def cost(u, s, vt):
            X = u @ torch.diag(s) @ vt
            return torch.norm(X - matrix) ** 2

    elif backend == "tensorflow":
        import tensorflow as tf

        modules["tf"] = tf

        matrix = tf.constant(matrix)

        @pymanopt.function.tensorflow(manifold)
        def cost(u, s, vt):
            X = u @ tf.linalg.diag(s) @ vt
            return tf.norm(X - matrix) ** 2

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
    optimizer = modules["pymanopt"].optimizers.ConjugateGradient(
        verbosity=0,
        beta_rule="PolakRibiere",
        min_gradient_norm=1e-8,
        max_iterations=10000,
    )
    res = optimizer.run(vars["problem"]).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    left_singular_vectors, singular_values, right_singular_vectors = res
    if not isinstance(left_singular_vectors, np.ndarray):
        if backend == "pytorch":
            left_singular_vectors = left_singular_vectors.detach().numpy()
            singular_values = singular_values.detach().numpy()
            right_singular_vectors = right_singular_vectors.detach().numpy()
        elif backend == "tensorflow":
            left_singular_vectors = left_singular_vectors.numpy()
            singular_values = singular_values.numpy()
            right_singular_vectors = right_singular_vectors.numpy()

    low_rank_approximation = (
        left_singular_vectors
        @ np.diag(singular_values)
        @ right_singular_vectors
    )

    # Compute the solution with SVD
    u, s, vt = np.linalg.svd(vars["matrix"], full_matrices=False)
    indices = np.argsort(s)[-vars["rank"] :]
    low_rank_solution = u[:, indices] @ np.diag(s[indices]) @ vt[indices, :]

    assert np.allclose(
        low_rank_approximation, low_rank_solution, atol=1e-6
    ), f"error: {np.linalg.norm(low_rank_approximation - low_rank_solution)}"


if __name__ == "__main__":
    from template import run_benchmark

    run_benchmark(
        f"benchmark_{os.path.basename(__file__)}",
        init,
        autodiff,
        optim,
        check_res,
    )
