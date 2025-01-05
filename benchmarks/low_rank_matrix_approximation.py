from typing import Any


def init(backend: str, dev=True):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import FixedRankEmbedded
    from pymanopt.manifolds.fixed_rank import _FixedRankPoint

    modules = {"np": np, "pymanopt": pymanopt}

    # setup problem
    m, n, rank = 20, 10, 5
    matrix = np.random.normal(size=(m, n))
    manifold = FixedRankEmbedded(m, n, rank)
    if dev:
        from pymanopt.backends.numpy_backend import NumpyBackend

        manifold.set_compatible_backend(NumpyBackend())
    initial_point = manifold.random_point()

    # create cost function and gradient
    euclidean_gradient = None
    if backend == "numpy":

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

    elif backend == "autograd":
        import autograd.numpy as anp

        if dev:
            from pymanopt.backends.autograd_backend import AutogradBackend

            manifold.set_compatible_backend(AutogradBackend())
            initial_point = _FixedRankPoint(
                anp.array(initial_point.u),
                anp.array(initial_point.s),
                anp.array(initial_point.vt),
            )

        @pymanopt.function.autograd(manifold)
        def cost(u, s, vt):
            X = u @ anp.diag(s) @ vt
            return anp.linalg.norm(X - matrix) ** 2

    elif backend == "jax":
        import jax.numpy as jnp

        matrix = jnp.array(matrix)

        if dev:
            from pymanopt.backends.jax_backend import JaxBackend

            manifold.set_compatible_backend(JaxBackend())
            initial_point = _FixedRankPoint(
                jnp.array(initial_point.u),
                jnp.array(initial_point.s),
                jnp.array(initial_point.vt),
            )

        @pymanopt.function.jax(manifold)
        def cost(u, s, vt):
            X = u @ jnp.diag(s) @ vt
            return jnp.linalg.norm(X - matrix) ** 2

    elif backend == "pytorch":
        import torch

        matrix = torch.from_numpy(matrix)

        if dev:
            from pymanopt.backends.pytorch_backend import PytorchBackend

            manifold.set_compatible_backend(PytorchBackend())
            initial_point = _FixedRankPoint(
                torch.from_numpy(initial_point.u),
                torch.from_numpy(initial_point.s),
                torch.from_numpy(initial_point.vt),
            )

        @pymanopt.function.pytorch(manifold)
        def cost(u, s, vt):
            X = u @ torch.diag(s) @ vt
            return torch.norm(X - matrix) ** 2

    elif backend == "tensorflow":
        import tensorflow as tf

        matrix = tf.constant(matrix)

        if dev:
            from pymanopt.backends.tensorflow_backend import TensorflowBackend

            manifold.set_compatible_backend(TensorflowBackend())
            initial_point = _FixedRankPoint(
                tf.constant(initial_point.u),
                tf.constant(initial_point.s),
                tf.constant(initial_point.vt),
            )

        @pymanopt.function.tensorflow(manifold)
        def cost(u, s, vt):
            X = u @ tf.linalg.diag(s) @ vt
            return tf.norm(X - matrix) ** 2

    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )
    vars = {
        "matrix": matrix,
        "rank": rank,
        "manifold": manifold,
        "problem": problem,
        "initial_point": initial_point,
    }
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
    res = optimizer.run(
        vars["problem"], initial_point=vars["initial_point"]
    ).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    left_singular_vectors = np.array(res.u)
    singular_values = np.array(res.s)
    right_singular_vectors = np.array(res.vt)

    low_rank_approximation = (
        left_singular_vectors
        @ np.diag(singular_values)
        @ right_singular_vectors
    )

    # Compute the solution with SVD
    u, s, vt = np.linalg.svd(np.array(vars["matrix"]), full_matrices=False)
    indices = np.argsort(s)[-vars["rank"] :]
    low_rank_solution = u[:, indices] @ np.diag(s[indices]) @ vt[indices, :]

    if not np.allclose(low_rank_approximation, low_rank_solution, atol=1e-6):
        print(
            f"error: {np.linalg.norm(low_rank_approximation - low_rank_solution)}"
        )
