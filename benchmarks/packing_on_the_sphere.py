from typing import Any


def init(backend: str, dev=True):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Elliptope

    modules = {"np": np, "pymanopt": pymanopt}

    # setup the problem
    dimension = 10  # Dimension of the embedding space, i.e. R^k
    num_points = 50  # Points on the sphere
    epsilon = 0.005

    # generate initial point for optimizer
    initial_point = np.random.normal(size=(num_points, dimension))
    initial_point = (
        initial_point / np.linalg.norm(initial_point, axis=1)[:, None]
    )

    # create manifold
    manifold = Elliptope(num_points, dimension)

    # create cost function
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

        if dev:
            initial_point = jnp.array(initial_point)

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

        if dev:
            initial_point = torch.from_numpy(initial_point)

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

        if dev:
            initial_point = tf.convert_to_tensor(initial_point)

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            Y = X @ tf.transpose(X)
            s = tf.reduce_max(tf.linalg.band_part(Y, 0, -1))
            expY = tf.exp((Y - s) / epsilon)
            expY = expY - tf.linalg.diag(tf.linalg.diag_part(expY))
            u = tf.reduce_sum(tf.linalg.band_part(Y, 0, -1))
            return s + epsilon * tf.math.log(u)

    problem = pymanopt.Problem(manifold, cost)
    vars = {
        "manifold": manifold,
        "problem": problem,
        "initial_point": initial_point,
    }
    return modules, vars


def autodiff(backend: str, modules: dict, vars: dict):
    point = vars["manifold"].random_point()
    vars["problem"].euclidean_gradient(point)


def optim(modules: dict, vars: dict):
    optimizer = modules["pymanopt"].optimizers.ConjugateGradient(
        verbosity=0, min_gradient_norm=1e-8, max_iterations=1e5
    )
    res = optimizer.run(
        vars["problem"], initial_point=vars["initial_point"]
    ).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    return
    np = modules["np"]
    Y = res

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    X = Y @ Y.T
    maxdot = np.triu(X, 1).max()
    print("Maximum angle between any two points:", maxdot)
