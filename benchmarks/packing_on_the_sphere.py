from typing import Any


def init(backend: str, dev=True):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Elliptope

    modules = {"np": np, "pymanopt": pymanopt}

    dimension = 10  # Dimension of the embedding space, i.e. R^k
    num_points = 50  # Points on the sphere
    epsilon = 0.005
    manifold = Elliptope(num_points, dimension)
    if dev:
        from pymanopt.backends.numpy_backend import NumpyBackend

        manifold.set_compatible_backend(NumpyBackend())

    initial_point = manifold.random_point()

    # create cost function
    if backend == "numpy":
        raise NotImplementedError(
            "numpy backend not implemented for this benchmark"
        )
    elif backend == "autograd":
        import autograd.numpy as anp

        if dev:
            from pymanopt.backends.autograd_backend import AutogradBackend

            manifold.set_compatible_backend(AutogradBackend())
            initial_point = manifold.backend.array(initial_point)

        @pymanopt.function.autograd(manifold)
        def cost(X):
            Y = X @ X.T
            # Shift the exponentials by the maximum value to reduce numerical
            # trouble due to possible overflows.
            s = anp.max(anp.triu(Y, 1))
            expY = anp.exp((Y - s) / epsilon)
            u = anp.sum(anp.triu(expY, 1))
            return s + epsilon * anp.log(u)

    elif backend == "jax":
        import jax.numpy as jnp

        if dev:
            from pymanopt.backends.jax_backend import JaxBackend

            manifold.set_compatible_backend(JaxBackend())
            initial_point = manifold.backend.array(initial_point)

        @pymanopt.function.jax(manifold)
        def cost(X):
            Y = X @ X.T
            s = jnp.max(jnp.triu(Y, 1))
            expY = jnp.exp((Y - s) / epsilon)
            u = jnp.sum(jnp.triu(expY, 1))
            return s + epsilon * jnp.log(u)

    elif backend == "pytorch":
        import torch

        if dev:
            from pymanopt.backends.pytorch_backend import PytorchBackend

            manifold.set_compatible_backend(PytorchBackend())
            initial_point = manifold.backend.array(initial_point)

        @pymanopt.function.pytorch(manifold)
        def cost(X: torch.Tensor):
            Y = X @ torch.transpose(X, 1, 0)
            s = torch.max(torch.triu(Y, 1))
            expY = torch.exp((Y - s) / epsilon)
            u = torch.sum(torch.triu(expY, 1))
            return s + epsilon * torch.log(u)

    elif backend == "tensorflow":
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp

        if dev:
            from pymanopt.backends.tensorflow_backend import TensorflowBackend

            manifold.set_compatible_backend(TensorflowBackend())
            initial_point = manifold.backend.array(initial_point)

        @pymanopt.function.tensorflow(manifold)
        def cost(X):
            Y = X @ tf.transpose(X)
            s = tnp.max(tnp.triu(Y, 1))
            expY = tnp.exp((Y - s) / epsilon)
            u = tnp.sum(tnp.triu(expY, 1))
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
        verbosity=1, min_gradient_norm=1e-8, max_iterations=1e5
    )
    res = optimizer.run(
        vars["problem"], initial_point=vars["initial_point"]
    ).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    # return
    np = modules["np"]
    Y = res

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    X = Y @ Y.T
    maxdot = np.triu(X, 1).max()
    print("Maximum angle between any two points:", maxdot)
