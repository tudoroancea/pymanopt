from typing import Any


def init(backend: str, dev=True):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Sphere

    modules = {"np": np, "pymanopt": pymanopt}

    n = 128
    matrix = np.random.normal(size=(n, n))
    matrix = 0.5 * (matrix + matrix.T)
    manifold = Sphere(n)
    if dev:
        from pymanopt.backends.numpy_backend import NumpyBackend

        manifold.set_compatible_backend(NumpyBackend())

    initial_point = manifold.random_point()

    euclidean_gradient = None
    if backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(x):
            return -x.T @ matrix @ x

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(x):
            return -2 * matrix @ x

    elif backend == "autograd":
        if dev:
            import autograd.numpy as anp

            from pymanopt.backends.autograd_backend import AutogradBackend

            manifold.set_compatible_backend(AutogradBackend())
            initial_point = anp.array(initial_point)

        @pymanopt.function.autograd(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "jax":
        import jax.numpy as jnp

        matrix = jnp.array(matrix)
        if dev:
            from pymanopt.backends.jax_backend import JaxBackend

            manifold.set_compatible_backend(JaxBackend())
            initial_point = jnp.array(initial_point)

        @pymanopt.function.jax(manifold)
        def cost(x):
            return -x.T @ matrix @ x

    elif backend == "pytorch":
        import torch

        matrix = torch.from_numpy(matrix)
        if dev:
            from pymanopt.backends.pytorch_backend import PytorchBackend

            manifold.set_compatible_backend(PytorchBackend())
            initial_point = torch.tensor(initial_point)

        @pymanopt.function.pytorch(manifold)
        def cost(x):
            return -x.reshape(1, -1) @ matrix @ x.reshape(-1, 1)

    elif backend == "tensorflow":
        import tensorflow as tf

        matrix = tf.constant(matrix)

        if dev:
            from pymanopt.backends.tensorflow_backend import TensorflowBackend

            manifold.set_compatible_backend(TensorflowBackend())
            initial_point = tf.constant(initial_point)

        @pymanopt.function.tensorflow(manifold)
        def cost(x):
            return -tf.tensordot(x, tf.tensordot(matrix, x, axes=1), axes=1)

    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient
    )
    vars = {
        "matrix": matrix,
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
    optimizer = modules["pymanopt"].optimizers.SteepestDescent(
        verbosity=0,
        max_iterations=1e5,
    )
    res = optimizer.run(
        vars["problem"], initial_point=vars["initial_point"]
    ).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    res = np.array(res)
    matrix = vars["matrix"]
    # Calculate the actual solution by a conventional eigenvalue decomposition.
    matrix = np.array(vars["matrix"])
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    ground_truth = eigenvectors[:, np.argmax(eigenvalues)]

    # Make sure both vectors have the same direction. Both are valid
    # eigenvectors, but for comparison we need to get rid of the sign
    # ambiguity.
    if np.sign(ground_truth[0]) != np.sign(res[0]):
        res = -res

    # Check norm between the two vectors is close to zero.
    if not np.allclose(ground_truth, res, atol=1.5e-5):
        print(
            "norm between the two vectors is "
            f"{np.linalg.norm(ground_truth - res)}"
        )
