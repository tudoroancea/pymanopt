from typing import Any


def init(backend: str, dev=True):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Stiefel

    modules = {"np": np, "pymanopt": pymanopt}

    dimension = 100
    num_samples = 1000
    num_components = 10
    samples = np.random.normal(size=(num_samples, dimension)) @ np.diag(
        np.arange(1, dimension + 1)
    )
    samples -= samples.mean(axis=0)

    manifold = Stiefel(dimension, num_components)
    if dev:
        from pymanopt.backends.numpy_backend import NumpyBackend

        manifold.set_compatible_backend(NumpyBackend())
    initial_point = manifold.random_point()
    euclidean_gradient = None
    euclidean_hessian = None
    if backend == "numpy":

        @pymanopt.function.numpy(manifold)
        def cost(w):
            return np.linalg.norm(samples - samples @ w @ w.T) ** 2

        @pymanopt.function.numpy(manifold)
        def euclidean_gradient(w):
            return (
                -2
                * (
                    samples.T @ (samples - samples @ w @ w.T)
                    + (samples - samples @ w @ w.T).T @ samples
                )
                @ w
            )

        @pymanopt.function.numpy(manifold)
        def euclidean_hessian(w, h):
            return -2 * (
                samples.T @ (samples - samples @ w @ h.T) @ w
                + samples.T @ (samples - samples @ h @ w.T) @ w
                + samples.T @ (samples - samples @ w @ w.T) @ h
                + (samples - samples @ w @ h.T).T @ samples @ w
                + (samples - samples @ h @ w.T).T @ samples @ w
                + (samples - samples @ w @ w.T).T @ samples @ h
            )

    elif backend == "autograd":
        import autograd.numpy as anp

        samples = anp.asarray(samples)

        if dev:
            from pymanopt.backends.autograd_backend import AutogradBackend

            manifold.set_compatible_backend(AutogradBackend())
            initial_point = anp.array(initial_point)

        @pymanopt.function.autograd(manifold)
        def cost(w):
            return anp.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "jax":
        import jax.numpy as jnp

        samples = jnp.asarray(samples)

        if dev:
            from pymanopt.backends.jax_backend import JaxBackend

            manifold.set_compatible_backend(JaxBackend())
            initial_point = jnp.array(initial_point)

        @pymanopt.function.jax(manifold)
        def cost(w):
            return jnp.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "pytorch":
        import torch

        samples = torch.from_numpy(samples)

        if dev:
            from pymanopt.backends.pytorch_backend import PytorchBackend

            manifold.set_compatible_backend(PytorchBackend())
            initial_point = torch.tensor(initial_point, requires_grad=True)

        @pymanopt.function.pytorch(manifold)
        def cost(w):
            projector = w @ torch.transpose(w, 1, 0)
            return torch.norm(samples - samples @ projector) ** 2

    elif backend == "tensorflow":
        import tensorflow as tf

        samples = tf.constant(samples)

        if dev:
            from pymanopt.backends.tensorflow_backend import TensorflowBackend

            manifold.set_compatible_backend(TensorflowBackend())
            initial_point = tf.constant(initial_point)

        @pymanopt.function.tensorflow(manifold)
        def cost(w):
            projector = w @ tf.transpose(w)
            return tf.norm(samples - samples @ projector) ** 2

    problem = pymanopt.Problem(
        manifold,
        cost,
        euclidean_gradient=euclidean_gradient,
        euclidean_hessian=euclidean_hessian,
    )
    vars = {
        "samples": samples,
        "num_components": num_components,
        "manifold": manifold,
        "initial_point": initial_point,
        "problem": problem,
    }
    return modules, vars


def autodiff(backend: str, modules: dict, vars: dict):
    if backend != "numpy":
        point = vars["manifold"].random_point()
        vec = vars["manifold"].random_tangent_vector(point)
        vars["problem"].euclidean_gradient(point)
        vars["problem"].euclidean_hessian(point, vec)


def optim(modules: dict, vars: dict):
    optimizer = modules["pymanopt"].optimizers.TrustRegions(verbosity=0)
    res = optimizer.run(
        vars["problem"], initial_point=vars["initial_point"]
    ).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    num_components = vars["num_components"]
    samples = np.array(vars["samples"])
    estimated_span_matrix = np.array(res)
    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    if not np.allclose(estimated_projector, projector, atol=1e-6):
        print(f"error: {np.linalg.norm(estimated_projector - projector)}")
