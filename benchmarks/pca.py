import os
from typing import Any

from template import run_benchmark


def init(backend: str):
    import numpy as np

    import pymanopt
    from pymanopt.manifolds import Stiefel

    modules = {"np": np, "pymanopt": pymanopt}

    dimension = 20
    num_samples = 500
    num_components = 5
    samples = np.random.normal(size=(num_samples, dimension)) @ np.diag(
        np.arange(1, dimension + 1)
    )
    samples -= samples.mean(axis=0)

    manifold = Stiefel(dimension, num_components)
    vars = {
        "samples": samples,
        "num_components": num_components,
        "manifold": manifold,
    }
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

        @pymanopt.function.autograd(manifold)
        def cost(w):
            return anp.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "jax":
        import jax.numpy as jnp

        @pymanopt.function.jax(manifold)
        def cost(w):
            return jnp.linalg.norm(samples - samples @ w @ w.T) ** 2

    elif backend == "pytorch":
        import torch

        modules["torch"] = torch

        samples = torch.from_numpy(samples)

        @pymanopt.function.pytorch(manifold)
        def cost(w):
            projector = w @ torch.transpose(w, 1, 0)
            return torch.norm(samples - samples @ projector) ** 2

    elif backend == "tensorflow":
        import tensorflow as tf

        modules["tf"] = tf

        samples = tf.constant(samples)

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
    vars["problem"] = problem
    return modules, vars


def autodiff(backend: str, modules: dict, vars: dict):
    if backend != "numpy":
        point = vars["manifold"].random_point()
        vec = vars["manifold"].random_tangent_vector(point)
        vars["problem"].euclidean_gradient(point)
        vars["problem"].euclidean_hessian(point, vec)


def optim(modules: dict, vars: dict):
    optimizer = modules["pymanopt"].optimizers.ConjugateGradient(
        verbosity=0, min_gradient_norm=1e-8, max_iterations=1e5
    )
    res = optimizer.run(vars["problem"]).point
    return res


def check_res(backend: str, modules: dict, vars: dict, res: Any):
    np = modules["np"]
    num_components = vars["num_components"]
    samples = vars["samples"]
    estimated_span_matrix = res

    if backend == "pytorch":
        estimated_span_matrix = estimated_span_matrix.cpu().detach().numpy()
    elif backend == "tensorflow":
        estimated_span_matrix = estimated_span_matrix.numpy()

    estimated_projector = estimated_span_matrix @ estimated_span_matrix.T

    eigenvalues, eigenvectors = np.linalg.eig(samples.T @ samples)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    span_matrix = eigenvectors[:, indices]
    projector = span_matrix @ span_matrix.T

    assert np.allclose(
        estimated_projector, projector, atol=1e-6
    ), f"error: {np.linalg.norm(estimated_projector - projector)}"


if __name__ == "__main__":
    run_benchmark(
        f"benchmark_{os.path.basename(__file__)}",
        init,
        autodiff,
        optim,
        check_res,
    )
