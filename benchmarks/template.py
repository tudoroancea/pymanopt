import argparse
import time
from typing import Any, Callable

import numpy as np


np.random.seed(127)


def run_benchmark(
    name: str,
    init: Callable[[str], tuple[dict, dict]],
    autodiff: Callable[[str, dict, dict], None],
    optim: Callable[[dict, dict], Any],
    check_res: Callable[[str, dict, dict, Any], None],
):
    parser = argparse.ArgumentParser(name)
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        help="backend to run the test on",
        choices=["numpy", "autograd", "jax", "pytorch", "tensorflow"],
        default="numpy",
    )
    backend: str = parser.parse_args().backend
    print(f"Running benchmark {name} with '{backend}' backend")

    t = time.perf_counter()
    modules, vars = init(backend)
    init_time = time.perf_counter() - t

    t = time.perf_counter()
    autodiff(backend, modules, vars)
    autodiff_time = time.perf_counter() - t

    t = time.perf_counter()
    res = optim(modules, vars)
    optim_time = time.perf_counter() - t

    check_res(backend, modules, vars, res)

    print(f"========= BENCHMARK SUMMARY: {__file__} =============")
    print(f"init time: {init_time:.3f}s")  # noqa: E231
    print(f"autodiff time: {autodiff_time:.3f}s")  # noqa: E231
    print(f"optim time: {optim_time:.3f}s")  # noqa: E231
