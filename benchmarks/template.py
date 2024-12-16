import argparse
import csv
import time
from typing import Any, Callable

import numpy as np


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
    parser.add_argument("-n", "--iter", type=int, default=1)
    backend: str = parser.parse_args().backend

    init_times, autodiff_times, optim_times = [], [], []

    # generate a set of seeds for each run, that will be the same no matter what
    # the backend is
    np.random.seed(127)
    seeds = np.random.randint(0, 1000, size=(parser.parse_args().iter))

    # run the benchmark for each seed
    for seed in seeds:
        np.random.seed(seed)
        t = time.perf_counter()
        modules, vars = init(backend)
        init_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        autodiff(backend, modules, vars)
        autodiff_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        res = optim(modules, vars)
        optim_times.append(time.perf_counter() - t)

        check_res(backend, modules, vars, res)

    print(f"init time: {np.mean(init_times):.3f}s")  # noqa: E231
    print(f"autodiff time: {np.mean(autodiff_times):.3f}s")  # noqa: E231
    print(f"optim time: {np.mean(optim_times):.3f}s")  # noqa: E231

    with open(f"results-{backend}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["init", "autodiff", "optim"])
        for init_time, autodiff_time, optim_time in zip(
            init_times, autodiff_times, optim_times
        ):
            writer.writerow([init_time, autodiff_time, optim_time])
