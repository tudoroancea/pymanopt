import argparse
import csv
import importlib
import time
from itertools import product

import numpy as np


def main(
    # init: Callable[[str], tuple[dict, dict]],
    # autodiff: Callable[[str, dict, dict], None],
    # optim: Callable[[dict, dict], Any],
    # check_res: Callable[[str, dict, dict, Any], None],
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str)
    parser.add_argument("--backends", type=str)
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--results_file", type=str)
    args = parser.parse_args()
    benchmarks = str(args.benchmarks).split(",")
    backends = str(args.backends).split(",")
    iter = args.iter
    results_file = args.results_file

    init_times, autodiff_times, optim_times = [], [], []

    # create a new csv file for the results
    with open(results_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "backend", "optim_times"])

    for benchmark, backend in product(benchmarks, backends):
        benchmark_module = importlib.import_module(benchmark)

        # generate a set of seeds for each run, that will be the same no matter what
        # the backend is
        np.random.seed(127)
        seeds = np.random.randint(0, 1000, size=(iter, 2))

        # run the benchmark for each seed
        for seed in seeds:
            np.random.seed(seed[0])
            t = time.perf_counter()
            modules, vars = benchmark_module.init(backend)
            init_times.append(time.perf_counter() - t)

            # np.random.seed(seed[1])
            # t = time.perf_counter()
            # benchmark_module.autodiff(backend, modules, vars)
            # autodiff_times.append(time.perf_counter() - t)
            autodiff_times.append(0.0)

            t = time.perf_counter()
            res = benchmark_module.optim(modules, vars)
            optim_times.append(time.perf_counter() - t)

            benchmark_module.check_res(backend, modules, vars, res)

        # print(f"init time: {np.mean(init_times):.3f}s")  # noqa: E231
        # print(f"autodiff time: {np.mean(autodiff_times):.3f}s")  # noqa: E231
        # print(f"optim time: {np.mean(optim_times):.3f}s")  # noqa: E231

        with open(results_file, "a") as f:
            f.write(
                f"{benchmark},{backend},[{','.join(map(str, optim_times))}]\n"  # noqa: E231, B950
            )


if __name__ == "__main__":
    main()
