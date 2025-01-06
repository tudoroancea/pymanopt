import argparse
import cProfile
import csv
import importlib
import os
import time
from itertools import product

import numpy as np


def main():
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

    branch = "master" if "master" in results_file else "dev"
    outdir = os.path.dirname(os.path.abspath(results_file))

    # create a new csv file for the results
    with open(results_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["benchmark", "backend", "optim_times"])

    for benchmark, backend in product(benchmarks, backends):
        if (benchmark, backend) == ("packing_on_the_sphere", "numpy"):
            continue
        print(
            f"running benchmark {benchmark} with backend {backend} "
            f"on branch {branch}"
        )
        benchmark_module = importlib.import_module(benchmark)

        # run the benchmark for each seed
        optim_times = []
        pr = cProfile.Profile()
        for _ in range(iter):
            np.random.seed(127)
            modules, vars = benchmark_module.init(backend, dev=branch == "dev")

            np.random.seed(127)
            benchmark_module.autodiff(backend, modules, vars)

            pr.enable()
            t = time.perf_counter()
            res = benchmark_module.optim(modules, vars)
            optim_times.append(time.perf_counter() - t)
            pr.disable()

            benchmark_module.check_res(backend, modules, vars, res)

        pr.dump_stats(
            os.path.join(outdir, f"{benchmark}_{backend}_{branch}.prof")
        )
        with open(results_file, "a") as f:
            f.write(
                f"{benchmark},{backend},\"[{','.join(map(str, optim_times))}]\"\n"  # noqa: E231, B950
            )


if __name__ == "__main__":
    main()
