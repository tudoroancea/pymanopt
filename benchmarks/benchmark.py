# flake8: noqa E231
import ast
import os
import subprocess

import colorama
import numpy as np
import pandas as pd


benchmarks = [
    "dominant_eigenvector",
    "low_rank_matrix_approximation",
    "packing_on_the_sphere",
    "pca",
]
versions = ["dev", "master"]
# versions = ["master", "dev"]
backends = [
    "numpy",
    "autograd",
    "pytorch",
    "jax",
    "tensorflow",
]

# check we are running in the correct directory
assert (
    os.path.basename((basedir := os.path.abspath(os.curdir))) == "benchmarks"
), f"must be in benchmarks folder, not {basedir}"
# create output directory
outdir = os.path.join(basedir, "out2")
os.makedirs(outdir, exist_ok=True)


def setup():
    # find the built pymanopt dev wheel
    wheel = None
    for f in os.listdir((distdir := os.path.join(basedir, "../dist"))):
        if f.endswith(".whl"):
            wheel = os.path.join(distdir, f)
    assert (
        wheel is not None
    ), f"dev wheel doesn't exist in {os.path.abspath(distdir)}"
    # create venvs
    if not os.path.exists(os.path.join(basedir, ".venv_master")):
        subprocess.run("uv venv .venv_master", cwd=basedir, shell=True)
        subprocess.run(
            "uv pip install --exact pymanopt[backends]",
            cwd=basedir,
            shell=True,
            env=os.environ
            | {"VIRTUAL_ENV": os.path.join(basedir, ".venv_master")},
        )
    if not os.path.exists(os.path.join(basedir, ".venv_dev")):
        subprocess.run("uv venv .venv_dev", cwd=basedir, shell=True)
        subprocess.run(
            f"uv pip install --exact {wheel}[backends]",
            cwd=basedir,
            shell=True,
            env=os.environ
            | {"VIRTUAL_ENV": os.path.join(basedir, ".venv_dev")},
        )


def run_benchmarks():
    for version in versions:
        results_file = os.path.join(outdir, f"results-{version}.csv")
        with open(os.path.join(basedir, results_file), "w") as f:
            f.write("benchmark,backend,optim_times")
        subprocess.run(
            f". .venv_{version}/bin/activate && python3 run_per_version.py "
            f"--benchmarks {','.join(benchmarks)} "  # noqa: E231
            f"--backends {','.join(backends)} "  # noqa: E231
            f" --iter 2 --results_file {results_file}",
            cwd=basedir,
            shell=True,
            stderr=subprocess.DEVNULL,
        )


def analyze_benchmarks():
    # create multi index for the benchmark, backend and phase
    index = pd.MultiIndex.from_product(
        [benchmarks, backends],
        names=["benchmark", "backend"],
    )
    if ("packing_on_the_sphere", "numpy") in index:
        index = index.drop(("packing_on_the_sphere", "numpy"))

    # create dataframe with two columns("master" and "dev")
    df = pd.DataFrame(columns=versions, index=index)

    # read dfs from results-dev.csv and results-master.csv to fill the two columns
    for version in versions:
        benchmark_timings = pd.read_csv(
            os.path.join(outdir, f"results-{version}.csv")
        )
        benchmark_timings["optim_times"] = benchmark_timings[
            "optim_times"
        ].apply(ast.literal_eval)
        df[version] = benchmark_timings.set_index(
            ["benchmark", "backend"]
        ).apply(lambda x: np.mean(x["optim_times"]), axis=1)

    df["speedup"] = df["master"] / df["dev"]

    colorama.init()
    print("=" * 85)
    print(
        f"{'Benchmark':<35}{'Backend':<15}{'Master (s)':<13}{'Dev (s)':<13}{'Speedup':<10}"
    )
    print("-" * 85)
    for idx in df.index:
        benchmark, backend = idx
        master_time = df.loc[idx, "master"]
        dev_time = df.loc[idx, "dev"]
        speedup = df.loc[idx, "speedup"]

        # Color the speedup based on whether it's an improvement or regression
        color = (
            colorama.Fore.YELLOW
            if 0.9 <= speedup <= 1.1
            else (colorama.Fore.GREEN if speedup > 1.1 else colorama.Fore.RED)
        )
        speedup_str = f"{color}{speedup:.2f}x{colorama.Fore.RESET}"

        print(
            f"{benchmark:<35}{backend:<15}{master_time:<13.3f}{dev_time:<13.3f}{speedup_str:<10}"
        )

    print("=" * 85)


if __name__ == "__main__":
    setup()
    run_benchmarks()
    analyze_benchmarks()
