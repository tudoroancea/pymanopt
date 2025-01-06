# flake8: noqa E231
import ast
import os
import subprocess

import colorama
import numpy as np
import pandas as pd
import scipy.stats


benchmarks = [
    "dominant_eigenvector",
    "low_rank_matrix_approximation",
    "pca",
    # "packing_on_the_sphere",
]
versions = ["dev", "master"]
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
            "uv pip install --exact -e ..[backends]",
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
            f" --iter 20 --results_file {results_file}",
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
    df = pd.DataFrame(index=index)

    # read dfs from results-dev.csv and results-master.csv to fill the two columns
    for version in versions:
        benchmark_timings = pd.read_csv(
            os.path.join(outdir, f"results-{version}.csv")
        ).set_index(["benchmark", "backend"])
        df[version + "_times"] = benchmark_timings["optim_times"].apply(
            ast.literal_eval
        )
        df[version + "_avg"] = df[version + "_times"].apply(
            lambda x: np.mean(x)
        )
        df[version + "_std"] = df[version + "_times"].apply(
            lambda x: np.std(x, ddof=1)
        )

    # compute speedup
    df["avg_speedup"] = df["master_avg"] / df["dev_avg"]

    # t-test to check if the speedup is statistically significant
    for id in df.index:
        avg_speedup = df.loc[id, "avg_speedup"]
        if avg_speedup > 1.1:
            alternative = "greater"
        elif avg_speedup < 0.9:
            alternative = "greater"
        else:
            alternative = "two-sided"

        res = scipy.stats.ttest_ind_from_stats(
            mean1=df.loc[id, "dev_avg"],
            std1=df.loc[id, "dev_std"],
            nobs1=len(df.loc[id, "dev_times"]),
            mean2=df.loc[id, "master_avg"],
            std2=df.loc[id, "master_std"],
            nobs2=len(df.loc[id, "master_times"]),
            equal_var=False,
            alternative=alternative,
        )
        df.loc[id, "pvalue"] = res.pvalue

    colorama.init()
    print("=" * 100)
    print(
        f"{'Benchmark':<35}{'Backend':<15}{'Master (s)':<15}{'Dev (s)':<15}{'Avg speedup':<15}{'p-value':<10}"
    )
    print("-" * 100)
    for idx in df.index:
        row_str = ""

        benchmark, backend = idx
        row_str += f"{benchmark:<35}{backend:<15}"

        master_avg = df.loc[idx, "master_avg"]
        master_std = df.loc[idx, "master_std"]
        row_str += f"{f'{master_avg:.3f}±{master_std:.3f}':<15}"

        dev_avg = df.loc[idx, "dev_avg"]
        dev_std = df.loc[idx, "dev_std"]
        row_str += f"{f'{dev_avg:.3f}±{dev_std:.3f}':<15}"

        avg_speedup = df.loc[idx, "avg_speedup"]
        row_str += (
            f"{colorama.Fore.YELLOW if 0.9 <= avg_speedup <= 1.1 else (colorama.Fore.GREEN if avg_speedup > 1.1 else colorama.Fore.RED)}"
            f"{f'{avg_speedup:.2f}x':<15}"
            f"{colorama.Fore.RESET}"
        )

        pvalue = df.loc[idx, "pvalue"]
        row_str += (
            f"{colorama.Fore.GREEN if pvalue < 0.05 else colorama.Fore.RED}"
            f"{f'{pvalue:.3f}':<15}"
            f"{colorama.Fore.RESET}"
        )

        # conv_interval_low = df.loc[idx, "conf_interval_low"]
        # conv_interval_high = df.loc[idx, "conf_interval_high"]

        print(row_str)

    print("=" * 100)

    # unravel index to columns and export results to csv
    df.reset_index().to_csv(
        os.path.join(outdir, "analysis.csv"), index=False, float_format="%.3f"
    )


if __name__ == "__main__":
    setup()
    # run_benchmarks()
    analyze_benchmarks()
