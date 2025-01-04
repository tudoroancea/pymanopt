import os
import subprocess
from itertools import product

import pandas as pd


benchmarks = [
    "dominant_eigenvector",
    "low_rank_matrix_approximation",
    "packing_on_the_sphere",
    "pca",
]
versions = ["master", "dev"]
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
    # create two venvs
    subprocess.run("uv venv .venv_master", cwd=basedir, shell=True)
    subprocess.run(
        "uv pip install --exact pymanopt[backends]",
        cwd=basedir,
        shell=True,
        env=os.environ
        | {"VIRTUAL_ENV": os.path.join(basedir, ".venv_master")},
    )
    subprocess.run("uv venv .venv_dev", cwd=basedir, shell=True)
    subprocess.run(
        f"uv pip install --exact {wheel}[backends]",
        cwd=basedir,
        shell=True,
        env=os.environ | {"VIRTUAL_ENV": os.path.join(basedir, ".venv_dev")},
    )


def run_benchmarks():
    for version in versions:
        results_file = f"out2/results-{version}.csv"
        with open(os.path.join(basedir, results_file), "w") as f:
            f.write("benchmark,backend,optim_times")
        subprocess.run(
            f". .venv_{version}/bin/activate && python3 run_per_version.py "
            f"--benchmarks {','.join(benchmarks)} "  # noqa: E231
            f"--backends {','.join(backends)} "  # noqa: E231
            f" --iter 10 --results_file {results_file}",
            cwd=basedir,
            shell=True,
        )


def analyze_benchmarks():
    # check we are running in the correct directory
    assert (
        os.path.basename((basedir := os.path.abspath(os.curdir)))
        == "benchmarks"
    ), f"must be in benchmarks folder, not {basedir}"
    outdir = os.path.join(basedir, "out")

    # create multi index for the benchmark, backend and phase
    index = pd.MultiIndex.from_product(
        [benchmarks, backends, ["init", "autodiff", "optim"]],
        names=["benchmark", "backend", "phase"],
    ).drop(("packing_on_the_sphere", "numpy"))

    # create dataframe with two columns("master" and "dev")
    df = pd.DataFrame(columns=["master", "dev"], index=index)
    for benchmark, version, backend in product(benchmarks, versions, backends):
        if benchmark == "packing_on_the_sphere" and backend == "numpy":
            continue
        benchmark_timings = pd.read_csv(
            os.path.join(
                outdir,
                f"{benchmark}-{version}-{backend}",
                f"results-{backend}.csv",
            )
        )
        # add 3 rows for each benchmark
        df.at[(benchmark, backend, "init"), version] = benchmark_timings[
            "init"
        ].mean()
        df.at[(benchmark, backend, "autodiff"), version] = benchmark_timings[
            "autodiff"
        ].mean()
        df.at[(benchmark, backend, "optim"), version] = benchmark_timings[
            "optim"
        ].mean()

    # compute improvement in %
    df["speedup"] = df["master"] / df["dev"] - 1
    df["improving"] = df["speedup"] > 0

    # print the results
    print(df)


if __name__ == "__main__":
    # setup()
    run_benchmarks()
    # analyze_benchmarks()
