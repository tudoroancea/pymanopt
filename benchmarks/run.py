import os
import subprocess
from itertools import product


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


def setup():
    # check we are running in the correct directory
    assert (
        os.path.basename((basedir := os.path.abspath(os.curdir)))
        == "benchmarks"
    ), f"must be in benchmarks folder, not {basedir}"
    # create output directory
    outdir = os.path.join(basedir, "out")
    os.makedirs(outdir, exist_ok=True)
    # find the built pymanopt dev wheel
    wheel = None
    for f in os.listdir((distdir := os.path.join(basedir, "../dist"))):
        if f.endswith(".whl"):
            wheel = os.path.join(distdir, f)
    assert (
        wheel is not None
    ), f"dev wheel doesn't exist in {os.path.abspath(distdir)}"
    # setup every benchmark
    for benchmark, version, backend in product(benchmarks, versions, backends):
        print(
            "========== SETUP BENCHMARK:",
            f"{benchmark}-{version}-{backend} ===========",
        )
        if benchmark == "packing_on_the_sphere" and backend == "numpy":
            continue
        # create the benchmark folder
        benchmark_dir = os.path.join(
            outdir, f"{benchmark}-{version}-{backend}"
        )
        os.makedirs(benchmark_dir, exist_ok=True)
        # create venv
        venv_dir = os.path.join(benchmark_dir, ".venv")
        if not os.path.exists(venv_dir):
            subprocess.run("uv venv", cwd=benchmark_dir, shell=True)
        # construct the requirements
        deps = "pymanopt" if version == "master" else wheel
        if backend == "autograd":
            deps += "[autograd]"
        elif backend == "pytorch":
            deps += "[torch]"
        elif backend == "jax":
            deps += "[jax]"
        elif backend == "tensorflow":
            deps += "[tensorflow]"
        # install requirements
        subprocess.run(
            f"uv pip install --exact '{deps}'",
            cwd=benchmark_dir,
            shell=True,
            env=os.environ | {"VIRTUAL_ENV": venv_dir},
        )
        # symlink the benchmark scripts
        for f in [f"{benchmark}.py", "template.py"]:
            if not os.path.exists(os.path.join(benchmark_dir, f)):
                os.symlink(
                    os.path.join(basedir, f), os.path.join(benchmark_dir, f)
                )


def run_benchmarks():
    # check we are running in the correct directory
    assert (
        os.path.basename((basedir := os.path.abspath(os.curdir)))
        == "benchmarks"
    ), f"must be in benchmarks folder, not {basedir}"
    outdir = os.path.join(basedir, "out")
    # run every benchmark
    for benchmark, version, backend in product(benchmarks, versions, backends):
        if benchmark == "packing_on_the_sphere" and backend == "numpy":
            continue
        benchmark_dir = os.path.join(
            outdir, f"{benchmark}-{version}-{backend}"
        )
        # call the process once or twice to warm up the cache
        print(
            "========= RUNNING BENCHMARK:"
            f" {os.path.basename(benchmark_dir)} ============="
        )
        subprocess.run(
            f"source .venv/bin/activate && python3 {benchmark}.py -b {backend}",
            cwd=benchmark_dir,
            shell=True,
        )


if __name__ == "__main__":
    setup()
    run_benchmarks()
