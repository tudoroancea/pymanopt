# pymanopt benchmarking

This folder contains benchmarks of pymanopt on several problems adapted from the examples.
The results are compared between the `master` branch and the current branch (corresponding
to two separate virtual environments). To setup these environments, we are using
[uv](https://docs.astral.sh/uv/) for its speed and simplicity. If you don't want to use it,
you should be able to reproduce the same results by approriately changing the `subprocess.run`
invocations in `benchmark.py` (which should amount to changing `uv venv` into `python -m venv`
`uv pip` into `pip`).

Particular attention was paid to make the tests as statistically significat as possible, and
comparable between the two branches. In particular:
- We run 20 tests and perform Welch's t-test to compare the mean runtimes of the two branches.
- We run each iteration using the same random seed, and the same initial guess (that was
  generated using functions from `np.random` instead of `manifold.random_point()` which would
  always use numpy functions on `master` but possibly other functions on the current branch).
