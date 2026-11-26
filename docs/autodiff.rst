Automatic Differentiation
=========================

Cost functions, gradients and Hessian-vector products (hvps) in Pymanopt must
be defined as Python callables annotated with one of the :ref:`backend
decorators<Backends>` below.
Decorating a callable with a backend decorator will wrap it in an instance of
the :class:`pymanopt.Function` class that provides a backend-agnostic
API to the :class:`pymanopt.core.problem.Problem` class to compute derivatives.

The signature of the decorated callable must match the point layout of the
manifold it is defined on.
For instance, for memory efficiency points on the
:class:`pymanopt.manifolds.fixed_rank.FixedRankEmbedded` manifold are not
represented as ``m x n`` matrices in the ambient space but rather as singular
value decompositions.
As such a cost function defined on the manifold must accept three arguments
``u``, ``s`` and ``vt``.
Refer to the documentation of the respective manifold on how points are
represented.

Cost functions written in plain NumPy can be differentiated by Pymanopt
itself, without any additional dependency, by decorating them with
``pymanopt.function.numpy(manifold, autodiff=True)``.
Without ``autodiff=True``, the gradient, and the Hessian if needed, must be
passed to :class:`pymanopt.Problem`.
Whenever a gradient or Hessian-vector product is needed, the cost function is
called on objects which record the NumPy operations it performs, as Autograd
does.
It may therefore read values other than its arguments, such as arrays or
parameters that change between calls, and Python control flow may depend on
the values of the arguments.

Elementwise functions, reductions, indexing, reshaping, matrix products and the
``np.linalg`` functions ``norm``, ``solve``, ``inv``, ``det`` and ``slogdet``
are supported.
Anything else raises an error, for instance ``np.einsum``, ``np.prod``,
``np.var``, ``np.linalg.eigh``, ``np.linalg.svd``, SciPy functions, writing a
value that depends on the arguments into an array, and converting such a value
to a Python float.
Use one of the other backends if a cost function needs them.

The derivative of the recorded operations is compiled to a function, which
later calls reuse as long as they record the same operations on arrays of the
same shapes and dtypes, with the same Python numbers.
A cost function using a Python number that changes on every call is therefore
differentiated again on every call, which is slower.

New backends can be created by inheriting from the
:class:`pymanopt.backends.backend.Backend` class, and creating a
backend decorator using :func:`pymanopt.function.decorator_factory`.

.. automodule:: pymanopt.function
