"""Compiles numpy autodiff programs with numba (experimental).

Each recorded NumPy call is lowered to code numba can compile, using the
shapes and dtypes recorded while tracing. Values of shape ``()`` are scalars in
the generated code, since numba handles zero-dimensional arrays poorly.
Programs with operations that cannot be lowered fall back to the plain NumPy
program with a warning.
"""

import warnings

import numba
import numpy as np
from numba.core.errors import NumbaError, NumbaPerformanceWarning

from pymanopt.backends import _numpy_autodiff as autodiff
from pymanopt.backends._numpy_autodiff import _Var

__all__ = ["compile"]


class _Unsupported(Exception):
    pass


def compile(program, args):
    """Compiles ``program`` with numba, falling back to NumPy if needed."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            function = _Compiler(program).build()
            # Compile now so that unsupported code falls back here.
            function(*args)
        return function
    except (_Unsupported, NumbaError) as error:
        message = str(error).strip().splitlines()[0]
        warnings.warn(
            f"numba could not compile the program, using NumPy instead: "
            f"{message}",
            stacklevel=2,
        )
        return program.compile()


def _number(value):
    if isinstance(value, np.generic):
        source = _number(value.item())
        if value.dtype in (np.float64, np.int64, np.bool_, np.complex128):
            return source
        return f"np.{value.dtype.name}({source})"
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, complex):
        return f"complex({_number(value.real)}, {_number(value.imag)})"
    if isinstance(value, float) and not np.isfinite(value):
        return "np.nan" if np.isnan(value) else f"{'-' * (value < 0)}np.inf"
    if isinstance(value, (int, float)):
        return repr(value)
    raise _Unsupported(f"constant of type {type(value).__name__}")


class _Compiler:
    def __init__(self, program):
        self.program = program
        self.constants = []
        self.names = {}
        self.lines = []

    def shape(self, value):
        if isinstance(value, _Var):
            return value.shape
        return np.shape(value)

    def dtype(self, value):
        if isinstance(value, _Var):
            return value.dtype
        return np.asarray(value).dtype

    def value(self, value):
        """Source code for an argument of a recorded call."""
        if isinstance(value, _Var):
            return f"v{value.id}"
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return _number(value[()])
            if id(value) not in self.names:
                self.names[id(value)] = f"c{len(self.constants)}"
                self.constants.append(value)
            return self.names[id(value)]
        if isinstance(value, (tuple, list)):
            return f"({''.join(f'{self.value(v)}, ' for v in value)})"
        if isinstance(value, (np.dtype, type)):
            return f"np.{np.dtype(value).name}"
        return _number(value)

    def cast(self, value, dtype):
        source = self.value(value)
        if self.dtype(value) == dtype:
            return source
        if self.shape(value) == ():
            return f"np.{np.dtype(dtype).name}({source})"
        return f"{source}.astype(np.{np.dtype(dtype).name})"

    def index(self, index):
        items = index if type(index) is tuple else (index,)
        parts = []
        num_arrays = 0
        for item in items:
            if isinstance(item, slice):
                parts.append(
                    ":".join(
                        "" if i is None else self.value(i)
                        for i in (item.start, item.stop, item.step)
                    )
                )
            elif item is None:
                parts.append("None")
            elif item is Ellipsis:
                parts.append("...")
            else:
                if isinstance(item, list):
                    item = np.array(item)
                num_arrays += self.shape(item) != ()
                parts.append(self.value(item))
        if num_arrays > 1:
            raise _Unsupported("indexing with several arrays")
        return ", ".join(parts), num_arrays

    def build(self):
        for function, args, kwargs, out in self.program.live_equations():
            lowering = _LOWERINGS.get(function)
            if lowering is None:
                name = getattr(function, "__name__", repr(function))
                raise _Unsupported(f"no lowering for '{name}'")
            source = lowering(self, out, *args, **kwargs)
            if source is not None:
                self.lines.append(f"    v{out.id} = {source}")
        returns = []
        for output in self.program.outputs:
            source = self.value(output)
            if self.shape(output) == ():
                returns.append(source)
            elif isinstance(output, _Var):
                returns.append(f"np.ascontiguousarray({source})")
            else:
                returns.append(f"{source}.copy()")
        parameters = [self.value(var) for var in self.program.inputs]
        parameters += [f"c{i}" for i in range(len(self.constants))]
        source = "\n".join(
            [f"def program({', '.join(parameters)}):"]
            + self.lines
            + [f"    return ({''.join(f'{r}, ' for r in returns)})"]
        )
        namespace = {"np": np, "_undiagonal_2d": _undiagonal_2d}
        exec(source, namespace)
        compiled = numba.njit(namespace["program"])
        constants = tuple(self.constants)

        def program(*args):
            return compiled(*args, *constants)

        program.source = source
        return program


@numba.njit
def _undiagonal_2d(g, rows, columns, offset):
    out = np.zeros((rows, columns), g.dtype)
    for i in range(g.shape[0]):
        out[i + max(-offset, 0), i + max(offset, 0)] = g[i]
    return out


def _reshape(c, x, shape):
    source = c.value(x)
    if c.shape(x) == shape:
        return source
    if c.shape(x) == ():
        return f"np.full({shape}, {source})"
    if shape == ():
        return f"np.ascontiguousarray({source}).ravel()[0]"
    return f"np.ascontiguousarray({source}).reshape({shape})"


def _reduction(name):
    def lower(c, out, x, axis=None, dtype=None, keepdims=False, **kwargs):
        source, shape = c.value(x), c.shape(x)
        if shape == ():
            return source
        axes = sorted(autodiff._normalize_axes(axis, len(shape)), reverse=True)
        if len(axes) == len(shape):
            source = f"np.{name}({source})"
        elif name != "sum":
            raise _Unsupported(f"np.{name} along an axis")
        else:
            for a in axes:
                source = f"np.sum({source}, axis={a})"
        if out.shape == ():
            return source
        if len(axes) == len(shape):
            return f"np.full({out.shape}, {source})"
        return f"np.reshape({source}, {out.shape})"

    return lower


def _binary(operator):
    return lambda c, out, x, y: f"({c.value(x)} {operator} {c.value(y)})"


def _unary(name):
    return lambda c, out, x: f"np.{name}({c.value(x)})"


def _matmul(c, out, a, b):
    if not (1 <= len(c.shape(a)) <= 2 and 1 <= len(c.shape(b)) <= 2):
        raise _Unsupported("batched matrix products")
    return f"({c.cast(a, out.dtype)} @ {c.cast(b, out.dtype)})"


def _transpose(c, out, x, axes=None):
    ndim = len(c.shape(x))
    if ndim <= 1:
        return c.value(x)
    if axes is None:
        return f"np.transpose({c.value(x)})"
    return (
        f"np.transpose({c.value(x)}, {autodiff._normalize_axes(axes, ndim)})"
    )


def _swapaxes(c, out, x, a, b):
    axes = list(range(len(c.shape(x))))
    axes[a], axes[b] = axes[b], axes[a]
    return _transpose(c, out, x, tuple(axes))


def _real(c, out, x):
    if np.issubdtype(c.dtype(x), np.complexfloating):
        return f"np.real({c.value(x)})"
    return c.value(x)


def _imag(c, out, x):
    if np.issubdtype(c.dtype(x), np.complexfloating):
        return f"np.imag({c.value(x)})"
    return (
        _number(np.zeros((), out.dtype)[()])
        if out.shape == ()
        else (f"np.zeros({out.shape}, np.{out.dtype.name})")
    )


def _conjugate(c, out, x):
    if np.issubdtype(c.dtype(x), np.complexfloating):
        return f"np.conj({c.value(x)})"
    return c.value(x)


def _diagonal(c, out, x, offset=0, axis1=0, axis2=1):
    if len(c.shape(x)) != 2 or (axis1, axis2) != (0, 1):
        raise _Unsupported("diagonals of arrays with more than two axes")
    return f"np.diag({c.value(x)}, {offset})"


def _undiagonal(c, out, g, shape, offset, axis1, axis2):
    if len(shape) != 2 or (axis1, axis2) != (0, 1):
        raise _Unsupported("diagonals of arrays with more than two axes")
    return f"_undiagonal_2d({c.value(g)}, {shape[0]}, {shape[1]}, {offset})"


def _getitem(c, out, x, index):
    return f"{c.value(x)}[{c.index(index)[0]}]"


def _scatter_add(c, out, g, shape, index):
    source, num_arrays = c.index(index)
    if num_arrays:
        raise _Unsupported("gradients of indexing with arrays")
    self = f"v{out.id}"
    c.lines.append(f"    {self} = np.zeros({shape}, np.{out.dtype.name})")
    c.lines.append(f"    {self}[{source}] = {c.value(g)}")


def _linalg(name):
    def lower(c, out, *args):
        if any(len(c.shape(a)) > 2 for a in args):
            raise _Unsupported(f"batched np.linalg.{name}")
        dtype = np.result_type(*(c.dtype(a) for a in args))
        sources = ", ".join(c.cast(a, dtype) for a in args)
        return f"np.linalg.{name}({sources})"

    return lower


def _no_axis(name):
    def lower(c, out, x, axis=None, **kwargs):
        if axis is not None:
            raise _Unsupported(f"np.{name} along an axis")
        return f"np.{name}({c.value(x)})"

    return lower


_LOWERINGS = {
    np.add: _binary("+"),
    np.subtract: _binary("-"),
    np.multiply: _binary("*"),
    np.true_divide: _binary("/"),
    np.power: _binary("**"),
    np.less: _binary("<"),
    np.less_equal: _binary("<="),
    np.greater: _binary(">"),
    np.greater_equal: _binary(">="),
    np.equal: _binary("=="),
    np.not_equal: _binary("!="),
    np.bitwise_and: _binary("&"),
    np.bitwise_or: _binary("|"),
    np.negative: lambda c, out, x: f"(-{c.value(x)})",
    np.matmul: _matmul,
    np.maximum: lambda c, out, x, y: f"np.maximum({c.value(x)}, {c.value(y)})",
    np.minimum: lambda c, out, x, y: f"np.minimum({c.value(x)}, {c.value(y)})",
    np.conjugate: _conjugate,
    np.absolute: _unary("abs"),
    np.real: _real,
    np.imag: _imag,
    np.sum: _reduction("sum"),
    np.max: _reduction("max"),
    np.amax: _reduction("max"),
    np.min: _reduction("min"),
    np.amin: _reduction("min"),
    np.argmax: _no_axis("argmax"),
    np.argmin: _no_axis("argmin"),
    np.any: _no_axis("any"),
    np.all: _no_axis("all"),
    np.reshape: lambda c, out, x, *args, **kwargs: _reshape(c, x, out.shape),
    np.ravel: lambda c, out, x, *args, **kwargs: _reshape(c, x, out.shape),
    np.squeeze: lambda c, out, x, *args, **kwargs: _reshape(c, x, out.shape),
    np.expand_dims: lambda c, out, x, *args: _reshape(c, x, out.shape),
    np.broadcast_to: lambda c, out, x, shape: (
        c.value(x)
        if c.shape(x) == out.shape
        else (
            f"np.full({out.shape}, {c.value(x)})"
            if c.shape(x) == ()
            else f"np.broadcast_to({c.value(x)}, {out.shape})"
        )
    ),
    np.transpose: _transpose,
    np.swapaxes: _swapaxes,
    np.where: lambda c, out, cond, x, y: (
        f"np.where({c.value(cond)}, {c.value(x)}, {c.value(y)})"
    ),
    np.clip: lambda c, out, x, a_min, a_max: (
        f"np.clip({c.value(x)}, {c.value(a_min)}, {c.value(a_max)})"
    ),
    np.triu: lambda c, out, x, k=0: f"np.triu({c.value(x)}, {k})",
    np.tril: lambda c, out, x, k=0: f"np.tril({c.value(x)}, {k})",
    np.diagonal: _diagonal,
    autodiff._diag: lambda c, out, v, k: f"np.diag({c.value(v)}, {k})",
    autodiff._undiagonal: _undiagonal,
    autodiff._getitem: _getitem,
    autodiff._scatter_add: _scatter_add,
    autodiff._astype: lambda c, out, x, dtype: c.cast(x, dtype),
    autodiff._concatenate: lambda c, out, *arrays, axis=0: (
        f"np.concatenate({c.value(arrays)}, axis={axis})"
    ),
    np.linalg.solve: _linalg("solve"),
    np.linalg.inv: _linalg("inv"),
    np.linalg.det: _linalg("det"),
    autodiff._logabsdet: lambda c, out, a: (
        f"{_linalg('slogdet')(c, out, a)}[1]"
    ),
    autodiff._slogdet_sign: lambda c, out, a: (
        f"{_linalg('slogdet')(c, out, a)}[0]"
    ),
}
for _name in [
    "exp",
    "expm1",
    "log",
    "log1p",
    "log2",
    "log10",
    "sqrt",
    "square",
    "reciprocal",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "sign",
    "floor",
    "ceil",
    "isfinite",
    "isinf",
    "isnan",
    "logical_not",
]:
    _LOWERINGS[getattr(np, _name)] = _unary(_name)
