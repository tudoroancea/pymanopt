"""Reverse-mode automatic differentiation of plain NumPy code.

A cost function is run once on :class:`Tracer` objects, which intercept NumPy
calls through ``__array_ufunc__`` and ``__array_function__`` and record them as
a straight-line :class:`Program`. The gradient is obtained by tracing the
backward pass of that program, so it is a program too, and Hessian-vector
products differentiate the gradient in the same way. Programs are compiled to
Python functions that only call NumPy, and cached per argument shapes and
dtypes.

Gradients with respect to complex arguments are ``df/dx + i df/dy``, the
gradient for the real inner product ``Re <a, b>``, which is the convention
used by pymanopt.
"""

import functools

import numpy as np

__all__ = ["TracingError", "gradient", "hessian_vector_product"]


class TracingError(Exception):
    """Raised when Python control flow depends on a traced value."""


class _Var:
    __slots__ = ("id",)

    def __init__(self, id):
        self.id = id


def _map(function, value, cls):
    if isinstance(value, cls):
        return function(value)
    if type(value) in (tuple, list):
        return type(value)(_map(function, item, cls) for item in value)
    return value


def _leaves(value, cls):
    if isinstance(value, cls):
        yield value
    elif type(value) in (tuple, list):
        for item in value:
            yield from _leaves(item, cls)


def _map_call(function, args, kwargs, cls):
    return (
        _map(function, args, cls),
        {key: _map(function, value, cls) for key, value in kwargs.items()},
    )


class _Trace:
    def __init__(self, concrete):
        self.concrete = concrete
        self.equations = []
        self.num_vars = 0

    def new_tracer(self, value):
        if not hasattr(value, "dtype"):
            value = np.asarray(value)
        tracer = Tracer(self, _Var(self.num_vars), value)
        self.num_vars += 1
        return tracer

    def _value(self, tracer):
        if tracer._trace is not self:
            raise TracingError("Values from different traces were mixed")
        return tracer.value

    def record(self, function, args, kwargs):
        values, value_kwargs = _map_call(self._value, args, kwargs, Tracer)
        tracer = self.new_tracer(function(*values, **value_kwargs))
        args, kwargs = _map_call(lambda t: t.var, args, kwargs, Tracer)
        self.equations.append((function, args, kwargs, tracer.var))
        return tracer


class Program:
    """A straight-line sequence of NumPy calls."""

    def __init__(self, inputs, equations, outputs):
        self.inputs = inputs
        self.equations = equations
        self.outputs = outputs

    def evaluate(self, *args):
        env = {var.id: arg for var, arg in zip(self.inputs, args)}

        def lookup(var):
            return env[var.id]

        for function, eq_args, eq_kwargs, out in self.equations:
            values, kwargs = _map_call(lookup, eq_args, eq_kwargs, _Var)
            env[out.id] = function(*values, **kwargs)
        return env, [_map(lookup, output, _Var) for output in self.outputs]

    def compile(self):
        live = {o.id for o in self.outputs if isinstance(o, _Var)}
        equations = []
        for equation in reversed(self.equations):
            if equation[3].id in live:
                equations.append(equation)
                live.update(
                    var.id
                    for var in _leaves(
                        (equation[1], *equation[2].values()), _Var
                    )
                )
        equations.reverse()

        namespace = {"_output": _output, "_copy": np.array}
        names = {}

        def name(value):
            if isinstance(value, _Var):
                return f"v{value.id}"
            if type(value) in (tuple, list):
                items = "".join(f"{name(item)}, " for item in value)
                return f"({items})" if type(value) is tuple else f"[{items}]"
            if id(value) not in names:
                names[id(value)] = f"c{len(names)}"
                namespace[names[id(value)]] = value
            return names[id(value)]

        inputs = ", ".join(name(var) for var in self.inputs)
        lines = [f"def program({inputs}):"]
        for function, args, kwargs, out in equations:
            call = [name(arg) for arg in args]
            call += [f"{key}={name(value)}" for key, value in kwargs.items()]
            lines.append(
                f"    v{out.id} = {name(function)}({', '.join(call)})"
            )
        returns = "".join(
            (
                f"_output({name(o)}), "
                if isinstance(o, _Var)
                else f"_copy({name(o)}), "
            )
            for o in self.outputs
        )
        lines.append(f"    return ({returns})")
        exec("\n".join(lines), namespace)
        return namespace["program"]


def _output(value):
    # Gradients built with np.broadcast_to are read-only views.
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        return value.copy()
    return value


def trace(function, args, concrete=False):
    """Records the NumPy calls ``function`` makes on ``args``."""
    trace = _Trace(concrete)
    tracers = [trace.new_tracer(arg) for arg in args]
    result = function(*tracers)
    results = result if isinstance(result, (list, tuple)) else [result]
    outputs = [r.var if isinstance(r, Tracer) else r for r in results]
    return Program([t.var for t in tracers], trace.equations, outputs)


##############################################################################
# Tracer
##############################################################################


class Tracer:
    """Stands in for an array while a function is traced."""

    __slots__ = ("_trace", "var", "value")
    __array_priority__ = 1000

    def __init__(self, trace, var, value):
        self._trace = trace
        self.var = var
        self.value = value

    def __repr__(self):
        return f"Tracer(shape={self.shape}, dtype={self.dtype})"

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or "out" in kwargs:
            raise NotImplementedError(
                f"numpy autodiff does not support '{ufunc.__name__}.{method}' "
                "or in-place updates of traced arrays"
            )
        if ufunc not in _VJPS and ufunc not in _NONDIFFERENTIABLE:
            raise NotImplementedError(
                f"numpy autodiff does not support 'np.{ufunc.__name__}'"
            )
        return self._trace.record(ufunc, inputs, kwargs)

    def __array_function__(self, function, types, args, kwargs):
        implementation = _COMPOSITES.get(function)
        if implementation is not None:
            return implementation(*args, **kwargs)
        if function in _VJPS or function in _NONDIFFERENTIABLE:
            return self._trace.record(function, args, kwargs)
        raise NotImplementedError(
            f"numpy autodiff does not support "
            f"'{function.__module__}.{function.__name__}'"
        )

    def __array__(self, dtype=None):
        raise TypeError(
            "A traced value cannot be converted to a NumPy array or stored in "
            "one. Use NumPy functions on it instead, e.g. np.stack or np.where."
        )

    def _concrete(self, convert):
        # Python control flow may depend on values when the function is traced
        # on every call, but converting floats to Python numbers would drop
        # their derivative.
        if np.issubdtype(self.dtype, np.inexact) and convert is not bool:
            raise TypeError(
                "A traced value cannot be converted to a Python number "
                "without losing its derivative. Keep it as an array."
            )
        if not self._trace.concrete:
            raise TracingError("Python control flow depends on a traced value")
        return convert(self.value)

    def __bool__(self):
        return self._concrete(bool)

    def __float__(self):
        return self._concrete(float)

    def __int__(self):
        return self._concrete(int)

    def __complex__(self):
        return self._concrete(complex)

    def __index__(self):
        return self._concrete(int)

    def item(self):
        return self._concrete(lambda value: value.item())

    def __setitem__(self, key, value):
        raise TypeError(
            "Traced arrays cannot be updated in place; build a new array "
            "instead, e.g. with np.where"
        )

    shape = property(lambda self: self.value.shape)
    dtype = property(lambda self: self.value.dtype)
    ndim = property(lambda self: self.value.ndim)
    size = property(lambda self: self.value.size)
    T = property(lambda self: np.transpose(self))
    real = property(lambda self: np.real(self))
    imag = property(lambda self: np.imag(self))

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index):
        for value in _leaves(index, Tracer):
            if value.dtype == bool and not self._trace.concrete:
                raise TracingError("Boolean masks give value-dependent shapes")
        return _getitem(self, index)

    def __neg__(self):
        return np.negative(self)

    def __pos__(self):
        return self

    def __abs__(self):
        return np.absolute(self)

    def __invert__(self):
        return np.invert(self)

    __hash__ = None

    def astype(self, dtype):
        return _astype(self, dtype)

    def conj(self):
        return np.conjugate(self)

    conjugate = conj

    def copy(self):
        return self

    def dot(self, other):
        return np.dot(self, other)

    def flatten(self):
        return np.ravel(self)

    def ravel(self):
        return np.ravel(self)

    def reshape(self, *shape, **kwargs):
        return np.reshape(
            self, shape[0] if len(shape) == 1 else shape, **kwargs
        )

    def transpose(self, *axes):
        if len(axes) == 1:
            axes = axes[0]
        return np.transpose(self, axes or None)

    def sum(self, *args, **kwargs):
        return np.sum(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        return np.mean(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        return np.max(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        return np.min(self, *args, **kwargs)

    def squeeze(self, *args, **kwargs):
        return np.squeeze(self, *args, **kwargs)

    def swapaxes(self, axis1, axis2):
        return np.swapaxes(self, axis1, axis2)

    def diagonal(self, *args, **kwargs):
        return np.diagonal(self, *args, **kwargs)

    def trace(self, *args, **kwargs):
        return np.trace(self, *args, **kwargs)

    def clip(self, *args, **kwargs):
        return np.clip(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        return np.argmax(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        return np.argmin(self, *args, **kwargs)

    def any(self, *args, **kwargs):
        return np.any(self, *args, **kwargs)

    def all(self, *args, **kwargs):
        return np.all(self, *args, **kwargs)


def _operator(ufunc, reflected=False):
    if reflected:
        return lambda self, other: ufunc(other, self)
    return lambda self, other: ufunc(self, other)


for _name, _ufunc in [
    ("add", np.add),
    ("sub", np.subtract),
    ("mul", np.multiply),
    ("truediv", np.true_divide),
    ("matmul", np.matmul),
    ("pow", np.power),
    ("and", np.bitwise_and),
    ("or", np.bitwise_or),
]:
    setattr(Tracer, f"__{_name}__", _operator(_ufunc))
    setattr(Tracer, f"__r{_name}__", _operator(_ufunc, reflected=True))
for _name, _ufunc in [
    ("lt", np.less),
    ("le", np.less_equal),
    ("gt", np.greater),
    ("ge", np.greater_equal),
    ("eq", np.equal),
    ("ne", np.not_equal),
]:
    setattr(Tracer, f"__{_name}__", _operator(_ufunc))


def _primitive(function):
    """Makes ``function`` record itself when called on tracers."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        for value in _leaves((args, *kwargs.values()), Tracer):
            return value._trace.record(wrapped, args, kwargs)
        return function(*args, **kwargs)

    return wrapped


##############################################################################
# Backward pass
##############################################################################


def _shape(value):
    return getattr(value, "shape", ())


def _dtype(value):
    return value.dtype if hasattr(value, "dtype") else np.asarray(value).dtype


def _is_complex(value):
    return np.issubdtype(_dtype(value), np.complexfloating)


def _conj(value):
    return np.conjugate(value) if _is_complex(value) else value


def _adjoint(value):
    return _conj(np.swapaxes(value, -1, -2))


def _fit(cotangent, value):
    """Sums out broadcast dimensions and casts to the dtype of ``value``."""
    shape = _shape(value)
    if _shape(cotangent) != shape:
        num_extra = len(_shape(cotangent)) - len(shape)
        if num_extra > 0:
            cotangent = np.sum(cotangent, axis=tuple(range(num_extra)))
        axes = tuple(
            axis
            for axis, size in enumerate(shape)
            if size == 1 and _shape(cotangent)[axis] != 1
        )
        if axes:
            cotangent = np.sum(cotangent, axis=axes, keepdims=True)
        if _shape(cotangent) != shape:
            cotangent = np.broadcast_to(cotangent, shape)
    dtype = _dtype(value)
    if _dtype(cotangent) != dtype:
        if _is_complex(cotangent) and not _is_complex(value):
            cotangent = np.real(cotangent)
        if _dtype(cotangent) != dtype:
            cotangent = _astype(cotangent, dtype)
    return cotangent


def _backprop(program, args):
    """Gradients of the scalar output of ``program`` w.r.t. its inputs."""
    env, (result,) = program.evaluate(*args)
    (output,) = program.outputs
    cotangents = {}
    if isinstance(output, _Var):
        if _is_complex(result) or _shape(result) != ():
            raise ValueError(
                "Cost function must return a real scalar, got a value of "
                f"shape {_shape(result)} and dtype {_dtype(result)}"
            )
        cotangents[output.id] = np.ones((), _dtype(result))

    def lookup(var):
        return env[var.id]

    for function, eq_args, eq_kwargs, out in reversed(program.equations):
        g = cotangents.pop(out.id, None)
        rule = _VJPS.get(function)
        if g is None or rule is None:
            continue
        values, kwargs = _map_call(lookup, eq_args, eq_kwargs, _Var)
        for i, arg in enumerate(eq_args):
            if not isinstance(arg, _Var) or not np.issubdtype(
                _dtype(values[i]), np.inexact
            ):
                continue
            cotangent = rule(i, g, env[out.id], *values, **kwargs)
            if cotangent is None:
                continue
            cotangent = _fit(cotangent, values[i])
            if arg.id in cotangents:
                cotangent = cotangents[arg.id] + cotangent
            cotangents[arg.id] = cotangent
    return [
        (
            cotangents[var.id]
            if var.id in cotangents
            else np.zeros(_shape(arg), _dtype(arg))
        )
        for var, arg in zip(program.inputs, args)
    ]


def _gradient_program(function, args, concrete):
    program = trace(function, args, concrete)
    return trace(lambda *xs: _backprop(program, xs), args, concrete)


def _hessian_program(function, args, concrete):
    n = len(args) // 2
    gradient = _gradient_program(function, args[:n], concrete)

    def directional_derivative(*args):
        _, gradients = gradient.evaluate(*args[:n])
        return sum(
            np.real(np.sum(_conj(g) * v)) for g, v in zip(gradients, args[n:])
        )

    program = trace(directional_derivative, args, concrete)
    return trace(lambda *xs: _backprop(program, xs)[:n], args, concrete)


class _Operator:
    """Builds, compiles and caches a program per argument shapes and dtypes.

    If the function cannot be traced once and replayed, because Python control
    flow depends on traced values, the program is rebuilt on every call.
    """

    def __init__(self, function, build):
        self._function = function
        self._build = build
        self._programs = {}
        self._retrace = False

    def __call__(self, *args):
        if self._retrace:
            program = self._build(self._function, args, concrete=True)
            return program.evaluate(*args)[1]
        key = tuple((np.shape(arg), _dtype(arg)) for arg in args)
        program = self._programs.get(key)
        if program is None:
            try:
                program = self._build(self._function, args, concrete=False)
            except TracingError:
                self._retrace = True
                return self(*args)
            program = self._programs[key] = program.compile()
        return program(*args)


def gradient(function):
    """Returns a function computing the gradients of ``function``."""
    return _Operator(function, _gradient_program)


def hessian_vector_product(function):
    """Returns a function computing Hessian-vector products of ``function``.

    The returned function takes the points followed by the vectors.
    """
    return _Operator(function, _hessian_program)


##############################################################################
# Differentiation rules
##############################################################################

# Maps a function to a rule ``rule(i, g, ans, *args, **kwargs)`` giving the
# cotangent of the ``i``-th positional argument, given the cotangent ``g`` of
# the output ``ans``. Rules only call NumPy functions so that they can be
# traced and differentiated again.
_VJPS = {}
# Functions which are recorded but have no derivative.
_NONDIFFERENTIABLE = set()
# Functions implemented in terms of other functions when called on tracers.
_COMPOSITES = {}


def _defvjp(function, *rules):
    def rule(i, g, ans, *args, **kwargs):
        if i >= len(rules):
            raise NotImplementedError(
                f"numpy autodiff cannot differentiate argument {i} of "
                f"'{function.__name__}'"
            )
        return rules[i](g, ans, *args, **kwargs)

    _VJPS[function] = rule


def _composite(*functions):
    def register(implementation):
        for function in functions:
            _COMPOSITES[function] = implementation
        return implementation

    return register


def _normalize_axes(axis, ndim):
    if axis is None:
        return tuple(range(ndim))
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    return tuple(a % ndim for a in axis)


def _unreduce(g, shape, axis):
    """Broadcasts the result of a reduction back to the input shape."""
    axes = _normalize_axes(axis, len(shape))
    kept_shape = tuple(1 if i in axes else n for i, n in enumerate(shape))
    return np.broadcast_to(np.reshape(g, kept_shape), shape)


# Elementwise functions.
_defvjp(np.add, lambda g, ans, x, y: g, lambda g, ans, x, y: g)
_defvjp(np.subtract, lambda g, ans, x, y: g, lambda g, ans, x, y: -g)
_defvjp(
    np.multiply,
    lambda g, ans, x, y: g * _conj(y),
    lambda g, ans, x, y: _conj(x) * g,
)
_defvjp(
    np.true_divide,
    lambda g, ans, x, y: g / _conj(y),
    lambda g, ans, x, y: -g * _conj(ans / y),
)
_defvjp(np.negative, lambda g, ans, x: -g)
_defvjp(np.conjugate, lambda g, ans, x: np.conjugate(g))
_defvjp(np.exp, lambda g, ans, x: g * _conj(ans))
_defvjp(np.expm1, lambda g, ans, x: g * _conj(ans + 1))
_defvjp(np.log, lambda g, ans, x: g / _conj(x))
_defvjp(np.log1p, lambda g, ans, x: g / _conj(x + 1))
_defvjp(np.log2, lambda g, ans, x: g / _conj(x * np.log(2)))
_defvjp(np.log10, lambda g, ans, x: g / _conj(x * np.log(10)))
_defvjp(np.sqrt, lambda g, ans, x: g / (2 * _conj(ans)))
_defvjp(np.square, lambda g, ans, x: 2 * g * _conj(x))
_defvjp(np.reciprocal, lambda g, ans, x: -g * _conj(ans * ans))
_defvjp(np.sin, lambda g, ans, x: g * _conj(np.cos(x)))
_defvjp(np.cos, lambda g, ans, x: -g * _conj(np.sin(x)))
_defvjp(np.tan, lambda g, ans, x: g * _conj(1 + ans * ans))
_defvjp(np.sinh, lambda g, ans, x: g * _conj(np.cosh(x)))
_defvjp(np.cosh, lambda g, ans, x: g * _conj(np.sinh(x)))
_defvjp(np.tanh, lambda g, ans, x: g * _conj(1 - ans * ans))
_defvjp(np.arcsin, lambda g, ans, x: g / _conj(np.sqrt(1 - x * x)))
_defvjp(np.arccos, lambda g, ans, x: -g / _conj(np.sqrt(1 - x * x)))
_defvjp(np.arctan, lambda g, ans, x: g / _conj(1 + x * x))
_defvjp(
    np.absolute,
    lambda g, ans, x: g * x / ans if _is_complex(x) else g * np.sign(x),
)
_defvjp(
    np.maximum,
    lambda g, ans, x, y: g * (x >= y),
    lambda g, ans, x, y: g * (x < y),
)
_defvjp(
    np.minimum,
    lambda g, ans, x, y: g * (x <= y),
    lambda g, ans, x, y: g * (x > y),
)


def _power_vjp_base(g, ans, x, y):
    if isinstance(y, (int, float)) and y == 2:
        return 2 * g * _conj(x)
    return g * _conj(y * x ** (y - 1))


_defvjp(
    np.power,
    _power_vjp_base,
    lambda g, ans, x, y: g * _conj(np.log(x) * ans),
)
_defvjp(np.real, lambda g, ans, x: g)
_defvjp(np.imag, lambda g, ans, x: 1j * g)
_defvjp(
    np.where,
    lambda g, ans, c, x, y: None,
    lambda g, ans, c, x, y: np.where(c, g, 0),
    lambda g, ans, c, x, y: np.where(c, 0, g),
)


def _clip_vjp(g, ans, x, a_min, a_max, **kwargs):
    mask = np.ones(_shape(x), bool)
    if a_min is not None:
        mask = mask & (x >= a_min)
    if a_max is not None:
        mask = mask & (x <= a_max)
    return g * mask


_defvjp(np.clip, _clip_vjp)

_NONDIFFERENTIABLE.update(
    [
        np.equal,
        np.not_equal,
        np.less,
        np.less_equal,
        np.greater,
        np.greater_equal,
        np.logical_and,
        np.logical_or,
        np.logical_not,
        np.bitwise_and,
        np.bitwise_or,
        np.invert,
        np.sign,
        np.floor,
        np.ceil,
        np.rint,
        np.isfinite,
        np.isinf,
        np.isnan,
        np.argmax,
        np.argmin,
        np.any,
        np.all,
    ]
)

# Reductions.
_defvjp(
    np.sum,
    lambda g, ans, x, axis=None, dtype=None, out=None, keepdims=False: (
        _unreduce(g, _shape(x), axis)
    ),
)


def _extremum_vjp(g, ans, x, axis=None, out=None, keepdims=False, **kwargs):
    shape = _shape(x)
    mask = x == _unreduce(ans, shape, axis)
    count = np.sum(mask, axis=_normalize_axes(axis, len(shape)), keepdims=True)
    return _unreduce(g, shape, axis) * mask / count


for _function in (np.max, np.amax, np.min, np.amin):
    _defvjp(_function, _extremum_vjp)


@_composite(np.mean)
def _mean(x, axis=None, dtype=None, out=None, keepdims=False):
    shape = _shape(x)
    count = int(np.prod([shape[a] for a in _normalize_axes(axis, len(shape))]))
    return np.sum(x, axis=axis, keepdims=keepdims) / count


# Shapes.
_defvjp(
    np.reshape,
    lambda g, ans, x, *args, **kwargs: np.reshape(g, _shape(x)),
)
for _function in (np.ravel, np.squeeze, np.expand_dims):
    _defvjp(
        _function, lambda g, ans, x, *args, **kwargs: np.reshape(g, _shape(x))
    )
_defvjp(np.broadcast_to, lambda g, ans, x, *args, **kwargs: g)


def _transpose_vjp(g, ans, x, axes=None):
    if axes is None:
        return np.transpose(g)
    return np.transpose(g, tuple(np.argsort(_normalize_axes(axes, len(axes)))))


_defvjp(np.transpose, _transpose_vjp)
_defvjp(np.swapaxes, lambda g, ans, x, a, b: np.swapaxes(g, a, b))


@_composite(np.moveaxis)
def _moveaxis(x, source, destination):
    ndim = len(_shape(x))
    source = _normalize_axes(source, ndim)
    destination = _normalize_axes(destination, ndim)
    order = [axis for axis in range(ndim) if axis not in source]
    for d, s in sorted(zip(destination, source)):
        order.insert(d, s)
    return np.transpose(x, order)


@_primitive
def _astype(x, dtype):
    return x.astype(dtype)


_defvjp(_astype, lambda g, ans, x, dtype: g)


@_primitive
def _getitem(x, index):
    return x[index]


@_primitive
def _scatter_add(g, shape, index):
    out = np.zeros(shape, g.dtype)
    if any(
        isinstance(i, (np.ndarray, list))
        for i in (index if isinstance(index, tuple) else (index,))
    ):
        np.add.at(out, index, g)
    else:
        out[index] = g
    return out


def _getitem_vjp(g, ans, x, index):
    return _scatter_add(g, _shape(x), index)


_defvjp(_getitem, _getitem_vjp)
_defvjp(_scatter_add, lambda g, ans, x, shape, index: _getitem(g, index))


@_primitive
def _concatenate(*arrays, axis=0):
    return np.concatenate(arrays, axis=axis)


def _concatenate_vjp(i, g, ans, *arrays, axis=0):
    axis %= len(_shape(ans))
    start = sum(_shape(a)[axis] for a in arrays[:i])
    index = (slice(None),) * axis + (
        slice(start, start + _shape(arrays[i])[axis]),
    )
    return _getitem(g, index)


_VJPS[_concatenate] = _concatenate_vjp


@_composite(np.concatenate)
def _concatenate_composite(arrays, axis=0, out=None, **kwargs):
    if axis is None:
        return _concatenate(*(np.ravel(a) for a in arrays))
    return _concatenate(*arrays, axis=axis)


@_composite(np.stack)
def _stack(arrays, axis=0, out=None):
    return _concatenate(*(np.expand_dims(a, axis) for a in arrays), axis=axis)


@_composite(np.append)
def _append(arr, values, axis=None):
    if axis is None:
        return _concatenate(np.ravel(arr), np.ravel(values))
    return _concatenate(arr, values, axis=axis)


@_composite(np.shape)
def _shape_composite(x):
    return _shape(x)


@_composite(np.ndim)
def _ndim(x):
    return len(_shape(x))


@_composite(np.size)
def _size(x, axis=None):
    return x.size if axis is None else _shape(x)[axis]


@_composite(np.iscomplexobj)
def _iscomplexobj(x):
    return _is_complex(x)


@_composite(np.isrealobj)
def _isrealobj(x):
    return not _is_complex(x)


@_composite(np.copy)
def _copy(x, *args, **kwargs):
    return x


def _register_constant_like(function, fill):
    def implementation(x, dtype=None, order="K", subok=True, shape=None):
        return np.full(
            _shape(x) if shape is None else shape,
            fill,
            _dtype(x) if dtype is None else dtype,
        )

    _COMPOSITES[function] = implementation


_register_constant_like(np.zeros_like, 0)
_register_constant_like(np.ones_like, 1)
_register_constant_like(np.empty_like, 0)


# Diagonals and triangles.
@_primitive
def _undiagonal(g, shape, offset, axis1, axis2):
    out = np.zeros(shape, g.dtype)
    view = np.moveaxis(out, (axis1, axis2), (-2, -1))
    n = np.shape(g)[-1]
    rows = np.arange(n) + max(-offset, 0)
    columns = np.arange(n) + max(offset, 0)
    view[..., rows, columns] = g
    return out


_defvjp(
    np.diagonal,
    lambda g, ans, x, offset=0, axis1=0, axis2=1: _undiagonal(
        g, _shape(x), offset, axis1, axis2
    ),
)
_defvjp(
    _undiagonal,
    lambda g, ans, x, shape, offset, axis1, axis2: np.diagonal(
        g, offset, axis1, axis2
    ),
)


@_primitive
def _diag(v, k):
    return np.diag(v, k)


_defvjp(_diag, lambda g, ans, v, k: np.diagonal(g, k))


@_composite(np.diag)
def _diag_composite(v, k=0):
    if len(_shape(v)) == 2:
        return np.diagonal(v, k)
    return _diag(v, k)


@_composite(np.trace)
def _trace(x, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    return np.sum(np.diagonal(x, offset, axis1, axis2), axis=-1)


_defvjp(np.triu, lambda g, ans, x, k=0: np.triu(g, k))
_defvjp(np.tril, lambda g, ans, x, k=0: np.tril(g, k))


# Products.
def _matmul_vjp(i, g, ans, a, b):
    a_is_vector, b_is_vector = len(_shape(a)) == 1, len(_shape(b)) == 1
    if a_is_vector and b_is_vector:
        return g * _conj(b) if i == 0 else g * _conj(a)
    if a_is_vector:
        a, g = np.expand_dims(a, 0), np.expand_dims(g, -2)
    if b_is_vector:
        b, g = np.expand_dims(b, -1), np.expand_dims(g, -1)
    if i == 0:
        cotangent = g @ _adjoint(b)
        return np.squeeze(cotangent, -2) if a_is_vector else cotangent
    cotangent = _adjoint(a) @ g
    return np.squeeze(cotangent, -1) if b_is_vector else cotangent


_VJPS[np.matmul] = _matmul_vjp


@_composite(np.tensordot)
def _tensordot(a, b, axes=2):
    ndim_a, ndim_b = len(_shape(a)), len(_shape(b))
    if isinstance(axes, int):
        axes_a = list(range(ndim_a - axes, ndim_a))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = (
            [
                ax % ndim
                for ax in (axs if isinstance(axs, (list, tuple)) else [axs])
            ]
            for axs, ndim in zip(axes, (ndim_a, ndim_b))
        )
    free_a = [k for k in range(ndim_a) if k not in axes_a]
    free_b = [k for k in range(ndim_b) if k not in axes_b]
    shape_a, shape_b = _shape(a), _shape(b)
    size = int(np.prod([shape_a[k] for k in axes_a]))
    a = np.reshape(np.transpose(a, free_a + axes_a), (-1, size))
    b = np.reshape(np.transpose(b, axes_b + free_b), (size, -1))
    return np.reshape(
        a @ b, [shape_a[k] for k in free_a] + [shape_b[k] for k in free_b]
    )


@_composite(np.dot)
def _dot(a, b, out=None):
    if len(_shape(a)) == 0 or len(_shape(b)) == 0:
        return a * b
    if len(_shape(b)) <= 2:
        return a @ b
    return _tensordot(a, b, ([-1], [-2]))


@_composite(np.inner)
def _inner(a, b):
    if len(_shape(a)) == 0 or len(_shape(b)) == 0:
        return a * b
    return _tensordot(a, b, ([-1], [-1]))


@_composite(np.vdot)
def _vdot(a, b):
    return np.sum(_conj(np.ravel(a)) * np.ravel(b))


@_composite(np.outer)
def _outer(a, b, out=None):
    return np.expand_dims(np.ravel(a), 1) * np.expand_dims(np.ravel(b), 0)


# Linear algebra.
@_composite(np.linalg.norm)
def _norm(x, ord=None, axis=None, keepdims=False):
    if axis is None:
        ndim = len(_shape(x))
    else:
        ndim = 1 if isinstance(axis, int) else len(axis)
    if ord in (None, "fro") or (ord == 2 and ndim == 1):
        squares = np.real(x * np.conjugate(x)) if _is_complex(x) else x * x
        return np.sqrt(np.sum(squares, axis=axis, keepdims=keepdims))
    if ndim == 1 and ord == 1:
        return np.sum(np.abs(x), axis=axis, keepdims=keepdims)
    if ndim == 1 and ord == np.inf:
        return np.max(np.abs(x), axis=axis, keepdims=keepdims)
    raise NotImplementedError(
        f"numpy autodiff does not support np.linalg.norm with ord={ord!r}"
    )


def _solve_vjp(i, g, ans, a, b):
    is_vector = len(_shape(b)) == len(_shape(a)) - 1
    g_b = np.linalg.solve(_adjoint(a), g)
    if i == 1:
        return g_b
    if is_vector:
        g_b, ans = np.expand_dims(g_b, -1), np.expand_dims(ans, -1)
    return -(g_b @ _adjoint(ans))


_VJPS[np.linalg.solve] = _solve_vjp
_defvjp(np.linalg.inv, lambda g, ans, a: -(_adjoint(ans) @ g @ _adjoint(ans)))
_defvjp(
    np.linalg.det,
    lambda g, ans, a: (
        np.expand_dims(g * _conj(ans), (-2, -1)) * _adjoint(np.linalg.inv(a))
    ),
)


@_primitive
def _slogdet_sign(a):
    return np.linalg.slogdet(a)[0]


@_primitive
def _logabsdet(a):
    return np.linalg.slogdet(a)[1]


_defvjp(
    _logabsdet,
    lambda g, ans, a: np.expand_dims(g, (-2, -1)) * _adjoint(np.linalg.inv(a)),
)


@_composite(np.linalg.slogdet)
def _slogdet(a):
    return _slogdet_sign(a), _logabsdet(a)
