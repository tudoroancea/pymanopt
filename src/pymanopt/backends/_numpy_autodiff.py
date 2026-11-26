"""Reverse-mode automatic differentiation of plain NumPy code.

A cost function is run on :class:`Tracer` objects, which intercept NumPy calls
through ``__array_ufunc__`` and ``__array_function__`` and record them as a
straight-line :class:`Program`. The gradient is obtained by tracing the
backward pass of that program, so it is a program too, and Hessian-vector
products differentiate the gradient in the same way. These programs are
compiled to Python functions that only call NumPy.

The cost function is traced on every call, like autograd does, so it may read
values other than its arguments and branch on the values of its arguments. The
arrays it reads become inputs of the derivative programs, and a compiled
program is reused as long as the trace records the same calls on arrays of the
same shapes and dtypes and the same other values, which is cheaper than
differentiating the trace again.

Gradients with respect to complex arguments are ``df/dx + i df/dy``, the
gradient for the real inner product ``Re <a, b>``, which is the convention
used by pymanopt.
"""

import functools
import operator

import numpy as np

__all__ = ["gradient", "hessian_vector_product"]


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
    def __init__(self):
        self.equations = []
        self.values = []

    def new_tracer(self, value):
        if not hasattr(value, "dtype"):
            value = np.asarray(value)
        tracer = Tracer(self, _Var(len(self.values)), value)
        self.values.append(value)
        return tracer

    def _value(self, tracer):
        if tracer._trace is not self:
            raise ValueError("Values from different traces were mixed")
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


def trace(function, args):
    """Records the NumPy calls ``function`` makes on ``args``.

    Returns the program and the values of its variables.
    """
    recording = _Trace()
    tracers = [recording.new_tracer(arg) for arg in args]
    result = function(*tracers)
    results = result if isinstance(result, (list, tuple)) else [result]
    outputs = [r.var if isinstance(r, Tracer) else r for r in results]
    program = Program([t.var for t in tracers], recording.equations, outputs)
    return program, recording.values


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
        # Python control flow may depend on values, but converting floats to
        # Python numbers would drop their derivative.
        if np.issubdtype(self.dtype, np.inexact) and convert is not bool:
            raise TypeError(
                "A traced value cannot be converted to a Python number "
                "without losing its derivative. Keep it as an array."
            )
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
        # Traced slice bounds are converted to integers, like Python does.
        return _getitem(self, _map(_slice_of_integers, index, slice))

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

    def clip(self, min=None, max=None):
        return np.clip(self, min, max)

    def argmax(self, *args, **kwargs):
        return np.argmax(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        return np.argmin(self, *args, **kwargs)

    def any(self, *args, **kwargs):
        return np.any(self, *args, **kwargs)

    def all(self, *args, **kwargs):
        return np.all(self, *args, **kwargs)


def _slice_of_integers(s):
    return slice(
        *(
            operator.index(b) if isinstance(b, Tracer) else b
            for b in (s.start, s.stop, s.step)
        )
    )


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


def _pow(self, exponent):
    # Like NumPy arrays, use the faster np.square for x ** 2.
    if type(exponent) is int and exponent == 2:
        return np.square(self)
    return np.power(self, exponent)


Tracer.__pow__ = _pow


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


def _backprop(program, args, n):
    """Gradients of the scalar output of ``program`` w.r.t. its first inputs."""
    env, (result,) = program.evaluate(*args)
    (output,) = program.outputs
    if _is_complex(result) or _shape(result) != ():
        raise ValueError(
            "Cost function must return a real scalar, got a value of "
            f"shape {_shape(result)} and dtype {_dtype(result)}"
        )
    cotangents = {}
    if isinstance(output, _Var):
        cotangents[output.id] = np.ones((), _dtype(result))
    # Variables depending on the first n inputs.
    active = {var.id for var in program.inputs[:n]}
    for _, eq_args, eq_kwargs, out in program.equations:
        if any(
            var.id in active
            for var in _leaves((eq_args, *eq_kwargs.values()), _Var)
        ):
            active.add(out.id)

    def lookup(var):
        return env[var.id]

    for function, eq_args, eq_kwargs, out in reversed(program.equations):
        g = cotangents.pop(out.id, None)
        rule = _VJPS.get(function)
        if g is None or rule is None:
            continue
        if any(
            var.id in active for var in _leaves(list(eq_kwargs.values()), _Var)
        ):
            raise NotImplementedError(
                f"numpy autodiff cannot differentiate '{function.__name__}' "
                "with traced keyword arguments; pass them positionally"
            )
        values, kwargs = _map_call(lookup, eq_args, eq_kwargs, _Var)
        for i, arg in enumerate(eq_args):
            if (
                not isinstance(arg, _Var)
                or arg.id not in active
                or not np.issubdtype(_dtype(values[i]), np.inexact)
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
        for var, arg in zip(program.inputs[:n], args)
    ]


def _gradient_program(program, args, constants):
    n = len(args)
    return trace(lambda *xs: _backprop(program, xs, n), [*args, *constants])


def _hessian_program(program, args, constants):
    n = len(args) // 2
    gradient, _ = _gradient_program(program, args[:n], constants)

    def directional_derivative(*xs):
        _, gradients = gradient.evaluate(*xs[:n], *xs[2 * n :])
        return sum(
            np.real(np.sum(_conj(g) * v))
            for g, v in zip(gradients, xs[n : 2 * n])
        )

    program, _ = trace(directional_derivative, [*args, *constants])
    return trace(lambda *xs: _backprop(program, xs, n), [*args, *constants])


def _lift_constants(program, num_vars):
    """Turns the arrays ``program`` reads into inputs following its own.

    Derivative programs then read these arrays on every call instead of
    keeping values computed from them when they were built.
    """
    constants = []

    def lift(array):
        constants.append(array)
        return _Var(num_vars + len(constants) - 1)

    equations = [
        (function, *_map_call(lift, args, kwargs, np.ndarray), out)
        for function, args, kwargs, out in program.equations
    ]
    inputs = [_Var(num_vars + i) for i in range(len(constants))]
    program = Program(program.inputs + inputs, equations, program.outputs)
    return program, constants


def _fingerprint(program, values):
    """Describes what the derivatives of ``program`` depend on.

    Returns a key comparable with ``==``.
    """

    def key(value):
        if isinstance(value, _Var):
            return value.id
        if isinstance(value, np.ndarray):
            return np.ndarray, value.shape, value.dtype, value.tobytes()
        if isinstance(value, (float, complex, np.inexact)):
            # Unlike ==, this tells signed zeros apart and matches NaNs.
            return type(value), repr(value)
        if type(value) in (tuple, list):
            return type(value), tuple(map(key, value))
        if type(value) is slice:
            return slice, key((value.start, value.stop, value.step))
        return type(value), value

    equations = tuple(
        (function, key(args), key(list(kwargs.items())))
        for function, args, kwargs, _ in program.equations
    )
    shapes = tuple((value.shape, value.dtype) for value in values)
    return equations, key(program.outputs), shapes


class _Operator:
    """Traces ``function`` on every call and differentiates the trace.

    The derivative program is compiled when the next trace has the same
    fingerprint, and then reused as long as the fingerprint does not change.
    """

    def __init__(self, function, build, num_points):
        self._function = function
        self._build = build
        self._num_points = num_points
        self._key = self._program = self._compiled = None

    def __call__(self, *args):
        program, values = trace(self._function, self._num_points(args))
        program, constants = _lift_constants(program, len(values))
        key = _fingerprint(program, [*values, *constants])
        key += tuple((np.shape(arg), _dtype(arg)) for arg in args)
        if key == self._key:
            if self._compiled is None:
                self._compiled = self._program.compile()
            return self._compiled(*args, *constants)
        self._program, values = self._build(program, args, constants)
        self._compiled = None
        self._key = key
        return tuple(
            _output(values[o.id]) if isinstance(o, _Var) else np.array(o)
            for o in self._program.outputs
        )


def gradient(function):
    """Returns a function computing the gradients of ``function``."""
    return _Operator(function, _gradient_program, lambda args: args)


def hessian_vector_product(function):
    """Returns a function computing Hessian-vector products of ``function``.

    The returned function takes the points followed by the vectors.
    """
    return _Operator(
        function, _hessian_program, lambda args: args[: len(args) // 2]
    )


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
    if axis is None or ndim == 0:
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
    lambda g, ans, x: (
        g * x / np.where(ans == 0, 1, ans)
        if _is_complex(x)
        else g * np.sign(x)
    ),
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


_defvjp(
    np.power,
    lambda g, ans, x, y: g * _conj(y * x ** (y - 1)),
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


def _clip_vjp(g, ans, x, a_min=None, a_max=None, min=None, max=None):
    a_min = min if a_min is None else a_min
    a_max = max if a_max is None else a_max
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
def _sum_vjp(
    g,
    ans,
    x,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    g = _unreduce(g, _shape(x), axis)
    return g if where is True else np.where(where, g, 0)


_defvjp(np.sum, _sum_vjp)


def _extremum_vjp(
    g, ans, x, axis=None, out=None, keepdims=False, initial=None, where=True
):
    shape = _shape(x)
    mask = (x == _unreduce(ans, shape, axis)) & where
    count = np.sum(mask, axis=_normalize_axes(axis, len(shape)), keepdims=True)
    return _unreduce(g, shape, axis) * mask / count


for _function in (np.max, np.amax, np.min, np.amin):
    _defvjp(_function, _extremum_vjp)


@_composite(np.mean)
def _mean(x, axis=None, dtype=None, keepdims=False):
    shape = _shape(x)
    count = int(np.prod([shape[a] for a in _normalize_axes(axis, len(shape))]))
    return np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims) / count


# Shapes.
def _reshape_vjp(g, ans, x, *args, **kwargs):
    order = kwargs.get("order", next((a for a in args if type(a) is str), "C"))
    if order not in ("C", "F"):
        raise NotImplementedError(
            f"numpy autodiff does not support order={order!r}"
        )
    return np.reshape(g, _shape(x), order=order)


for _function in (np.reshape, np.ravel, np.squeeze, np.expand_dims):
    _defvjp(_function, _reshape_vjp)
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
        isinstance(i, (np.ndarray, list, tuple))
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
def _concatenate_composite(arrays, axis=0):
    if axis is None:
        return _concatenate(*(np.ravel(a) for a in arrays))
    return _concatenate(*arrays, axis=axis)


@_composite(np.stack)
def _stack(arrays, axis=0):
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
def _trace(x, offset=0, axis1=0, axis2=1):
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
def _dot(a, b):
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
def _outer(a, b):
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
    is_vector = len(_shape(b)) == 1
    if is_vector:
        g, ans = np.expand_dims(g, -1), np.expand_dims(ans, -1)
    g_b = np.linalg.solve(_adjoint(a), g)
    if i == 1:
        return np.squeeze(g_b, -1) if is_vector else g_b
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
    _slogdet_sign,
    lambda g, ans, a: (
        np.expand_dims(-1j * np.imag(np.conjugate(g) * ans), (-2, -1))
        * _adjoint(np.linalg.inv(a))
        if _is_complex(a)
        else None
    ),
)
_defvjp(
    _logabsdet,
    lambda g, ans, a: np.expand_dims(g, (-2, -1)) * _adjoint(np.linalg.inv(a)),
)


@_composite(np.linalg.slogdet)
def _slogdet(a):
    return _slogdet_sign(a), _logabsdet(a)
