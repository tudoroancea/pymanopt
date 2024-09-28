from typing import Any, Sequence

import numpy as np
import numpy.testing as np_testing
import packaging.version as pv
import scipy
import scipy.linalg
import jax.numpy as jnp

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class JaxNumericsBackend(NumericsBackend):
    _dtype: jnp.dtype

    def __init__(self, dtype=jnp.float64):
        self._dtype = dtype

    @property
    def dtype(self) -> jnp.dtype:
        return self._dtype

    def __repr__(self):
        return f"JaxNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################
