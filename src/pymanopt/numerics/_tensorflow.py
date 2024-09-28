from typing import Any, Sequence

import numpy as np
import numpy.testing as np_testing
import packaging.version as pv
import scipy
import scipy.linalg
import tensorflow as tf

from pymanopt.numerics.array_t import array_t
from pymanopt.numerics.core import NumericsBackend


class TensorflowNumericsBackend(NumericsBackend):
    _dtype: tf.DType

    def __init__(self, dtype=tf.float64):
        self._dtype = dtype

    @property
    def dtype(self) -> tf.DType:
        return self._dtype

    def __repr__(self):
        return f"TensorflowNumericsBackend(dtype={self.dtype})"

    ##############################################################################
    # Numerics functions
    ##############################################################################
