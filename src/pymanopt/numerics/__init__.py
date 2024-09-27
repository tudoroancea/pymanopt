import importlib

from .array_t import AVAILABLE_NUMERICS_BACKENDS, SUPPORTED_NUMERICS_BACKENDS, array_t
from .core import NumericsBackend

for backend in AVAILABLE_NUMERICS_BACKENDS:
    importlib.import_module(f"._{backend}")
