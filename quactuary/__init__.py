try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
from .backend import get_backend, set_backend
