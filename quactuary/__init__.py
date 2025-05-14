try:
    from ._version import version as __version__
except ImportError:
    try:
        # use installed distribution metadata
        import importlib.metadata as _im
    except ImportError:
        import importlib_metadata as _im  # python <3.8 fallback
    __version__ = _im.version("quactuary")

from .backend import get_backend, set_backend
