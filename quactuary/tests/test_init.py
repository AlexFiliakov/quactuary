import builtins
import importlib
import sys
from importlib.metadata import version as dist_version

import pytest


def reload_quactuary():
    # Remove quactuary modules to force fresh import
    to_remove = [m for m in sys.modules if m ==
                 "quactuary" or m.startswith("quactuary.")]
    for m in to_remove:
        sys.modules.pop(m, None)
    return importlib.import_module("quactuary")


def test_version_import_success():
    # Ensure we get version from _version.py
    qa = reload_quactuary()
    assert hasattr(qa, "__version__"), "__version__ not set"
    from quactuary._version import version as expected
    assert qa.__version__ == expected


def test_version_import_fallback(monkeypatch):
    # Simulate ImportError when importing quactuary._version
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "quactuary._version" or (name == "quactuary" and "_version" in fromlist):
            raise ImportError("Simulated missing _version module")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Patch metadata.version to return fallback
    try:
        import importlib.metadata as metadata
    except ImportError:
        import importlib_metadata as metadata
    monkeypatch.setattr(metadata, "version", lambda pkg: "fallback.version")

    qa = reload_quactuary()
    assert qa.__version__ == dist_version("quactuary")


def test_get_and_set_backend_classical():
    qa = reload_quactuary()
    # Ensure get_backend returns a manager and is singleton
    bm1 = qa.get_backend()
    bm2 = qa.get_backend()
    assert bm1 is bm2
    # Set a dummy classical backend
    dummy = object()
    ret = qa.set_backend("classical", backend=dummy)
    assert ret is dummy
    assert qa.get_backend().get_backend() is dummy


def test_set_backend_invalid_mode():
    qa = reload_quactuary()
    with pytest.raises(ValueError):
        qa.set_backend("unsupported_mode")
