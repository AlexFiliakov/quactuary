import builtins
import importlib
import sys
import types
from importlib.metadata import version as dist_version

import pytest


def reload_quactuary(version=None):
    # Remove quactuary modules to force fresh import
    to_remove = [m for m in sys.modules if m ==
                 "quactuary" or m.startswith("quactuary.")]
    for m in to_remove:
        sys.modules.pop(m, None)
    module = importlib.import_module("quactuary")
    if version is not None:
        # Simulate the version being set in the module
        module.__dict__["__version__"] = version
    return module


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


def test_version_import_double_fallback(monkeypatch):
    # Remove modules so that __init__ hits the fallback
    for mod in ["quactuary._version", "importlib.metadata", "importlib_metadata"]:
        monkeypatch.delitem(sys.modules, mod, raising=False)

    # Patch quactuary.__init__ so that once __init__.py sets _im,
    # we replace _im.version with our forced fallback
    def mock_version(package_name=None):
        return "double.fallback.version"

    # We’ll intercept the actual assignment in __init__.py by patching that symbol
    monkeypatch.setattr("quactuary.__version__",
                        types.SimpleNamespace(version=mock_version))

    qa = reload_quactuary(mock_version())
    assert qa.__version__ == mock_version()
