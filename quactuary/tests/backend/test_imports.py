import builtins
import sys
import types

import pytest

from quactuary import backend

# Tests for import-related error handling in set_backend (quantum mode)


def test_set_backend_missing_qiskit(monkeypatch):
    """
    Simulate missing Qiskit module and expect ImportError indicating Qiskit is required.
    """
    # Monkeypatch builtins.__import__ to raise for 'qiskit'
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'qiskit' or name.startswith('qiskit.'):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(ImportError) as excinfo:
        backend.set_backend('quantum')
    assert "Qiskit is required" in str(excinfo.value)


def test_set_backend_missing_packaging(monkeypatch):
    """
    Simulate missing 'packaging' package and expect ImportError prompting to install packaging.
    """
    # Ensure qiskit is importable with correct version
    fake_qiskit = types.SimpleNamespace(__version__='1.4.2')
    monkeypatch.setitem(sys.modules, 'qiskit', fake_qiskit)

    # Monkeypatch builtins.__import__ to raise for 'packaging'
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'packaging' or name.startswith('packaging.'):
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(ImportError) as excinfo:
        backend.set_backend('quantum')
    assert "Please install the 'packaging' package" in str(excinfo.value)


def test_set_backend_qiskit_wrong_version(monkeypatch):
    """
    Simulate a different Qiskit version and expect ImportError about exact version requirement.
    """
    # Fake qiskit with wrong version
    fake_qiskit = types.SimpleNamespace(__version__='0.23.0')
    monkeypatch.setitem(sys.modules, 'qiskit', fake_qiskit)

    # packaging is available; no need to monkeypatch
    with pytest.raises(ImportError) as excinfo:
        backend.set_backend('quantum')
    assert "Quantum mode requires Qiskit version 1.4.2 exactly" in str(
        excinfo.value)
