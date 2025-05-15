import os
import runpy

import pytest
import setuptools


def test_setup_py_invokes_setup_correctly(monkeypatch):
    # Capture arguments passed to setuptools.setup
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)
    monkeypatch.setattr(setuptools, 'setup', fake_setup)

    # Run setup.py from project root
    root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir))
    setup_path = os.path.join(root, 'setup.py')
    runpy.run_path(setup_path, run_name='__main__')

    # Basic sanity checks
    assert captured, "setup() was not called"
    # use_scm_version
    assert 'use_scm_version' in captured
    assert captured['use_scm_version'] == {'write_to': 'quactuary/_version.py'}
    # setup_requires
    assert captured.get('setup_requires') == ['setuptools_scm']
    # packages
    pkgs = captured.get('packages')
    assert isinstance(pkgs, list)
    assert 'quactuary' in pkgs, "Root package 'quactuary' not found in packages"
    # package_dir
    assert captured.get('package_dir') == {'': 'quactuary'}
    # description
    assert captured.get('description') == 'Quantum-powered actuarial tools'
    # install_requires should match requirements file
    reqs_path = os.path.join(root, r'quactuary', 'requirements.txt')
    with open(reqs_path) as f:
        expected_reqs = f.read().splitlines()
    assert captured.get('install_requires') == expected_reqs
