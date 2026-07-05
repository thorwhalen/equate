"""Tests for equate._dependencies (optional-dependency machinery) and that the core
`import equate` stays numpy/scipy-light (heavy deps imported lazily)."""

import subprocess
import sys

import pytest

from equate import require, have, MissingDependencyError


def test_require_present_returns_module():
    assert require("numpy").__name__ == "numpy"
    # dotted submodule paths work too
    assert require("scipy.optimize").__name__ == "scipy.optimize"


def test_require_missing_raises_actionable_error_with_extra():
    with pytest.raises(MissingDependencyError) as ei:
        require("definitely_not_installed_xyz", extra="fuzzy", purpose="testing")
    msg = str(ei.value)
    assert "definitely_not_installed_xyz" in msg
    assert "equate[fuzzy]" in msg  # names the extra to install
    assert "testing" in msg  # weaves in the purpose


def test_require_missing_without_extra_hints_plain_pip():
    with pytest.raises(MissingDependencyError) as ei:
        require("definitely_not_installed_xyz")
    assert "pip install definitely_not_installed_xyz" in str(ei.value)


def test_missing_dependency_is_an_importerror():
    # so existing `except ImportError` handlers still catch it
    assert issubclass(MissingDependencyError, ImportError)


def test_have_probes_without_importing():
    assert have("numpy") is True
    assert have("definitely_not_installed_xyz") is False
    # a probe must not import the module as a side effect
    assert "definitely_not_installed_xyz" not in sys.modules


def test_import_equate_stays_light():
    """`import equate` must not eagerly pull heavy/optional deps."""
    code = (
        "import equate, sys;"
        "heavy=[m for m in ('grub','sklearn','networkx','pandas','rapidfuzz') "
        "if m in sys.modules];"
        "print(heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"import equate pulled heavy modules: {r.stdout.strip()}"


def test_default_match_path_still_works():
    """Back-compat: the default TF-IDF+cosine+Hungarian path still resolves (lazily)."""
    from equate import match_keys_to_values

    keys = ["apple pie", "banana split"]
    values = ["american pie", "banana republic"]
    result = dict(match_keys_to_values(keys, values))
    assert set(result.keys()) == set(keys)
    assert len(set(result.values())) == 2  # a 1:1 matching
