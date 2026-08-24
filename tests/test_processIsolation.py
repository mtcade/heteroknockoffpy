#
#//  tests/test_processIsolation.py
#//  heteroknockoffpy
#//
#//  Tests for heteroknockoffpy._processIsolation.run_isolated_if_loaded, plus
#//  end-to-end coverage of the actual scenario it exists for: calling every
#//  xgb*/prism* importance.py function (and knockoff.get_knockoffs's torch_GAN
#//  path) in both possible orders (xgboost-family first vs. torch-family
#//  first), confirming the "second" family's calls are transparently isolated
#//  into a subprocess rather than crashing.
#//
#//  Because pytest runs the whole suite in one process, and other test files
#//  import torch at module scope, the "does the isolation decision come out
#//  right for a genuinely fresh process" behavior can't be tested reliably as
#//  plain in-process assertions here -- it needs real subprocesses with clean
#//  sys.modules. Each order below is a small worker script run via
#//  subprocess.run, asserting inside the worker and printing a distinct
#//  success marker this test checks for.
#//
import importlib.util
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import _processIsolation


# ── direct unit tests of run_isolated_if_loaded ──────────────────────────────
#
# Target real, already-installed, pure functions in heteroknockoffpy.utilities
# rather than adding test-only helpers to production code, or relying on
# `tests` package importability inside a freshly spawned subprocess (which
# isn't guaranteed the way an installed package's is).

def _sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": pl.Series(["x", "y", "x", "y"]).cast(pl.Categorical),
    })


def _patch_guarded_modules(monkeypatch, own_native: str, other_native: str) -> None:
    # heteroknockoffpy.utilities has no real native dependency, so it isn't a
    # genuine entry in _GUARDED_MODULES -- stand one up for the duration of the
    # test, paired with a second fake entry whose native module we control by
    # choosing whether its name is already in sys.modules.
    monkeypatch.setattr(
        _processIsolation,
        "_GUARDED_MODULES",
        {
            "heteroknockoffpy.utilities": own_native,
            "this_module_definitely_does_not_exist_zzz.other_target": other_native,
        },
    )


def test_run_isolated_if_loaded_in_process_when_no_conflict(monkeypatch):
    # Neither entry's native module is loaded, so the call stays in-process.
    _patch_guarded_modules(
        monkeypatch,
        own_native="own_native_definitely_does_not_exist_zzz",
        other_native="other_native_definitely_does_not_exist_zzz",
    )
    df = _sample_df()
    result = _processIsolation.run_isolated_if_loaded(
        "heteroknockoffpy.utilities",
        "get_ohe_np",
        X=df, drop_first=True,
    )
    from heteroknockoffpy.utilities import get_ohe_np
    np.testing.assert_array_equal(result, get_ohe_np(df, drop_first=True))


def test_run_isolated_if_loaded_isolated_when_conflict_present(monkeypatch):
    # 'sys' is always already imported, so mapping the *other* entry to it
    # forces the subprocess branch (the target's own native module is excluded
    # from the conflict check, so it alone wouldn't trigger isolation).
    _patch_guarded_modules(
        monkeypatch,
        own_native="own_native_definitely_does_not_exist_zzz",
        other_native="sys",
    )
    df = _sample_df()
    result = _processIsolation.run_isolated_if_loaded(
        "heteroknockoffpy.utilities",
        "get_ohe_np",
        X=df, drop_first=True,
    )
    from heteroknockoffpy.utilities import get_ohe_np
    np.testing.assert_array_equal(result, get_ohe_np(df, drop_first=True))


def test_run_isolated_if_loaded_propagates_exceptions_in_process(monkeypatch):
    _patch_guarded_modules(
        monkeypatch,
        own_native="own_native_definitely_does_not_exist_zzz",
        other_native="other_native_definitely_does_not_exist_zzz",
    )
    with pytest.raises(AttributeError):
        _processIsolation.run_isolated_if_loaded(
            "heteroknockoffpy.utilities",
            "this_function_does_not_exist",
        )


def test_run_isolated_if_loaded_propagates_exceptions_isolated(monkeypatch):
    _patch_guarded_modules(
        monkeypatch,
        own_native="own_native_definitely_does_not_exist_zzz",
        other_native="sys",
    )
    with pytest.raises(AttributeError):
        _processIsolation.run_isolated_if_loaded(
            "heteroknockoffpy.utilities",
            "this_function_does_not_exist",
        )


# ── cross-order end-to-end: every xgb*/prism* call, both orders ─────────────
#
# Worker scripts are built by concatenating flush-left (no shared leading
# indentation) string blocks -- keep every block below flush-left so simple
# string concatenation produces valid, correctly-indented top-level Python.

_HAS_SHAP = importlib.util.find_spec("shap") is not None

_COMMON_SETUP = """
import sys
import numpy as np
import polars as pl
from heteroknockoffpy import importance, knockoff

rng = np.random.default_rng(0)
n = 120
x0 = rng.standard_normal(n)
X = pl.DataFrame({'x0': x0})
Xk = pl.DataFrame({'x0': rng.standard_normal(n)})
y = pl.Series('y', x0 + rng.standard_normal(n) * 0.5)
"""

_XGB_CALLS_TEMPLATE = """
imp = importance.xgbImportances(X=X, Xk=Xk, y=y)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "xgbImportances leaked {other_family}"

imp = importance.xgbPrismImportances(X=X, Xk=Xk, y=y)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "xgbPrismImportances leaked {other_family}"
"""

_XGB_SHAP_CALL_TEMPLATE = """
imp = importance.xgbShapImportances(X=X, Xk=Xk, y=y)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "xgbShapImportances leaked {other_family}"
"""

_PRISM_CALLS_TEMPLATE = """
imp = importance.grip2Importances(X=X, Xk=Xk, y=y, layers=[8], epochs=3)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "grip2Importances leaked {other_family}"

imp = importance.torchPrismImportances(X=X, Xk=Xk, y=y, layers=[8], epochs=3)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "torchPrismImportances leaked {other_family}"

g, w = importance.prismGrip2Importances(X=X, Xk=Xk, y=y, layers=[8], epochs=3)
assert g.shape == (2,) and w.shape == (2,), (g, w)
assert {other_family!r} not in sys.modules, "prismGrip2Importances leaked {other_family}"

grads = importance.torchPrismLocalGradients(X=X, Xk=Xk, y=y, layers=[8], epochs=3)
assert grads.shape[1] == 1, grads.shape
assert {other_family!r} not in sys.modules, "torchPrismLocalGradients leaked {other_family}"

imp = importance.grip2ImportancesPerOHE(X=X, Xk=Xk, y=y, layers=[8], epochs=3)
assert imp.shape == (2,) and np.all(np.isfinite(imp)), imp
assert {other_family!r} not in sys.modules, "grip2ImportancesPerOHE leaked {other_family}"
"""

_GAN_TORCH_CALL_TEMPLATE = """
Xk_gan = knockoff.get_knockoffs(
    X, method="torch_GAN", rng=rng, categorical_method="ohe", niter=3,
)
assert Xk_gan.shape == (n, 1), Xk_gan.shape
assert {other_family!r} not in sys.modules, "torch_GAN leaked {other_family}"
"""


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def _run_worker(tmp_path, name: str, script_body: str):
    marker = f"{name}_OK"
    script_path = tmp_path / f"{name}.py"
    # multiprocessing's spawn context re-imports __main__ to bootstrap each
    # child process, so the driver code below must be guarded -- same
    # requirement documented in _processIsolation.run_isolated_if_loaded's
    # docstring and the README for any top-level script using these functions.
    full_body = script_body + f'\nprint("{marker}")\n'
    script_path.write_text(
        'if __name__ == "__main__":\n' + _indent(full_body)
    )
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"worker {name} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert marker in result.stdout, result.stdout


def test_xgboost_family_first_then_torch_family(tmp_path):
    """xgboost-family calls stay in-process; every torch-family call after must
    isolate itself (torch never appears in this process's sys.modules)."""
    body = _COMMON_SETUP + _XGB_CALLS_TEMPLATE.format(other_family="torch")
    if _HAS_SHAP:
        body += _XGB_SHAP_CALL_TEMPLATE.format(other_family="torch")
    body += _PRISM_CALLS_TEMPLATE.format(other_family="torch")
    body += _GAN_TORCH_CALL_TEMPLATE.format(other_family="torch")
    _run_worker(tmp_path, "xgb_first", body)


def test_torch_family_first_then_xgboost_family(tmp_path):
    """torch-family calls stay in-process; every xgboost-family call after must
    isolate itself (xgboost never appears in this process's sys.modules)."""
    body = _COMMON_SETUP + _PRISM_CALLS_TEMPLATE.format(other_family="xgboost")
    body += _XGB_CALLS_TEMPLATE.format(other_family="xgboost")
    if _HAS_SHAP:
        body += _XGB_SHAP_CALL_TEMPLATE.format(other_family="xgboost")
    _run_worker(tmp_path, "torch_first", body)
