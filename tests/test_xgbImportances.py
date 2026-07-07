#
#//  tests/test_xgbImportances.py
#//  heteroknockoffpy
#//
#//  Tests for heteroknockoffpy.xgbImportances: score_importances, prism_importances,
#//  and shap_importances, called directly (bypassing the torch/xgboost isolation
#//  wrapper in importance.py -- that wrapper is covered separately in
#//  test_processIsolation.py).
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import xgbImportances


def _make_mixed_synthetic(n: int = 200, seed: int = 0):
    """2 numeric vars + 1 categorical (K=3) per side."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    cat = pl.Series([str(v) for v in rng.integers(0, 3, n)]).cast(pl.Categorical)

    xk0 = rng.standard_normal(n)
    xk1 = rng.standard_normal(n)
    catk = pl.Series([str(v) for v in rng.integers(0, 3, n)]).cast(pl.Categorical)

    X = pl.DataFrame({"x0": x0, "x1": x1, "xc": cat})
    Xk = pl.DataFrame({"x0": xk0, "x1": xk1, "xc": catk})
    return X, Xk, x0, rng


# ── score_importances ────────────────────────────────────────────────────────

def test_score_importances_continuous_shape_and_finite():
    X, Xk, x0, rng = _make_mixed_synthetic()
    y = pl.Series("y", x0 + rng.standard_normal(len(x0)) * 0.5)

    imp = xgbImportances.score_importances(X=X, Xk=Xk, y=y)
    assert imp.shape == (6,)  # 2 numeric + 1 categorical, per side
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


@pytest.mark.parametrize("importance_type", ["weight", "gain", "cover", "total_gain", "total_cover"])
def test_score_importances_importance_type_variants(importance_type):
    X, Xk, x0, rng = _make_mixed_synthetic()
    y = pl.Series("y", x0 + rng.standard_normal(len(x0)) * 0.5)

    imp = xgbImportances.score_importances(X=X, Xk=Xk, y=y, importance_type=importance_type)
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))


# ── prism_importances ────────────────────────────────────────────────────────

def test_prism_importances_continuous():
    X, Xk, x0, rng = _make_mixed_synthetic()
    y = pl.Series("y", x0 + rng.standard_normal(len(x0)) * 0.5)

    imp = xgbImportances.prism_importances(X=X, Xk=Xk, y=y, outcome_type="continuous")
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


def test_prism_importances_count():
    X, Xk, x0, rng = _make_mixed_synthetic()
    y = pl.Series("y", rng.poisson(np.exp(0.3 * x0)).astype(np.int64))

    imp = xgbImportances.prism_importances(X=X, Xk=Xk, y=y, outcome_type="count")
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


def test_prism_importances_categorical():
    X, Xk, x0, rng = _make_mixed_synthetic()
    cat_idx = rng.integers(0, 3, len(x0))
    y = pl.Series(
        "y", [str(v) for v in (x0 > 0).astype(int) + np.clip(cat_idx, 0, 1)]
    ).cast(pl.Categorical)

    imp = xgbImportances.prism_importances(X=X, Xk=Xk, y=y, outcome_type="categorical")
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


def test_prism_importances_all_numeric():
    n, p = 200, 3
    rng = np.random.default_rng(1)
    X_np = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    cols = [f"x{i}" for i in range(p)]
    X = pl.DataFrame(dict(zip(cols, X_np.T)))
    Xk = pl.DataFrame(dict(zip(cols, Xk_np.T)))
    y = pl.Series("y", X_np[:, 0] + rng.standard_normal(n) * 0.5)

    imp = xgbImportances.prism_importances(X=X, Xk=Xk, y=y, outcome_type="continuous")
    assert imp.shape == (2 * p,)
    assert np.all(np.isfinite(imp))


# ── shap_importances ─────────────────────────────────────────────────────────

def test_shap_importances_continuous():
    pytest.importorskip("shap")
    X, Xk, x0, rng = _make_mixed_synthetic()
    y = pl.Series("y", x0 + rng.standard_normal(len(x0)) * 0.5)

    imp = xgbImportances.shap_importances(X=X, Xk=Xk, y=y, outcome_type="continuous")
    assert imp.shape == (6,)
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)
