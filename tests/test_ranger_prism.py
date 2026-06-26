#
#//  tests/test_ranger_prism.py
#//  heteroknockoffpy
#//
#//  Tests for rangerPrismImportances / stat_forest_prism_*.
#//
#//  3 × 3 grid: predictor type (numeric, categorical, mixed) ×
#//              outcome type (continuous, count, categorical).
#//
#//  Each test uses a strong signal so the signal variable's PRISM importance
#//  reliably dominates all noise variables.  Knockoffs are i.i.d. random
#//  draws from the same marginal distribution, so knockoff importances stay small.
#

import numpy as np
import polars as pl

from heteroknockoffpy import importance


# ── helpers ───────────────────────────────────────────────────────────────────

def _cat(name: str, values: list[str]) -> pl.Series:
    return pl.Series(name=name, values=values, dtype=pl.Utf8).cast(pl.Categorical)


def _rand_cat(rng: np.random.Generator, levels: list[str], n: int, name: str) -> pl.Series:
    return _cat(name, rng.choice(levels, size=n).tolist())


def _W(imp: np.ndarray) -> np.ndarray:
    return importance.wFromImportances(imp)


# ── numeric predictors ────────────────────────────────────────────────────────

def test_ranger_prism_numeric_continuous():
    """Numeric X, continuous outcome: x0 is the only signal."""
    rng = np.random.default_rng(20)
    n, p = 500, 5
    X_np  = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    X  = pl.DataFrame({f"x{i}": X_np[:, i]  for i in range(p)})
    Xk = pl.DataFrame({f"x{i}": Xk_np[:, i] for i in range(p)})
    y  = pl.Series("y", 20.0 * X_np[:, 0] + rng.standard_normal(n) * 0.1)

    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)

    assert imp.shape == (2 * p,), f"Expected shape ({2*p},), got {imp.shape}"
    assert W[0] > 0,          f"Signal W stat should be positive, got {W[0]:.3f}"
    assert W[0] > max(W[1:]), f"Signal W[0]={W[0]:.3f} should exceed all noise W stats {W[1:].tolist()}"


def test_ranger_prism_numeric_count():
    """Numeric X, count outcome: Poisson rate driven by x0."""
    rng = np.random.default_rng(21)
    n, p = 500, 5
    X_np  = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    X  = pl.DataFrame({f"x{i}": X_np[:, i]  for i in range(p)})
    Xk = pl.DataFrame({f"x{i}": Xk_np[:, i] for i in range(p)})
    counts = np.maximum(0, np.round(10.0 + 20.0 * X_np[:, 0])).astype(np.int64)
    y = pl.Series("y", counts)

    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y, outcome_type='count')
    W   = _W(imp)

    assert imp.shape == (2 * p,)
    assert W[0] > 0,          f"Signal W stat should be positive, got {W[0]:.3f}"
    assert W[0] > max(W[1:]), f"Signal W[0]={W[0]:.3f} should exceed all noise W stats {W[1:].tolist()}"


def test_ranger_prism_numeric_categorical():
    """Numeric X, categorical outcome: 3-class y binned from x0."""
    rng = np.random.default_rng(22)
    n, p = 500, 5
    X_np  = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    X  = pl.DataFrame({f"x{i}": X_np[:, i]  for i in range(p)})
    Xk = pl.DataFrame({f"x{i}": Xk_np[:, i] for i in range(p)})
    q      = np.quantile(X_np[:, 0], [1/3, 2/3])
    labels = np.where(X_np[:, 0] < q[0], "A", np.where(X_np[:, 0] < q[1], "B", "C"))
    y = pl.Series("y", labels, dtype=pl.Utf8).cast(pl.Categorical)

    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)

    assert imp.shape == (2 * p,)
    assert W[0] > 0,          f"Signal W stat should be positive, got {W[0]:.3f}"
    assert W[0] > max(W[1:]), f"Signal W[0]={W[0]:.3f} should exceed all noise W stats {W[1:].tolist()}"


# ── categorical predictors ─────────────────────────────────────────────────────

def test_ranger_prism_categorical_continuous():
    """Categorical X, continuous outcome: a (cat0 vs rest) is the only signal."""
    rng = np.random.default_rng(23)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]
    levels_c = ["p", "q", "r", "s"]

    a_vals = rng.choice(levels_a, size=n)
    X  = pl.DataFrame({
        "a": _cat("a", a_vals.tolist()),
        "b": _rand_cat(rng, levels_b, n, "b"),
        "c": _rand_cat(rng, levels_c, n, "c"),
    })
    Xk = pl.DataFrame({
        "a": _rand_cat(rng, levels_a, n, "a"),
        "b": _rand_cat(rng, levels_b, n, "b"),
        "c": _rand_cat(rng, levels_c, n, "c"),
    })
    y = pl.Series("y", 20.0 * (a_vals == "cat0").astype(float) + rng.standard_normal(n) * 0.1)

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Signal a W stat should be positive, got {W[0]:.3f}"
    assert W[0] > W[1], f"Signal a ({W[0]:.3f}) should exceed noise b ({W[1]:.3f})"
    assert W[0] > W[2], f"Signal a ({W[0]:.3f}) should exceed noise c ({W[2]:.3f})"


def test_ranger_prism_categorical_count():
    """Categorical X, count outcome: Poisson rate determined by level of a."""
    rng = np.random.default_rng(24)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]

    a_vals = rng.choice(levels_a, size=n)
    X  = pl.DataFrame({
        "a": _cat("a", a_vals.tolist()),
        "b": _rand_cat(rng, levels_b, n, "b"),
    })
    Xk = pl.DataFrame({
        "a": _rand_cat(rng, levels_a, n, "a"),
        "b": _rand_cat(rng, levels_b, n, "b"),
    })
    rate_map = {"cat0": 20, "cat1": 5, "cat2": 1}
    rates  = np.array([rate_map[v] for v in a_vals], dtype=np.float64)
    counts = rng.poisson(rates).astype(np.int64)
    y = pl.Series("y", counts)

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y, outcome_type='count')
    W   = _W(imp)

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Signal a W stat should be positive, got {W[0]:.3f}"
    assert W[0] > W[1], f"Signal a ({W[0]:.3f}) should exceed noise b ({W[1]:.3f})"


def test_ranger_prism_categorical_categorical():
    """Categorical X, categorical outcome: class == level of a."""
    rng = np.random.default_rng(25)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]

    a_vals = rng.choice(levels_a, size=n)
    X  = pl.DataFrame({
        "a": _cat("a", a_vals.tolist()),
        "b": _rand_cat(rng, levels_b, n, "b"),
    })
    Xk = pl.DataFrame({
        "a": _rand_cat(rng, levels_a, n, "a"),
        "b": _rand_cat(rng, levels_b, n, "b"),
    })
    y = pl.Series("y", a_vals, dtype=pl.Utf8).cast(pl.Categorical)

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Signal a W stat should be positive, got {W[0]:.3f}"
    assert W[0] > W[1], f"Signal a ({W[0]:.3f}) should exceed noise b ({W[1]:.3f})"


# ── mixed (numeric + categorical) predictors ───────────────────────────────────

def test_ranger_prism_mixed_continuous():
    """Mixed X, continuous outcome: x0 (numeric) and a (categorical) are both signals."""
    rng = np.random.default_rng(26)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]

    x0     = rng.standard_normal(n)
    x1     = rng.standard_normal(n)
    a_vals = rng.choice(levels_a, size=n)

    X  = pl.DataFrame({
        "x0": x0,
        "x1": x1,
        "a":  _cat("a", a_vals.tolist()),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    Xk = pl.DataFrame({
        "x0": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "a":  _rand_cat(rng, levels_a, n, "a"),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    y = pl.Series("y",
        20.0 * x0 + 20.0 * (a_vals == "cat0").astype(float) + rng.standard_normal(n) * 0.1
    )

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)   # [x0, x1, a, b]

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Numeric signal x0 W stat should be positive, got {W[0]:.3f}"
    assert W[2] > 0,    f"Categorical signal a W stat should be positive, got {W[2]:.3f}"
    assert W[0] > W[1], f"Numeric signal x0 ({W[0]:.3f}) should exceed noise x1 ({W[1]:.3f})"
    assert W[2] > W[3], f"Categorical signal a ({W[2]:.3f}) should exceed noise b ({W[3]:.3f})"


def test_ranger_prism_mixed_count():
    """Mixed X, count outcome: Poisson rate driven by x0 (numeric) and a (categorical)."""
    rng = np.random.default_rng(27)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]

    x0     = rng.standard_normal(n)
    x1     = rng.standard_normal(n)
    a_vals = rng.choice(levels_a, size=n)

    X  = pl.DataFrame({
        "x0": x0,
        "x1": x1,
        "a":  _cat("a", a_vals.tolist()),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    Xk = pl.DataFrame({
        "x0": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "a":  _rand_cat(rng, levels_a, n, "a"),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    cat_contrib = np.where(a_vals == "cat0", 15.0, 0.0)
    rates  = np.maximum(1.0, 10.0 + 20.0 * x0 + cat_contrib)
    counts = rng.poisson(rates).astype(np.int64)
    y = pl.Series("y", counts)

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y, outcome_type='count')
    W   = _W(imp)   # [x0, x1, a, b]

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Numeric signal x0 W stat should be positive, got {W[0]:.3f}"
    assert W[2] > 0,    f"Categorical signal a W stat should be positive, got {W[2]:.3f}"
    assert W[0] > W[1], f"Numeric signal x0 ({W[0]:.3f}) should exceed noise x1 ({W[1]:.3f})"
    assert W[2] > W[3], f"Categorical signal a ({W[2]:.3f}) should exceed noise b ({W[3]:.3f})"


def test_ranger_prism_mixed_categorical():
    """Mixed X, categorical outcome: x0 and a jointly determine 3-class y."""
    rng = np.random.default_rng(28)
    n = 500
    levels_a = ["cat0", "cat1", "cat2"]
    levels_b = ["x", "y", "z"]

    x0     = rng.standard_normal(n)
    x1     = rng.standard_normal(n)
    a_vals = rng.choice(levels_a, size=n)

    X  = pl.DataFrame({
        "x0": x0,
        "x1": x1,
        "a":  _cat("a", a_vals.tolist()),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    Xk = pl.DataFrame({
        "x0": rng.standard_normal(n),
        "x1": rng.standard_normal(n),
        "a":  _rand_cat(rng, levels_a, n, "a"),
        "b":  _rand_cat(rng, levels_b, n, "b"),
    })
    raw = np.where(a_vals == "cat0", "C", np.where(x0 > 0, "B", "A"))
    y = pl.Series("y", raw, dtype=pl.Utf8).cast(pl.Categorical)

    p   = X.width
    imp = importance.rangerPrismImportances(X=X, Xk=Xk, y=y)
    W   = _W(imp)   # [x0, x1, a, b]

    assert imp.shape == (2 * p,)
    assert W[0] > 0,    f"Numeric signal x0 W stat should be positive, got {W[0]:.3f}"
    assert W[2] > 0,    f"Categorical signal a W stat should be positive, got {W[2]:.3f}"
    assert W[0] > W[1], f"Numeric signal x0 ({W[0]:.3f}) should exceed noise x1 ({W[1]:.3f})"
    assert W[2] > W[3], f"Categorical signal a ({W[2]:.3f}) should exceed noise b ({W[3]:.3f})"
