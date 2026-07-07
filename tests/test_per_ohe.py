#
#//  test_per_ohe.py
#//  heteroknockoffpy
#//
#//  Tests for prismWImportancesPerOHE: every OHE dummy column is treated as its
#//  own independent variable (no group-level aggregation), unlike prismWImportances.
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import importance


def _make_mixed_synthetic(n: int = 200, seed: int = 0):
    """2 numeric vars + 1 categorical (K=3, so 2 dummy columns after drop_first).

    p_vars = 3 (grouped view), p_ohe = 4 (2 numeric + 2 dummy columns per side).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    cat_idx = rng.integers(0, 3, n)
    cat = pl.Series([str(v) for v in cat_idx]).cast(pl.Categorical)

    xk0 = rng.standard_normal(n)
    xk1 = rng.standard_normal(n)
    catk_idx = rng.integers(0, 3, n)
    catk = pl.Series([str(v) for v in catk_idx]).cast(pl.Categorical)

    X = pl.DataFrame({"x0": x0, "x1": x1, "xc": cat})
    Xk = pl.DataFrame({"x0": xk0, "x1": xk1, "xc": catk})
    y = pl.Series("y", x0 + rng.standard_normal(n) * 0.5)
    return X, Xk, y


P_OHE = 4  # 2 numeric + 2 dummy columns (K=3, drop_first) per side


@pytest.mark.parametrize("model_type", ["mlp", "pairwise"])
def test_per_ohe_shape_is_ohe_width_not_variable_count(model_type):
    X, Xk, y = _make_mixed_synthetic()
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismWImportancesPerOHE(
        X=X, Xk=Xk, y=y,
        layers=[16, 8],
        model_type=model_type,
        lambda_path=lambda_path,
        epochs=3,
    )
    assert imp.shape == (2 * P_OHE,), f"Expected shape ({2 * P_OHE},), got {imp.shape}"
    assert np.all(np.isfinite(imp))
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["mlp", "pairwise"])
def test_per_ohe_differs_from_grouped_shape(model_type):
    """Sanity check: the grouped prismWImportances returns one score per variable (3),
    while prismWImportancesPerOHE returns one per OHE column (4) -- confirming the two
    are not accidentally identical for this mixed fixture."""
    X, Xk, y = _make_mixed_synthetic()
    lambda_path = np.logspace(0, -1, 5)

    grouped = importance.prismWImportances(
        X=X, Xk=Xk, y=y,
        layers=[16, 8],
        model_type=model_type,
        lambda_path=lambda_path,
        epochs=3,
    )
    per_ohe = importance.prismWImportancesPerOHE(
        X=X, Xk=Xk, y=y,
        layers=[16, 8],
        model_type=model_type,
        lambda_path=lambda_path,
        epochs=3,
    )
    assert grouped.shape == (6,)  # 2 * 3 variables
    assert per_ohe.shape == (8,)  # 2 * 4 OHE columns
    assert grouped.shape != per_ohe.shape


@pytest.mark.parametrize("model_type", ["additive"])
def test_per_ohe_rejects_unsupported_model_types(model_type):
    X, Xk, y = _make_mixed_synthetic()
    with pytest.raises(ValueError):
        importance.prismWImportancesPerOHE(
            X=X, Xk=Xk, y=y,
            layers=[16, 8],
            model_type=model_type,
            epochs=3,
        )


def test_per_ohe_all_numeric_matches_grouped_shape():
    """With no categorical columns, per-OHE and grouped shapes coincide (every
    variable is already its own singleton group)."""
    n, p = 200, 5
    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    cols = [f"x{i}" for i in range(p)]
    X = pl.DataFrame(dict(zip(cols, X_np.T)))
    Xk = pl.DataFrame(dict(zip(cols, Xk_np.T)))
    y = pl.Series("y", X_np[:, 0] + rng.standard_normal(n) * 0.5)

    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismWImportancesPerOHE(
        X=X, Xk=Xk, y=y,
        layers=[16, 8],
        model_type="mlp",
        lambda_path=lambda_path,
        epochs=3,
    )
    assert imp.shape == (2 * p,)
