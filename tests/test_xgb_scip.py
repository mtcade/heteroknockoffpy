#
#//  tests/test_xgb_scip.py
#//  heteroknockoffpy
#//
#//  Regression tests for xgbScip._categories_for: with polars' Categorical
#//  dtype now backed by one process-global dictionary (pl.Categories),
#//  `.cat.get_categories()` returns the union of categories across every
#//  categorical column/array built so far in the process, not just the
#//  queried column's own values. That broke get_knockoffs_SCIP/
#//  get_knockoffs_with_Xk_numeric on any X with 2+ categorical columns (or
#//  even one, once a prior test/column had polluted the shared dictionary)
#//  with xgboost raising "Invalid classes inferred from unique values of
#//  `y`." on the non-contiguous class codes that fell out of it.
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy.xgbScip import _categories_for, get_knockoffs_SCIP


def test_categories_for_returns_only_this_columns_values():
    # Two categorical columns with disjoint value sets sharing the same
    # process-global pl.Categories dictionary -- the exact shape that
    # produced non-contiguous class codes before the fix.
    df = pl.DataFrame({
        "a": pl.Series(["x", "y", "z"]).cast(pl.Categorical),
        "b": pl.Series(["p", "q", "p"]).cast(pl.Categorical),
    })
    assert _categories_for(df, "a") == ["x", "y", "z"]
    assert _categories_for(df, "b") == ["p", "q"]


def test_categories_for_sorted_and_deduplicated():
    df = pl.DataFrame({
        "a": pl.Series(["z", "x", "z", "y", "x"]).cast(pl.Categorical),
    })
    assert _categories_for(df, "a") == ["x", "y", "z"]


def test_categories_for_drops_nulls():
    df = pl.DataFrame({
        "a": pl.Series(["x", None, "y"]).cast(pl.Categorical),
    })
    assert _categories_for(df, "a") == ["x", "y"]


def test_categories_for_unaffected_by_prior_columns_in_process():
    # Simulates the global-dictionary pollution scenario across two
    # separately-built DataFrames (mirrors what happens across sequential
    # SCIP columns/prior test runs in the same process).
    _first = pl.DataFrame({"other": pl.Series(["m", "n", "o", "p"]).cast(pl.Categorical)})
    second = pl.DataFrame({"a": pl.Series(["x", "y"]).cast(pl.Categorical)})
    assert _categories_for(second, "a") == ["x", "y"]


def test_xgb_SCIP_multi_categorical_knockoffs_stay_within_own_categories():
    # Regression test for the original crash: get_knockoffs_SCIP on X with
    # 3+ categorical columns of differing cardinality used to raise
    # ValueError("Invalid classes inferred from unique values of `y`.")
    # inside xgboost's classifier fit.
    n = 300
    rng = np.random.default_rng(0)
    X = pl.DataFrame({
        "diet_type": rng.choice(["veg", "non-veg", "balanced", "missing"], size=n),
        "stress_level": rng.choice(["low", "medium", "high"], size=n),
        "sleep_quality": rng.choice(["good", "average", "poor", "missing"], size=n),
        "sleep_duration": rng.normal(7, 1, size=n),
    }).with_columns(
        pl.col("diet_type").cast(pl.Categorical),
        pl.col("stress_level").cast(pl.Categorical),
        pl.col("sleep_quality").cast(pl.Categorical),
    )

    Xk = get_knockoffs_SCIP(X, rng=rng, residuals_method="normal", n_estimators=20, max_depth=3)

    assert Xk.schema == X.schema
    assert len(Xk) == len(X)
    for col in ("diet_type", "stress_level", "sleep_quality"):
        # every knockoff value must come from that column's *own* levels,
        # not from another categorical column's levels leaking in via the
        # shared dictionary
        assert set(Xk[col].unique().to_list()) <= set(_categories_for(X, col))


@pytest.mark.parametrize("residuals_method", ["normal", "permute"])
def test_xgb_SCIP_two_categorical_columns_same_cardinality(residuals_method):
    # Even equal-cardinality categorical columns must not silently swap
    # each other's categories via the shared dictionary.
    n = 200
    rng = np.random.default_rng(1)
    X = pl.DataFrame({
        "a": rng.choice(["red", "green", "blue"], size=n),
        "b": rng.choice(["cat", "dog", "bird"], size=n),
    }).with_columns(
        pl.col("a").cast(pl.Categorical),
        pl.col("b").cast(pl.Categorical),
    )

    Xk = get_knockoffs_SCIP(X, rng=rng, residuals_method=residuals_method, n_estimators=20, max_depth=3)

    assert set(Xk["a"].unique().to_list()) <= {"red", "green", "blue"}
    assert set(Xk["b"].unique().to_list()) <= {"cat", "dog", "bird"}
