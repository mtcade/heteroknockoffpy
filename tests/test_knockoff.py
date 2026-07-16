#
#//  tests/test_knockoff.py
#//  heteroknockoffpy
#//
#//  Tests for heteroknockoffpy.knockoff.get_knockoffs: second_order and
#//  ranger_SCIP/xgb_SCIP dispatch paths. torch_GAN is deliberately not tested
#//  here -- its interesting behavior post-refactor is specifically the
#//  torch/xgboost isolation dispatch, covered in test_processIsolation.py.
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import knockoff


def _make_mixed(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    cat = pl.Series([str(v) for v in rng.integers(0, 3, n)]).cast(pl.Categorical)
    return pl.DataFrame({"x0": x0, "x1": x1, "xc": cat}), rng


def test_second_order_linear_mixed_schema_matches():
    X, rng = _make_mixed()
    Xk = knockoff.get_knockoffs(
        X, method="second_order", rng=rng, categorical_method="linear",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_second_order_ohe_numeric_only():
    n = 60
    rng = np.random.default_rng(1)
    X = pl.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    Xk = knockoff.get_knockoffs(
        X, method="second_order", rng=rng, categorical_method="ohe",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_ranger_SCIP_mixed_schema_matches():
    X, rng = _make_mixed()
    Xk = knockoff.get_knockoffs(
        X, method="ranger_SCIP", rng=rng, residuals_method="normal",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


@pytest.mark.parametrize("residuals_method", ["normal", "permute"])
def test_xgb_SCIP_mixed_schema_matches(residuals_method):
    X, rng = _make_mixed()
    Xk = knockoff.get_knockoffs(
        X, method="xgb_SCIP", rng=rng, residuals_method=residuals_method,
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_xgb_SCIP_numeric_only():
    n = 60
    rng = np.random.default_rng(1)
    X = pl.DataFrame({"x0": rng.standard_normal(n), "x1": rng.standard_normal(n)})
    Xk = knockoff.get_knockoffs(
        X, method="xgb_SCIP", rng=rng, residuals_method="normal",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_second_order_xgb_scip_mixed_schema_matches():
    # get_knockoffs requires conditional_expectations up front for
    # categorical_method in ("ranger_scip", "xgb_scip"); exercise the lower-level
    # get_second_order entry point directly instead, which computes them
    # internally (via xgbScip.get_forest_conditional_expectations) when omitted.
    X, rng = _make_mixed()
    Xk = knockoff.get_second_order(
        X, rng=rng, categorical_method="xgb_scip",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_unrecognized_method_raises():
    X, rng = _make_mixed(n=10)
    with pytest.raises(ValueError):
        knockoff.get_knockoffs(X, method="not_a_real_method", rng=rng)
