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


def test_second_order_xgb_categorical_mixed_schema_matches():
    X, rng = _make_mixed()
    Xk = knockoff.get_knockoffs(
        X, method="second_order", rng=rng, categorical_method="xgb",
        max_depth=3, n_estimators=20,
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


def test_second_order_shrink_kwarg_is_forwarded():
    # Regression test: get_second_order's knockoffCallable used to have its
    # **kwargs forwarding commented out, so e.g. shrink=False silently had no
    # effect (always used rbridge's shrink=True default). Confirm the kwarg
    # now actually reaches rKnockoff.create_second_order and changes output.
    n = 200
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n) + 0.5 * x0
    X = pl.DataFrame({"x0": x0, "x1": x1})

    Xk_shrink = knockoff.get_second_order(
        X, rng=np.random.default_rng(1), categorical_method="ohe", shrink=True,
    )
    Xk_noshrink = knockoff.get_second_order(
        X, rng=np.random.default_rng(1), categorical_method="ohe", shrink=False,
    )
    assert not Xk_shrink.equals(Xk_noshrink)


def test_unrecognized_method_raises():
    X, rng = _make_mixed(n=10)
    with pytest.raises(ValueError):
        knockoff.get_knockoffs(X, method="not_a_real_method", rng=rng)


def test_get_knockoffs_xgb_scip_computes_conditional_expectations_when_omitted():
    # Regression test: get_knockoffs used to dereference
    # conditional_expectations.columns unconditionally for
    # categorical_method in ("ranger_scip", "xgb_scip"), so omitting it
    # crashed with AttributeError instead of computing it internally.
    X, rng = _make_mixed()
    Xk = knockoff.get_knockoffs(
        X, method="second_order", rng=rng, categorical_method="xgb_scip",
    )
    assert Xk.schema == X.schema
    assert len(Xk) == len(X)


def test_get_knockoffs_xgb_SCIP_residuals_method_defaults_to_normal():
    # Regression test: get_knockoffs used to index kwargs['residuals_method']
    # directly, so omitting it raised KeyError despite get_xgbSCIP's default.
    X, _ = _make_mixed()
    Xk_default = knockoff.get_knockoffs(X, method="xgb_SCIP", rng=np.random.default_rng(3))
    Xk_normal = knockoff.get_knockoffs(
        X, method="xgb_SCIP", rng=np.random.default_rng(3), residuals_method="normal",
    )
    assert Xk_default.equals(Xk_normal)


def test_conditional_expectations_matched_by_column_name():
    # Regression test: a user-supplied conditional_expectations frame with the
    # right names in a different order used to be applied positionally, putting
    # each numeric knockoff in the wrong column. It is now matched by name.
    from heteroknockoffpy import xgbScip

    X, _ = _make_mixed()
    ce = xgbScip.get_forest_conditional_expectations(X, rng=np.random.default_rng(5))
    Xk_ordered = knockoff.get_second_order(
        X, rng=np.random.default_rng(2), categorical_method="xgb_scip",
        conditional_expectations=ce,
    )
    Xk_reversed = knockoff.get_second_order(
        X, rng=np.random.default_rng(2), categorical_method="xgb_scip",
        conditional_expectations=ce.select(list(reversed(ce.columns))),
    )
    assert Xk_ordered.equals(Xk_reversed)


@pytest.mark.parametrize("bad", ["missing", "extra"])
def test_conditional_expectations_wrong_columns_raise(bad):
    X, rng = _make_mixed()
    ce = pl.DataFrame({"x0": X["x0"], "x1": X["x1"]})
    ce = ce.drop("x1") if bad == "missing" else ce.with_columns(pl.col("x0").alias("zz"))
    with pytest.raises(ValueError, match="matched by name"):
        knockoff.get_knockoffs(
            X, method="second_order", rng=rng, categorical_method="xgb_scip",
            conditional_expectations=ce,
        )
