#
#//  tests/test_weight.py
#//  heteroknockoffpy
#//
#//  Targeted regression tests for the optional `weight` parameter added
#//  across knockoff/importance methods -- one test per underlying weighting
#//  mechanism (xgboost sample_weight, ranger case.weights, sklearn
#//  sample_weight, second_order's weighted mean/covariance, torch-PRISM's
#//  weighted loss). Not exhaustive -- see CLAUDE.md/the session plan for
#//  the full per-function survey; this just proves weight is actually
#//  reaching each mechanism, not merely accepted and ignored.
#

import numpy as np
import polars as pl

from heteroknockoffpy import importance, knockoff, rbridge


def _mixture_X_y(rng: np.random.Generator, n: int = 400):
    """
    Two subpopulations sharing the same X but different generative
    relationships to y: rows 0..n//2-1 have y driven by x0, rows n//2..n-1
    have y driven by x1 (independent of x0). A weight vector favoring one
    half should make that half's variable dominate the fitted importances;
    weight=None (or uniform) should leave both comparably important.
    """
    half = n // 2
    X_np = rng.standard_normal((n, 3))
    y = np.empty(n)
    y[:half] = 5.0 * X_np[:half, 0] + 0.1 * rng.standard_normal(half)
    y[half:] = 5.0 * X_np[half:, 1] + 0.1 * rng.standard_normal(n - half)
    X = pl.DataFrame({f"x{i}": X_np[:, i] for i in range(3)})
    Xk = pl.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(3)})
    y_series = pl.Series("y", y)
    weight_favor_first_half = np.concatenate([np.full(half, 1.0), np.full(n - half, 1e-6)])
    return X, Xk, y_series, weight_favor_first_half


def test_xgb_sample_weight_shifts_importance_to_weighted_subpopulation():
    rng = np.random.default_rng(0)
    X, Xk, y, weight = _mixture_X_y(rng)

    imp_weighted = importance.xgbImportances(X=X, Xk=Xk, y=y, weight=weight, rng=rng)
    imp_unweighted = importance.xgbImportances(X=X, Xk=Xk, y=y, rng=rng)

    # x0 (index 0) drives y in the heavily-weighted first half; weighting
    # toward it should make x0 relatively more important than x1 (index 1)
    # compared to the unweighted fit, which sees both halves equally.
    ratio_weighted = imp_weighted[0] / max(imp_weighted[1], 1e-9)
    ratio_unweighted = imp_unweighted[0] / max(imp_unweighted[1], 1e-9)
    assert ratio_weighted > ratio_unweighted, (
        f"weighted x0/x1 importance ratio ({ratio_weighted:.3f}) should exceed "
        f"unweighted ({ratio_unweighted:.3f})"
    )


def test_ranger_case_weights_shifts_importance_to_weighted_subpopulation():
    rng = np.random.default_rng(1)
    X, Xk, y, weight = _mixture_X_y(rng, n=300)

    imp_weighted = importance.rangerGiniImportances(X=X, Xk=Xk, y=y, weight=weight, num_trees=50)
    imp_unweighted = importance.rangerGiniImportances(X=X, Xk=Xk, y=y, num_trees=50)

    ratio_weighted = imp_weighted[0] / max(imp_weighted[1], 1e-9)
    ratio_unweighted = imp_unweighted[0] / max(imp_unweighted[1], 1e-9)
    assert ratio_weighted > ratio_unweighted, (
        f"weighted x0/x1 importance ratio ({ratio_weighted:.3f}) should exceed "
        f"unweighted ({ratio_unweighted:.3f})"
    )


def test_lasso_sample_weight_shifts_importance_to_weighted_subpopulation():
    rng = np.random.default_rng(2)
    X, Xk, y, weight = _mixture_X_y(rng)

    imp_weighted = importance.lassoImportances(X=X, Xk=Xk, y=y, weight=weight)
    imp_unweighted = importance.lassoImportances(X=X, Xk=Xk, y=y)

    assert imp_weighted[0] > imp_unweighted[0], (
        f"weighted x0 importance ({imp_weighted[0]:.4f}) should exceed "
        f"unweighted ({imp_unweighted[0]:.4f}) when weight favors x0's subpopulation"
    )
    assert imp_weighted[1] < imp_unweighted[1], (
        f"weighted x1 importance ({imp_weighted[1]:.4f}) should be below "
        f"unweighted ({imp_unweighted[1]:.4f}) when weight disfavors x1's subpopulation"
    )


def test_second_order_weighted_covariance_matches_weighted_cluster():
    """
    X is two well-separated clusters (mean -10 and +10). A weight vector
    that puts nearly all mass on the +10 cluster should make the returned
    knockoffs' empirical mean land near +10, not near the unweighted
    overall mean (~0) -- direct evidence the weighted mu/Sigma path in
    get_knockoffs_second_order_np is actually being used, not the
    original create.second_order's own unweighted internal estimate.
    """
    rng = np.random.default_rng(3)
    n_per_cluster = 300
    p = 2
    cluster_low = rng.normal(loc=-10.0, scale=1.0, size=(n_per_cluster, p))
    cluster_high = rng.normal(loc=10.0, scale=1.0, size=(n_per_cluster, p))
    X = np.vstack([cluster_low, cluster_high])
    weight = np.concatenate([np.full(n_per_cluster, 1e-6), np.full(n_per_cluster, 1.0)])

    Xk_weighted = rbridge.get_knockoffs_second_order_np(X=X, weight=weight, rng=np.random.default_rng(4))
    Xk_unweighted = rbridge.get_knockoffs_second_order_np(X=X, rng=np.random.default_rng(4))

    assert Xk_weighted.mean() > 5.0, (
        f"weighted knockoffs' mean ({Xk_weighted.mean():.2f}) should be near the "
        "heavily-weighted +10 cluster"
    )
    assert abs(Xk_unweighted.mean()) < 5.0, (
        f"unweighted knockoffs' mean ({Xk_unweighted.mean():.2f}) should be near the "
        "overall (~0) mean of both clusters"
    )


def test_torch_prism_weight_shifts_importance_to_weighted_subpopulation():
    rng = np.random.default_rng(5)
    X, Xk, y, weight = _mixture_X_y(rng, n=200)

    kwargs = dict(
        # epochs=100 (not the smaller value used elsewhere): the BSS loop now
        # keeps a persistent deep-layer weight-decay term active throughout
        # every block, including lambda_path=[0.0] (GRIP2 Remark 1) -- a
        # too-short single block spends most of its steps fighting that decay
        # rather than letting the weighted-vs-unweighted loss reshape the
        # first-layer group norms, which washes out the effect under test.
        X=X, Xk=Xk, y=y, layers=[8],
        lambda_path=[0.0], epochs=100, model_type='mlp', verbose=0,
    )
    # rng must be fixed and identical for both calls: without it, model init
    # and minibatch/randperm draws fall back to torch's ambient global RNG
    # state, which depends on what ran earlier in the process -- making the
    # weighted-vs-unweighted comparison nondeterministic (order-dependent)
    # instead of isolating the effect of `weight` alone.
    imp_weighted = importance.grip2Importances(**kwargs, weight=weight, rng=np.random.default_rng(42))
    imp_unweighted = importance.grip2Importances(**kwargs, rng=np.random.default_rng(42))

    ratio_weighted = imp_weighted[0] / max(imp_weighted[1], 1e-9)
    ratio_unweighted = imp_unweighted[0] / max(imp_unweighted[1], 1e-9)
    assert ratio_weighted > ratio_unweighted, (
        f"weighted x0/x1 group-norm ratio ({ratio_weighted:.3f}) should exceed "
        f"unweighted ({ratio_unweighted:.3f})"
    )
