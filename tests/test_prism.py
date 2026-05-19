#
#//  test_prism.py
#//  heteroknockoffpy
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import importance


def _make_synthetic(n: int = 200, p: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    cols = [f"x{i}" for i in range(p)]
    X = pl.DataFrame(dict(zip(cols, X_np.T)))
    Xk = pl.DataFrame(dict(zip(cols, Xk_np.T)))
    y = pl.Series("y", X_np[:, 0] + rng.standard_normal(n) * 0.5)
    return X, Xk, y


def test_prism_w_shape_and_nonneg():
    X, Xk, y = _make_synthetic()
    # short path so the test runs fast
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [16, 8],
        lambda_path = lambda_path,
        epochs_per_batch = 3,
    )
    assert imp.shape == (20,), f"Expected shape (20,), got {imp.shape}"
    assert np.all(imp >= 0), "Importances should be non-negative"


def test_prism_g_shape_and_nonneg():
    X, Xk, y = _make_synthetic()
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [16, 8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        epochs_per_batch = 3,
    )
    assert imp.shape == (20,), f"Expected shape (20,), got {imp.shape}"
    assert np.all(imp >= 0), "Importances should be non-negative"


def test_prism_g_bandwidth():
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 3)
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = lambda_path,
        epochs_per_batch = 2,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_adaptive():
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_min = 1e-3,
        epochs_per_batch = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)
