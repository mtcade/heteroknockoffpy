#
#//  test_prism.py
#//  heteroknockoffpy
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import importance
from heteroknockoffpy.heteroknockofftorch.torchImportances import PRISMPredictionModel
from heteroknockoffpy.heteroknockofftorch.prismImportances import _prism_setup


def _make_synthetic(n: int = 200, p: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_np = rng.standard_normal((n, p))
    Xk_np = rng.standard_normal((n, p))
    cols = [f"x{i}" for i in range(p)]
    X = pl.DataFrame(dict(zip(cols, X_np.T)))
    Xk = pl.DataFrame(dict(zip(cols, Xk_np.T)))
    y = pl.Series("y", X_np[:, 0] + rng.standard_normal(n) * 0.5)
    return X, Xk, y


# ---------------------------------------------------------------------------
# Basic shape / non-negativity — a_path=None (default: use lambda values)
# ---------------------------------------------------------------------------

def test_grip2_shape_and_nonneg():
    X, Xk, y = _make_synthetic()
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [16, 8],
        lambda_path = lambda_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (20,), f"Expected shape (20,), got {imp.shape}"
    assert np.all(imp >= 0), "Importances should be non-negative"


def test_torch_prism_shape_and_nonneg():
    X, Xk, y = _make_synthetic()
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [16, 8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (20,), f"Expected shape (20,), got {imp.shape}"
    assert np.all(imp >= 0), "Importances should be non-negative"


def test_torch_prism_bandwidth():
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 3)
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = lambda_path,
        epochs = 2, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# a_path — explicit per-stage input-layer penalty values
# ---------------------------------------------------------------------------

def test_grip2_a_path_constant():
    """Constant a_path replicates the old scalar a_min behaviour."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 5)
    a_path = [1e-3] * len(lambda_path)
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_a_path_uniform():
    """Linearly increasing a_path — one value per lambda stage."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 5)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_a_path_shuffled():
    """Shuffled (non-monotone) a_path should run without error."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=42)
    lambda_path = np.logspace(0, -1, 5)
    rng = np.random.default_rng(42)
    a_path = list(rng.permutation(np.linspace(0.3, 1.0, len(lambda_path))))
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_a_path_uniform():
    """torchPrism with a linearly spaced a_path."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_a_path_shuffled():
    """torchPrism with a shuffled a_path."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=11)
    lambda_path = np.logspace(0, -1, 4)
    rng = np.random.default_rng(11)
    a_path = list(rng.permutation(np.linspace(0.3, 1.0, len(lambda_path))))
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_a_path_none_runs():
    """a_path=None (default) uses lambda_path values as input-layer penalty."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = None,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_a_path_increasing_lambda():
    """Increasing lambda_path (low→high) with a_path=None."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = list(np.linspace(1e-3, 0.1, 8))
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        epochs = 4, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# batch_size — mini-batch gradient steps
# ---------------------------------------------------------------------------

def test_grip2_batch_size():
    """batch_size < n enables per-epoch mini-batch gradient steps."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        batch_size = 32,
        epochs = 4, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_batch_size():
    """batch_size < n enables per-epoch mini-batch gradient steps for torchPrism."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        batch_size = 32,
        epochs = 4, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_batch_size_larger_than_n():
    """batch_size >= n falls back to full-batch (no crash)."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 3)
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        batch_size = 512,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_grip2_a_path_and_batch_size():
    """a_path and batch_size combined."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=7)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.grip2Importances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = 32,
        epochs = 4, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_a_path_and_batch_size():
    """torchPrism with both a_path and batch_size."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=8)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.torchPrismImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = 32,
        epochs = 4, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical outcome — varying k, auto_diff
# ---------------------------------------------------------------------------

def _make_cat_y(n: int, k: int, seed: int = 0) -> pl.Series:
    rng = np.random.default_rng(seed)
    labels = [str(i) for i in range(k)]
    vals = [labels[int(v)] for v in rng.integers(0, k, size=n)]
    return pl.Series("y", vals).cast(pl.Categorical)


def _make_mixed_X(
    n: int,
    p_numeric: int,
    cat_cols: list,
    seed: int = 0,
):
    """Returns (X, Xk, p_total). cat_cols[i] = number of categories for column i."""
    rng = np.random.default_rng(seed)
    data: dict = {}
    kdata: dict = {}
    for i in range(p_numeric):
        col = f"xn{i}"
        data[col]  = rng.standard_normal(n).tolist()
        kdata[col] = rng.standard_normal(n).tolist()
    for i, n_cats in enumerate(cat_cols):
        col = f"xc{i}"
        cats = [str(j) for j in range(n_cats)]
        data[col]  = pl.Series(col, [cats[int(v)] for v in rng.integers(0, n_cats, n)]).cast(pl.Categorical)
        kdata[col] = pl.Series(col, [cats[int(v)] for v in rng.integers(0, n_cats, n)]).cast(pl.Categorical)
    X  = pl.DataFrame(data)
    Xk = pl.DataFrame(kdata)
    return X, Xk, p_numeric + len(cat_cols)


_LAMBDA_PATH_SHORT = np.logspace(-2, -3, 3)


def test_torch_prism_cat_output_k2_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=1)
    y = _make_cat_y(150, k=2, seed=1)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_output_k3_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=2)
    y = _make_cat_y(150, k=3, seed=2)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_output_k5_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=3)
    y = _make_cat_y(150, k=5, seed=3)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical outcome — bandwidth
# ---------------------------------------------------------------------------

def test_torch_prism_cat_output_k3_bandwidth():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=4)
    y = _make_cat_y(150, k=3, seed=4)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical input — varying category counts, continuous output
# ---------------------------------------------------------------------------

def test_torch_prism_cat_input_2cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 2], seed=5)
    y = pl.Series("y", np.random.default_rng(5).standard_normal(150))
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_input_3cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[3, 3], seed=6)
    y = pl.Series("y", np.random.default_rng(6).standard_normal(150))
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_input_4cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[4, 4], seed=7)
    y = pl.Series("y", np.random.default_rng(7).standard_normal(150))
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical input + a_path — exercises the categorical proximal branch
# ---------------------------------------------------------------------------

def test_grip2_cat_input_a_path():
    """a_path with categorical input features exercises block-Frobenius group regularisation."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=20)
    y = pl.Series("y", np.random.default_rng(20).standard_normal(150))
    lambda_path = _LAMBDA_PATH_SHORT
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.grip2Importances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_input_a_path():
    """torchPrism with categorical input + a_path."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=21)
    y = pl.Series("y", np.random.default_rng(21).standard_normal(150))
    lambda_path = _LAMBDA_PATH_SHORT
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Mixed: categorical input + categorical output
# ---------------------------------------------------------------------------

def test_torch_prism_cat_input_cat_output_auto_diff():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[3, 3], seed=8)
    y = _make_cat_y(150, k=3, seed=8)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_torch_prism_cat_input_cat_output_bandwidth():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[3, 3], seed=9)
    y = _make_cat_y(150, k=3, seed=9)
    imp = importance.torchPrismImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# model_type — pairwise and additive architectures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_grip2_model_type_numeric(model_type):
    X, Xk, y = _make_synthetic(n=100, p=5)
    imp = importance.grip2Importances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_torch_prism_model_type_numeric(model_type):
    X, Xk, y = _make_synthetic(n=100, p=5)
    imp = importance.torchPrismImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        local_grad_method="auto_diff",
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_grip2_model_type_cat_input(model_type):
    """OHE columns are treated as independent feature pairs in pairwise/additive."""
    X, Xk, p = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=30)
    y = pl.Series("y", np.random.default_rng(30).standard_normal(100))
    imp = importance.grip2Importances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_torch_prism_model_type_cat_input(model_type):
    X, Xk, p = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=31)
    y = pl.Series("y", np.random.default_rng(31).standard_normal(100))
    imp = importance.torchPrismImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        local_grad_method="auto_diff",
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (2 * p,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Warmup — direct PRISMPredictionModel
# ---------------------------------------------------------------------------

def _make_groups_np(n: int = 100, p: int = 5, seed: int = 0):
    """Return (X_all_np, y_np, groups) for a numeric-only dataset."""
    rng = np.random.default_rng(seed)
    X_all = np.concatenate(
        [rng.standard_normal((n, p)), rng.standard_normal((n, p))], axis=1
    )
    y = rng.standard_normal(n)
    groups = [[j] for j in range(2 * p)]
    return X_all, y, groups


@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_warmup_basic(model_type):
    """n_warmup > 0 runs without error and produces correct snapshot shape."""
    X_all, y, groups = _make_groups_np(n=100, p=5)
    m = PRISMPredictionModel(
        input_size=10, layers=[16], model_type=model_type,
        n_warmup=50, warmup_patience=0,
    )
    snaps = m.fit(X_all, y, groups, lambda_path=np.logspace(-1, -2, 4))
    arr = np.array(snaps)
    assert arr.shape == (4, 10)
    assert np.all(arr >= 0)


@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_warmup_with_patience(model_type):
    """Patience-based early stopping: warmup exits before hitting n_warmup cap."""
    X_all, y, groups = _make_groups_np(n=100, p=5, seed=1)
    m = PRISMPredictionModel(
        input_size=10, layers=[16], model_type=model_type,
        n_warmup=50000,           # high cap — should stop early
        warmup_patience=2,
        warmup_check_interval=10,
        warmup_tol=1e-10,         # very tight → triggers patience quickly
        warmup_val_frac=0.2,
    )
    snaps = m.fit(X_all, y, groups, lambda_path=np.logspace(-1, -2, 3))
    assert np.array(snaps).shape == (3, 10)


def test_warmup_and_batch_size():
    """Warmup + minibatch training combined."""
    X_all, y, groups = _make_groups_np(n=100, p=5, seed=2)
    m = PRISMPredictionModel(
        input_size=10, layers=[16],
        n_warmup=30, warmup_patience=0,
    )
    snaps = m.fit(X_all, y, groups, lambda_path=np.logspace(-1, -2, 3), batch_size=32)
    assert np.array(snaps).shape == (3, 10)


# ---------------------------------------------------------------------------
# Count outcome — PoissonNLLLoss(log_input=True)
# ---------------------------------------------------------------------------

def _make_count_y(n: int, seed: int = 0) -> pl.Series:
    rng = np.random.default_rng(seed)
    return pl.Series("y", rng.poisson(lam=5, size=n))


def test_grip2_count_outcome():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=40)
    y = _make_count_y(150, seed=40)
    imp = importance.grip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_count_outcome_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=41)
    y = _make_count_y(150, seed=41)
    imp = importance.torchPrismImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_torch_prism_count_outcome_bandwidth():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=42)
    y = _make_count_y(150, seed=42)
    imp = importance.torchPrismImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='bandwidth',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical outcome — grip2 (CrossEntropyLoss path)
# ---------------------------------------------------------------------------

def test_grip2_cat_outcome():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=43)
    y = _make_cat_y(150, k=3, seed=43)
    imp = importance.grip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# torchPrismLocalGradients — shape, finite values, both local_grad_method values
# ---------------------------------------------------------------------------

def test_torch_prism_local_gradients_numeric_auto_diff():
    """Returns (n, p) for numeric-only X with auto_diff."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=44)
    grads = importance.torchPrismLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


def test_torch_prism_local_gradients_numeric_bandwidth():
    """Returns (n, p) for numeric-only X with bandwidth finite difference."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=45)
    grads = importance.torchPrismLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='bandwidth',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


def test_torch_prism_local_gradients_cat_input():
    """Categorical input: p_ohe_x = p_numeric + sum(c_k - 1 per cat variable)."""
    # 3 numeric + 2-cat (1 ohe col) + 3-cat (2 ohe cols) → p_ohe_x = 3+1+2 = 6
    X, Xk, _ = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=46)
    y = pl.Series("y", np.random.default_rng(46).standard_normal(100))
    grads = importance.torchPrismLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert grads.shape == (100, 6)
    assert np.isfinite(grads).all()


def test_torch_prism_local_gradients_count_outcome():
    """torchPrismLocalGradients with Poisson count outcome."""
    X, Xk, _ = _make_synthetic(n=100, p=5, seed=47)
    y = _make_count_y(100, seed=47)
    grads = importance.torchPrismLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


# ---------------------------------------------------------------------------
# prismGrip2Importances — single-pass dual importances
# ---------------------------------------------------------------------------

def test_prism_grip2_returns_tuple_of_two_arrays():
    """prismGrip2Importances returns a 2-tuple of numpy arrays."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=50)
    result = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert isinstance(result, tuple) and len(result) == 2
    g_imp, w_imp = result
    assert isinstance(g_imp, np.ndarray)
    assert isinstance(w_imp, np.ndarray)


def test_prism_grip2_shape_and_nonneg():
    """Both G and W importances have shape (2*p,) and are non-negative."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=51)
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (10,), g_imp.shape
    assert w_imp.shape == (10,), w_imp.shape
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_grip2_bandwidth():
    """prismGrip2Importances with bandwidth local_grad_method."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=52)
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='bandwidth',
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_grip2_g_and_w_differ():
    """G and W importances use different scoring methods and must not be identical."""
    X, Xk, y = _make_synthetic(n=150, p=8, seed=53)
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[16, 8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 5), epochs = 4, n_warmup = 0,
    )
    assert not np.allclose(g_imp, w_imp), "PRISM_g and PRISM_w should differ"


def test_prism_grip2_w_matches_standalone_grip2_structure():
    """W importances are group norms — all non-negative and finite, like standalone grip2."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=54)
    _, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs = 3, n_warmup = 0,
    )
    assert np.all(w_imp >= 0)
    assert np.isfinite(w_imp).all()


def test_prism_grip2_cat_input():
    """prismGrip2Importances with categorical input columns."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=55)
    y = pl.Series("y", np.random.default_rng(55).standard_normal(150))
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (2 * p,), g_imp.shape
    assert w_imp.shape == (2 * p,), w_imp.shape
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_grip2_cat_output():
    """prismGrip2Importances with categorical outcome (CrossEntropyLoss path)."""
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=56)
    y = _make_cat_y(150, k=3, seed=56)
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_grip2_count_outcome():
    """prismGrip2Importances with Poisson count outcome."""
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=57)
    y = _make_count_y(150, seed=57)
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_grip2_snapshot_count_matches_lambda_stages():
    """Each lambda stage produces one G snapshot and one W snapshot."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=58)
    lambda_path = np.logspace(0, -1, 6)
    # Both outputs are means over 6 snapshots; we can't inspect them directly,
    # but running without error and returning the right shape confirms the counts match.
    g_imp, w_imp = importance.prismGrip2Importances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=lambda_path, epochs = 3, n_warmup = 0,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)


# ---------------------------------------------------------------------------
# _prism_setup layout — X-then-Xk contiguous blocks (regression: previously
# oheDict/groups were grouped numeric-then-categorical across the whole
# concatenated [X, Xk] frame -- get_oheDict's documented canonical order for a
# single encoded frame -- which only coincides with an X-then-Xk split when a
# dataset has no categorical columns. wFromImportances, calculatorOps.py's
# torch_prism_grip2 row-builder, and _PRISMNetworkPairwise/_PRISMNetworkAdditive's
# `p = input_size // 2` split all assume the latter. Shape/non-negativity
# checks (used everywhere else in this file) can't catch an ordering bug, since
# they're identical regardless of internal group order -- these tests assert
# on identity/order instead.
# ---------------------------------------------------------------------------

def _assert_prism_setup_x_then_xk_layout(X: pl.DataFrame, Xk: pl.DataFrame, y: pl.Series) -> None:
    p = X.shape[1]
    X_all_np, y_np, groups, oheDict, loss_func, out_dim, desc = _prism_setup(
        X=X, Xk=Xk, y=y, layers=[8], outcome_type=None, drop_first=True,
    )
    assert len(groups) == 2 * p

    x_cols = list(X.columns)
    keys = list(oheDict.keys())
    assert keys[:p] == x_cols, f"expected X's own columns first, got {keys[:p]}"
    assert keys[p:] == [c + '~' for c in x_cols], f"expected Xk's columns second, got {keys[p:]}"

    # group i (X feature) and group i+p (its Xk copy) must reference the SAME
    # relative OHE offset, just shifted by the per-side OHE width -- i.e. they
    # really are the same original feature's X and Xk copy, not unrelated columns.
    p_ohe_x = X_all_np.shape[1] // 2
    for i in range(p):
        g_x = groups[i]
        g_xk = groups[i + p]
        shifted = [idx - p_ohe_x for idx in g_xk]
        assert shifted == g_x, (
            f"feature {x_cols[i]}: X group {g_x} vs Xk group {g_xk} shifted {shifted} "
            f"(expected Xk group == X group shifted by {p_ohe_x})"
        )


def test_prism_setup_layout_numeric_only():
    """Sanity baseline: purely numeric data, where old and new code agree."""
    X, Xk, y = _make_synthetic(n=50, p=6, seed=1)
    _assert_prism_setup_x_then_xk_layout(X, Xk, y)


def test_prism_setup_layout_mixed_numeric_categorical():
    """
    Regression test for the X/Xk OHE layout bug. Before the fix, with
    p_numeric=4 and cat_cols=[3, 3] (p=6), _prism_setup's oheDict grouped ALL
    non-categorical columns (X's 4 numeric AND Xk's 4 numeric) before any
    categorical columns, so oheDict.keys()[:p] was
    [xn0,xn1,xn2,xn3,xn0~,xn1~] -- NOT X's own columns
    [xn0,xn1,xn2,xn3,xc0,xc1].
    """
    X, Xk, p = _make_mixed_X(n=100, p_numeric=4, cat_cols=[3, 3], seed=42)
    y = pl.Series("y", np.random.default_rng(42).standard_normal(100))
    _assert_prism_setup_x_then_xk_layout(X, Xk, y)


def test_prism_setup_layout_matches_synth_sweep_shape():
    """
    Same p_numeric/p_categorical shape as the silverknockoff synth_sweep_3
    bundle that originally surfaced this bug (p_numeric=20, 10 categorical
    variables with 4 categories each).
    """
    X, Xk, p = _make_mixed_X(n=64, p_numeric=20, cat_cols=[4] * 10, seed=99)
    y = pl.Series("y", np.random.default_rng(99).standard_normal(64))
    _assert_prism_setup_x_then_xk_layout(X, Xk, y)


# ---------------------------------------------------------------------------
# Flip-sign antisymmetry: W_j(swap_j[X,Xk]) == -W_j([X,Xk])
#
# Swapping ALL feature/knockoff pairs simultaneously (pass Xk as X and X as Xk)
# with the SAME rng seed is the composition of swapping every individual j, so
# it exercises the same coupled-randomness property the reference torch_prism.py/
# grip2.py docstrings prove architecturally. This is an optimization-
# dependent (not exactly-zero-tolerance) property -- it only holds once
# training has moved enough beyond the shared random initialization for the
# loss landscape's symmetry (rather than the arbitrary un-swapped initial
# weights) to dominate -- so a strong single-signal column with enough
# training is used, and the check is a sign flip + approximate magnitude
# match rather than exact equality.
# ---------------------------------------------------------------------------

def _make_signal_XXk(seed: int, n: int = 300, p: int = 5):
    """X_0 is the only signal; everything else is null."""
    X = np.random.default_rng(seed).standard_normal((n, p))
    Xk = np.random.default_rng(seed + 1).standard_normal((n, p))
    y = X[:, 0] * 3 + np.random.default_rng(seed + 2).standard_normal(n) * 0.2
    Xdf = pl.DataFrame({f"x{i}": X[:, i] for i in range(p)})
    Xkdf = pl.DataFrame({f"x{i}": Xk[:, i] for i in range(p)})
    return Xdf, Xkdf, pl.Series("y", y)


_ANTISYM_LAMBDA_PATH = list(np.logspace(-1, -2, 20))


def _assert_flip_sign(W: np.ndarray, W_swapped: np.ndarray, j: int = 0) -> None:
    assert W[j] > 0, f"expected a strong positive W for the signal feature, got {W[j]}"
    assert W_swapped[j] < 0, f"swap should flip the signal feature's sign, got {W_swapped[j]}"
    assert abs(W_swapped[j] + W[j]) < 0.5 * abs(W[j]), (
        f"swap should approximately negate: W={W[j]}, W_swapped={W_swapped[j]}"
    )


@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_torch_prism_antisymmetry_continuous(model_type):
    Xdf, Xkdf, yS = _make_signal_XXk(seed=100)
    kwargs = dict(
        layers=[8], epochs=300, n_warmup=1000,
        model_type=model_type, lambda_path=_ANTISYM_LAMBDA_PATH,
    )
    imp = importance.torchPrismImportances(X=Xdf, Xk=Xkdf, y=yS, rng=np.random.default_rng(99), **kwargs)
    W = importance.wFromImportances(imp)

    imp_sw = importance.torchPrismImportances(X=Xkdf, Xk=Xdf, y=yS, rng=np.random.default_rng(99), **kwargs)
    W_sw = importance.wFromImportances(imp_sw)

    _assert_flip_sign(W, W_sw)


@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_grip2_antisymmetry_continuous(model_type):
    Xdf, Xkdf, yS = _make_signal_XXk(seed=101)
    kwargs = dict(
        layers=[8], epochs=300, n_warmup=1000,
        model_type=model_type, lambda_path=_ANTISYM_LAMBDA_PATH,
    )
    imp = importance.grip2Importances(X=Xdf, Xk=Xkdf, y=yS, rng=np.random.default_rng(98), **kwargs)
    W = importance.wFromImportances(imp)

    imp_sw = importance.grip2Importances(X=Xkdf, Xk=Xdf, y=yS, rng=np.random.default_rng(98), **kwargs)
    W_sw = importance.wFromImportances(imp_sw)

    _assert_flip_sign(W, W_sw)


@pytest.mark.parametrize("model_type", ["mlp", "pairwise", "additive"])
def test_prism_grip2_antisymmetry_continuous(model_type):
    """Both the G and W halves of prismGrip2Importances must flip sign under swap."""
    Xdf, Xkdf, yS = _make_signal_XXk(seed=102)
    kwargs = dict(
        layers=[8], epochs=300, n_warmup=1000,
        model_type=model_type, lambda_path=_ANTISYM_LAMBDA_PATH,
    )
    g_imp, w_imp = importance.prismGrip2Importances(X=Xdf, Xk=Xkdf, y=yS, rng=np.random.default_rng(97), **kwargs)
    g_imp_sw, w_imp_sw = importance.prismGrip2Importances(X=Xkdf, Xk=Xdf, y=yS, rng=np.random.default_rng(97), **kwargs)

    _assert_flip_sign(importance.wFromImportances(g_imp), importance.wFromImportances(g_imp_sw))
    _assert_flip_sign(importance.wFromImportances(w_imp), importance.wFromImportances(w_imp_sw))


def test_torch_prism_antisymmetry_categorical_outcome():
    """Flip-sign property with a categorical (multiclass) outcome."""
    n, p = 300, 5
    X = np.random.default_rng(103).standard_normal((n, p))
    Xk = np.random.default_rng(104).standard_normal((n, p))
    logits = X[:, 0] * 3
    labels = (logits + np.random.default_rng(105).standard_normal(n) * 0.3 > 0).astype(int)
    Xdf = pl.DataFrame({f"x{i}": X[:, i] for i in range(p)})
    Xkdf = pl.DataFrame({f"x{i}": Xk[:, i] for i in range(p)})
    yS = pl.Series("y", [str(v) for v in labels]).cast(pl.Categorical)

    kwargs = dict(
        layers=[8], epochs=300, n_warmup=1000,
        model_type='mlp', lambda_path=_ANTISYM_LAMBDA_PATH,
    )
    imp = importance.torchPrismImportances(X=Xdf, Xk=Xkdf, y=yS, rng=np.random.default_rng(96), **kwargs)
    W = importance.wFromImportances(imp)

    imp_sw = importance.torchPrismImportances(X=Xkdf, Xk=Xdf, y=yS, rng=np.random.default_rng(96), **kwargs)
    W_sw = importance.wFromImportances(imp_sw)

    _assert_flip_sign(W, W_sw)


# ---------------------------------------------------------------------------
# wFromImportances — signed_max sign fix
# ---------------------------------------------------------------------------

def test_wFromImportances_signed_max_negates_knockoff_dominant():
    """
    signed_max must be antisymmetric like difference: when the knockoff
    column's importance dominates, W_j must come out negative, not positive
    (the pre-fix bug: W_out[j] = importances[j+p] instead of -importances[j+p]).
    """
    p = 3
    # feature 0: original dominates -> W[0] > 0
    # feature 1: knockoff dominates -> W[1] must be < 0 (this was the bug)
    # feature 2: tie -> W[2] == 0
    importances = np.array([5.0, 1.0, 2.0,   1.0, 4.0, 2.0])
    W = importance.wFromImportances(importances, W_method='signed_max')
    assert W.shape == (p,)
    assert W[0] == 5.0
    assert W[1] == -4.0, f"expected -4.0 (knockoff-dominant, negated), got {W[1]}"
    assert W[2] == 0.0


def test_wFromImportances_signed_max_matches_difference_sign():
    """signed_max and difference must always agree on sign."""
    rng = np.random.default_rng(200)
    p = 20
    importances = rng.uniform(0, 5, size=2 * p)
    W_diff = importance.wFromImportances(importances, W_method='difference')
    W_signed_max = importance.wFromImportances(importances, W_method='signed_max')
    same_sign = np.sign(W_diff) == np.sign(W_signed_max)
    assert same_sign.all(), (
        f"difference and signed_max disagree on sign at indices "
        f"{np.where(~same_sign)[0]}"
    )


# ---------------------------------------------------------------------------
# calibrate / n_blocks / lambda_min / lambda_max / a_min / a_max
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("conflicting_kwarg", [
    {"lambda_path": [0.1, 0.01]},
    {"a_path": [0.5, 0.6]},
    {"lambda_min": 1e-4},
    {"lambda_max": 1e-1},
    {"a_min": 0.2},
    {"a_max": 0.9},
])
def test_calibrate_conflicts_with_explicit_path_params(conflicting_kwarg):
    """calibrate=True must raise ValueError if the caller also supplies any
    of the parameters calibration itself determines."""
    X, Xk, y = _make_synthetic()
    with pytest.raises(ValueError, match="calibrate=True determines"):
        importance.grip2Importances(
            X=X, Xk=Xk, y=y, layers=[8],
            calibrate=True, epochs=3, n_warmup=0,
            **conflicting_kwarg,
        )


@pytest.mark.parametrize("path_kwarg", [
    {"lambda_path": [0.1, 0.01]},
    {"a_path": [0.5, 0.6]},
])
def test_n_blocks_conflicts_with_explicit_path(path_kwarg):
    """n_blocks only means something when a path is auto-generated -- must
    raise ValueError if combined with an explicit lambda_path/a_path,
    calibrate or not."""
    X, Xk, y = _make_synthetic()
    with pytest.raises(ValueError, match="n_blocks only applies"):
        importance.grip2Importances(
            X=X, Xk=Xk, y=y, layers=[8],
            n_blocks=5, epochs=3, n_warmup=0,
            **path_kwarg,
        )


def test_calibrate_lambda_range_matches_measured_gradient_ratio():
    """_calibrate_lambda_range's returned (lambda_min, lambda_max) must satisfy
    lambda_min == r_min / ratio_avg and lambda_max == r_max / ratio_avg, where
    ratio_avg is the SAME quantity independently re-derived here via a second,
    separate autograd computation -- proves the calibration formula itself is
    correct, not just that fit(calibrate=True) runs without crashing."""
    import torch
    from heteroknockoffpy.heteroknockofftorch.torchImportances import PRISMPredictionModel

    rng = np.random.default_rng(7)
    n, p = 150, 6
    X_np = rng.standard_normal((n, p)).astype(np.float32)
    Xk_np = rng.standard_normal((n, p)).astype(np.float32)
    y_np = (X_np[:, 0] * 2 + rng.standard_normal(n) * 0.5).astype(np.float32)
    groups = [[i] for i in range(2 * p)]
    X_all_np = np.concatenate([X_np, Xk_np], axis=1)

    model = PRISMPredictionModel(
        input_size=2 * p, layers=[8], model_type='mlp',
        n_warmup=100, warmup_patience=0,
        rng=np.random.default_rng(7),
    )
    X_tensor = torch.tensor(X_all_np).to(model.device)
    y_tensor = torch.tensor(y_np).to(model.device)

    # Warm up exactly as fit() would, so the model is at the same near-
    # converged state the real calibration pilot pass measures at.
    model.model._precompute_group_reg(groups, model.device)
    warmup_opt = torch.optim.Adam(model.model.parameters(), lr=model.learning_rate)
    model.model.train()
    for _ in range(100):
        loss = torch.nn.functional.mse_loss(model.model(X_tensor), y_tensor)
        warmup_opt.zero_grad(); loss.backward(); warmup_opt.step()

    a_path = [0.4, 0.6, 0.8]
    r_min, r_max = 0.01, 0.5
    lambda_min, lambda_max = model._calibrate_lambda_range(
        X_tensor, y_tensor, groups, None, a_path=a_path, r_min=r_min, r_max=r_max,
    )

    # Independently re-derive ratio_avg the same way, via a fresh set of
    # autograd calls (not reusing any internal state from the call above).
    params = list(model.model.group_parameters())
    pred = model.model(X_tensor)
    loss = torch.nn.functional.mse_loss(pred, y_tensor)
    grad_L = torch.autograd.grad(loss, params)
    grad_L_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_L))
    ratios = []
    for a_b in a_path:
        R = model.model.group_regularization(1.0, a_b, groups)
        grad_R = torch.autograd.grad(R, params)
        grad_R_norm = torch.sqrt(sum(g.pow(2).sum() for g in grad_R))
        ratios.append((grad_R_norm / grad_L_norm).item())
    ratio_avg = sum(ratios) / len(ratios)

    assert lambda_min == pytest.approx(r_min / ratio_avg, rel=1e-4)
    assert lambda_max == pytest.approx(r_max / ratio_avg, rel=1e-4)
    assert 0.0 < lambda_min < lambda_max


def test_lambda_min_max_change_sampled_range():
    """lambda_min/lambda_max (calibrate=False) should change the range
    lambda_path is drawn from, relative to the hardcoded module defaults."""
    from heteroknockoffpy.heteroknockofftorch.prismImportances import _resolve_lambda_a_path

    rng = np.random.default_rng(3)
    lp_default, _ = _resolve_lambda_a_path(None, None, rng, n_blocks=200)
    rng2 = np.random.default_rng(3)
    lp_narrow, _ = _resolve_lambda_a_path(
        None, None, rng2, n_blocks=200, lambda_min=10.0, lambda_max=20.0,
    )
    assert min(lp_narrow) >= 10.0 and max(lp_narrow) <= 20.0
    assert not (min(lp_default) >= 10.0)


def test_a_min_max_change_sampled_range():
    from heteroknockoffpy.heteroknockofftorch.prismImportances import _resolve_lambda_a_path

    rng = np.random.default_rng(3)
    _, ap_narrow = _resolve_lambda_a_path(
        None, None, rng, n_blocks=200, a_min=0.05, a_max=0.09,
    )
    assert min(ap_narrow) >= 0.05 and max(ap_narrow) <= 0.09


# ---------------------------------------------------------------------------
# Group-size normalization + zero-anchored range importance (categorical groups)
# ---------------------------------------------------------------------------

def test_group_regularization_size_normalized():
    """A K-column categorical group and a 1-column numeric singleton with the
    same per-column weight magnitude must receive the same regularization
    penalty. Before the group-size normalization fix, the K-column group's
    raw (summed, not averaged) grp_sq made it K^(a/2)x harder-penalized than
    the singleton."""
    import torch
    import torch.nn as nn
    from heteroknockoffpy.heteroknockofftorch.torchImportances import _PRISMNetworkMLP

    torch.manual_seed(0)
    a = 0.7

    singleton = _PRISMNetworkMLP(input_size=1, layers=[4], activation_class=nn.ReLU, output_size=1)
    singleton._precompute_group_reg([[0]], device='cpu')
    with torch.no_grad():
        singleton.net[0].weight.zero_()
        singleton.net[0].weight[0, 0] = 2.0
    R_singleton = singleton.group_regularization(1.0, a, [[0]])

    grouped = _PRISMNetworkMLP(input_size=2, layers=[4], activation_class=nn.ReLU, output_size=1)
    grouped._precompute_group_reg([[0, 1]], device='cpu')
    with torch.no_grad():
        grouped.net[0].weight.zero_()
        grouped.net[0].weight[0, 0] = 2.0
        grouped.net[0].weight[0, 1] = 2.0
    R_grouped = grouped.group_regularization(1.0, a, [[0, 1]])

    assert R_grouped.item() == pytest.approx(R_singleton.item(), rel=1e-4)


def test_range_importance_helper_sign_regimes():
    """_range_importance's zero-anchored spread: singleton passes through
    unchanged; an all-nonnegative group reduces to its max (min clamped to
    0); an all-nonpositive group reduces to |min| (max clamped to 0); a
    mixed-sign group is the ordinary max - min range."""
    import torch
    from heteroknockoffpy.heteroknockofftorch.torchImportances import _range_importance

    assert _range_importance(torch.tensor([-3.0])) == pytest.approx(-3.0)
    assert _range_importance(torch.tensor([1.0, 5.0, 3.0])) == pytest.approx(5.0)
    assert _range_importance(torch.tensor([-1.0, -5.0, -3.0])) == pytest.approx(5.0)
    assert _range_importance(torch.tensor([-2.0, 4.0, 1.0])) == pytest.approx(6.0)


def test_get_group_importances_categorical_collapse_method():
    """get_group_importances's categorical_collapse_method switch: 'l2_norm'
    (default) returns the Frobenius/L2 norm of the whole group's weight
    block; 'range' returns the zero-anchored max-min range over its columns'
    L2 norms instead. A numeric singleton returns its own column norm
    unchanged under either method."""
    import torch
    import torch.nn as nn
    from heteroknockoffpy.heteroknockofftorch.torchImportances import _PRISMNetworkMLP

    torch.manual_seed(0)
    model = _PRISMNetworkMLP(input_size=4, layers=[4], activation_class=nn.ReLU, output_size=1)
    groups = [[0], [1, 2, 3]]
    with torch.no_grad():
        w = model.net[0].weight  # (4, 4)
        w.zero_()
        w[0, 0] = 2.0                              # singleton column norm = 2.0
        w[0, 1], w[0, 2], w[0, 3] = 1.0, 5.0, 3.0   # categorical columns' norms

    frobenius_norm = (1.0 ** 2 + 5.0 ** 2 + 3.0 ** 2) ** 0.5

    imp_default = model.get_group_importances(groups)
    assert imp_default[0] == pytest.approx(2.0)
    assert imp_default[1] == pytest.approx(frobenius_norm)

    imp_l2 = model.get_group_importances(groups, categorical_collapse_method='l2_norm')
    assert imp_l2[0] == pytest.approx(2.0)
    assert imp_l2[1] == pytest.approx(frobenius_norm)

    imp_range = model.get_group_importances(groups, categorical_collapse_method='range')
    assert imp_range[0] == pytest.approx(2.0)
    assert imp_range[1] == pytest.approx(5.0)
    assert imp_range[1] != pytest.approx(frobenius_norm)


# ---------------------------------------------------------------------------
# Warmup best-checkpoint restoration
# ---------------------------------------------------------------------------

def test_warmup_restores_best_checkpoint_not_final_step():
    """Warmup's patience mechanism tracks a best val loss but must also
    restore the model weights at that best point, not leave the model at
    whatever (possibly-overfit) state training happens to reach at the
    n_warmup step cap. Engineers a clear overfit-then-plateau scenario (tiny
    n, wide MLP, long warmup, patience set high enough to never trigger so
    the loop always hits the step cap) and spies on _eval_loss to confirm the
    final (post-warmup) val loss equals the best one observed during
    training, not a later, worse one."""
    import torch
    from heteroknockoffpy.heteroknockofftorch.torchImportances import PRISMPredictionModel

    rng = np.random.default_rng(11)
    n, p = 40, 6
    X_np = rng.standard_normal((n, 2 * p)).astype(np.float32)
    y_np = (X_np[:, 0] + rng.standard_normal(n) * 0.1).astype(np.float32)
    groups = [[i] for i in range(2 * p)]

    val_losses: list[float] = []
    orig_eval_loss = PRISMPredictionModel._eval_loss

    def _spy_eval_loss(model, X_t, y_t, loss_fn, weight_t=None):
        result = orig_eval_loss(model, X_t, y_t, loss_fn, weight_t)
        val_losses.append((X_t.shape[0], result))
        return result

    PRISMPredictionModel._eval_loss = staticmethod(_spy_eval_loss)
    try:
        pm = PRISMPredictionModel(
            input_size=2 * p, layers=[64], model_type='mlp',
            n_warmup=300, warmup_patience=10_000, warmup_check_interval=10,
            warmup_val_frac=0.3, warmup_weight_decay=0.0,
            learning_rate=0.02, verbose=1, rng=np.random.default_rng(3),
        )
        pm.fit(X=X_np, y=y_np, groups=groups, lambda_path=None, calibrate=False)
    finally:
        PRISMPredictionModel._eval_loss = staticmethod(orig_eval_loss)

    # Chronological order: periodic val-shaped checks during the loop, then
    # (post-loop, verbose block) one full-train-shaped eval, then one final
    # val-shaped eval -- the restored model's actual val loss.
    n_val = val_losses[0][0]
    val_shaped = [v for shape, v in val_losses if shape == n_val]
    periodic_checks, final_val = val_shaped[:-1], val_shaped[-1]

    assert len(periodic_checks) >= 5, "scenario didn't run enough checks to be a meaningful test"
    assert min(periodic_checks) < periodic_checks[-1], (
        "scenario didn't overfit as engineered -- last periodic check should be "
        "worse than some earlier one, or this test doesn't actually exercise restoration"
    )
    assert final_val == pytest.approx(min(periodic_checks), rel=1e-5), (
        f"final val loss {final_val} should equal the best periodic check "
        f"{min(periodic_checks)} (restoration), not drift to a later, worse value"
    )
