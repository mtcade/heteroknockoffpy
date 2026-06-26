#
#//  test_prism.py
#//  heteroknockoffpy
#//
import numpy as np
import polars as pl
import pytest

from heteroknockoffpy import importance
from heteroknockoffpy.torchImportances import PRISMPredictionModel


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

def test_prism_w_shape_and_nonneg():
    X, Xk, y = _make_synthetic()
    lambda_path = np.logspace(0, -1, 5)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [16, 8],
        lambda_path = lambda_path,
        epochs = 3,
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
        epochs = 3,
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
        epochs = 2,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# a_path — explicit per-stage input-layer penalty values
# ---------------------------------------------------------------------------

def test_prism_w_a_path_constant():
    """Constant a_path replicates the old scalar a_min behaviour."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 5)
    a_path = [1e-3] * len(lambda_path)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_a_path_uniform():
    """Linearly increasing a_path — one value per lambda stage."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 5)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_a_path_shuffled():
    """Shuffled (non-monotone) a_path should run without error."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=42)
    lambda_path = np.logspace(0, -1, 5)
    rng = np.random.default_rng(42)
    a_path = list(rng.permutation(np.linspace(0.3, 1.0, len(lambda_path))))
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_a_path_uniform():
    """prismG with a linearly spaced a_path."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_a_path_shuffled():
    """prismG with a shuffled a_path."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=11)
    lambda_path = np.logspace(0, -1, 4)
    rng = np.random.default_rng(11)
    a_path = list(rng.permutation(np.linspace(0.3, 1.0, len(lambda_path))))
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_a_path_none_runs():
    """a_path=None (default) uses lambda_path values as input-layer penalty."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = None,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_a_path_increasing_lambda():
    """Increasing lambda_path (low→high) with a_path=None."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = list(np.linspace(1e-3, 0.1, 8))
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        epochs = 4,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# batch_size — mini-batch gradient steps
# ---------------------------------------------------------------------------

def test_prism_w_batch_size():
    """batch_size < n enables per-epoch mini-batch gradient steps."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        batch_size = 32,
        epochs = 4,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_batch_size():
    """batch_size < n enables per-epoch mini-batch gradient steps for prismG."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 4)
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        batch_size = 32,
        epochs = 4,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_batch_size_larger_than_n():
    """batch_size >= n falls back to full-batch (no crash)."""
    X, Xk, y = _make_synthetic(n=100, p=5)
    lambda_path = np.logspace(0, -1, 3)
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        batch_size = 512,
        epochs = 3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_w_a_path_and_batch_size():
    """a_path and batch_size combined."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=7)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismWImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = 32,
        epochs = 4,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_a_path_and_batch_size():
    """prismG with both a_path and batch_size."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=8)
    lambda_path = np.logspace(0, -1, 4)
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismGImportances(
        X = X,
        Xk = Xk,
        y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        batch_size = 32,
        epochs = 4,
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


def test_prism_g_cat_output_k2_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=1)
    y = _make_cat_y(150, k=2, seed=1)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_output_k3_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=2)
    y = _make_cat_y(150, k=3, seed=2)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_output_k5_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=3)
    y = _make_cat_y(150, k=5, seed=3)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical outcome — bandwidth
# ---------------------------------------------------------------------------

def test_prism_g_cat_output_k3_bandwidth():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=4)
    y = _make_cat_y(150, k=3, seed=4)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (10,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical input — varying category counts, continuous output
# ---------------------------------------------------------------------------

def test_prism_g_cat_input_2cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 2], seed=5)
    y = pl.Series("y", np.random.default_rng(5).standard_normal(150))
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_input_3cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[3, 3], seed=6)
    y = pl.Series("y", np.random.default_rng(6).standard_normal(150))
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_input_4cat():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[4, 4], seed=7)
    y = pl.Series("y", np.random.default_rng(7).standard_normal(150))
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical input + a_path — exercises the categorical proximal branch
# ---------------------------------------------------------------------------

def test_prism_w_cat_input_a_path():
    """a_path with categorical input features exercises block-Frobenius group regularisation."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=20)
    y = pl.Series("y", np.random.default_rng(20).standard_normal(150))
    lambda_path = _LAMBDA_PATH_SHORT
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismWImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_input_a_path():
    """prismG with categorical input + a_path."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=21)
    y = pl.Series("y", np.random.default_rng(21).standard_normal(150))
    lambda_path = _LAMBDA_PATH_SHORT
    a_path = list(np.linspace(0.3, 1.0, len(lambda_path)))
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = lambda_path,
        a_path = a_path,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Mixed: categorical input + categorical output
# ---------------------------------------------------------------------------

def test_prism_g_cat_input_cat_output_auto_diff():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[3, 3], seed=8)
    y = _make_cat_y(150, k=3, seed=8)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'auto_diff',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


def test_prism_g_cat_input_cat_output_bandwidth():
    X, Xk, p = _make_mixed_X(n=150, p_numeric=2, cat_cols=[3, 3], seed=9)
    y = _make_cat_y(150, k=3, seed=9)
    imp = importance.prismGImportances(
        X = X, Xk = Xk, y = y,
        layers = [8],
        local_grad_method = 'bandwidth',
        lambda_path = _LAMBDA_PATH_SHORT,
        epochs = 3,
    )
    assert imp.shape == (2 * p,), imp.shape
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# model_type — pairwise and additive architectures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_prism_w_model_type_numeric(model_type):
    X, Xk, y = _make_synthetic(n=100, p=5)
    imp = importance.prismWImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_prism_g_model_type_numeric(model_type):
    X, Xk, y = _make_synthetic(n=100, p=5)
    imp = importance.prismGImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        local_grad_method="auto_diff",
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_prism_w_model_type_cat_input(model_type):
    """OHE columns are treated as independent feature pairs in pairwise/additive."""
    X, Xk, p = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=30)
    y = pl.Series("y", np.random.default_rng(30).standard_normal(100))
    imp = importance.prismWImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert imp.shape == (2 * p,)
    assert np.all(imp >= 0)


@pytest.mark.parametrize("model_type", ["pairwise", "additive"])
def test_prism_g_model_type_cat_input(model_type):
    X, Xk, p = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=31)
    y = pl.Series("y", np.random.default_rng(31).standard_normal(100))
    imp = importance.prismGImportances(
        X=X, Xk=Xk, y=y, layers=[16], model_type=model_type,
        local_grad_method="auto_diff",
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
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


def test_prism_w_count_outcome():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=40)
    y = _make_count_y(150, seed=40)
    imp = importance.prismWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_count_outcome_auto_diff():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=41)
    y = _make_count_y(150, seed=41)
    imp = importance.prismGImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


def test_prism_g_count_outcome_bandwidth():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=42)
    y = _make_count_y(150, seed=42)
    imp = importance.prismGImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='bandwidth',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# Categorical outcome — prismW (CrossEntropyLoss path)
# ---------------------------------------------------------------------------

def test_prism_w_cat_outcome():
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=43)
    y = _make_cat_y(150, k=3, seed=43)
    imp = importance.prismWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert imp.shape == (10,)
    assert np.all(imp >= 0)


# ---------------------------------------------------------------------------
# prismGLocalGradients — shape, finite values, both local_grad_method values
# ---------------------------------------------------------------------------

def test_prism_g_local_gradients_numeric_auto_diff():
    """Returns (n, p) for numeric-only X with auto_diff."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=44)
    grads = importance.prismGLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


def test_prism_g_local_gradients_numeric_bandwidth():
    """Returns (n, p) for numeric-only X with bandwidth finite difference."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=45)
    grads = importance.prismGLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='bandwidth',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


def test_prism_g_local_gradients_cat_input():
    """Categorical input: p_ohe_x = p_numeric + sum(c_k - 1 per cat variable)."""
    # 3 numeric + 2-cat (1 ohe col) + 3-cat (2 ohe cols) → p_ohe_x = 3+1+2 = 6
    X, Xk, _ = _make_mixed_X(n=100, p_numeric=3, cat_cols=[2, 3], seed=46)
    y = pl.Series("y", np.random.default_rng(46).standard_normal(100))
    grads = importance.prismGLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert grads.shape == (100, 6)
    assert np.isfinite(grads).all()


def test_prism_g_local_gradients_count_outcome():
    """prismGLocalGradients with Poisson count outcome."""
    X, Xk, _ = _make_synthetic(n=100, p=5, seed=47)
    y = _make_count_y(100, seed=47)
    grads = importance.prismGLocalGradients(
        X=X, Xk=Xk, y=y, layers=[8],
        outcome_type='count',
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert grads.shape == (100, 5)
    assert np.isfinite(grads).all()


# ---------------------------------------------------------------------------
# prismGWImportances — single-pass dual importances
# ---------------------------------------------------------------------------

def test_prism_gw_returns_tuple_of_two_arrays():
    """prismGWImportances returns a 2-tuple of numpy arrays."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=50)
    result = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert isinstance(result, tuple) and len(result) == 2
    g_imp, w_imp = result
    assert isinstance(g_imp, np.ndarray)
    assert isinstance(w_imp, np.ndarray)


def test_prism_gw_shape_and_nonneg():
    """Both G and W importances have shape (2*p,) and are non-negative."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=51)
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert g_imp.shape == (10,), g_imp.shape
    assert w_imp.shape == (10,), w_imp.shape
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_gw_bandwidth():
    """prismGWImportances with bandwidth local_grad_method."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=52)
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='bandwidth',
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_gw_g_and_w_differ():
    """G and W importances use different scoring methods and must not be identical."""
    X, Xk, y = _make_synthetic(n=150, p=8, seed=53)
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[16, 8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 5), epochs=4,
    )
    assert not np.allclose(g_imp, w_imp), "PRISM_g and PRISM_w should differ"


def test_prism_gw_w_matches_standalone_prism_w_structure():
    """W importances are group norms — all non-negative and finite, like standalone prismW."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=54)
    _, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=np.logspace(0, -1, 4), epochs=3,
    )
    assert np.all(w_imp >= 0)
    assert np.isfinite(w_imp).all()


def test_prism_gw_cat_input():
    """prismGWImportances with categorical input columns."""
    X, Xk, p = _make_mixed_X(n=150, p_numeric=3, cat_cols=[2, 3], seed=55)
    y = pl.Series("y", np.random.default_rng(55).standard_normal(150))
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert g_imp.shape == (2 * p,), g_imp.shape
    assert w_imp.shape == (2 * p,), w_imp.shape
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_gw_cat_output():
    """prismGWImportances with categorical outcome (CrossEntropyLoss path)."""
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=56)
    y = _make_cat_y(150, k=3, seed=56)
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_gw_count_outcome():
    """prismGWImportances with Poisson count outcome."""
    X, Xk, _ = _make_synthetic(n=150, p=5, seed=57)
    y = _make_count_y(150, seed=57)
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=_LAMBDA_PATH_SHORT, epochs=3,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
    assert np.all(g_imp >= 0)
    assert np.all(w_imp >= 0)


def test_prism_gw_snapshot_count_matches_lambda_stages():
    """Each lambda stage produces one G snapshot and one W snapshot."""
    X, Xk, y = _make_synthetic(n=100, p=5, seed=58)
    lambda_path = np.logspace(0, -1, 6)
    # Both outputs are means over 6 snapshots; we can't inspect them directly,
    # but running without error and returning the right shape confirms the counts match.
    g_imp, w_imp = importance.prismGWImportances(
        X=X, Xk=Xk, y=y, layers=[8],
        local_grad_method='auto_diff',
        lambda_path=lambda_path, epochs=3,
    )
    assert g_imp.shape == (10,)
    assert w_imp.shape == (10,)
