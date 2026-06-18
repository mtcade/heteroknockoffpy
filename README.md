# heteroknockoffpy

Knockoffs and PRISM importances for heterogeneous (mixed numeric/categorical) data, using conditional residuals and random forests.

---

## Installation

```
pip install heteroknockoffpy
```

`categorical_method='forest'` requires R and `rpy2`. Install the `ranger` R package before using it.

---

## Knockoffs

```python
from heteroknockoffpy import knockoff
import numpy as np

rng = np.random.default_rng(0)

# X is a polars DataFrame; categorical columns must have dtype pl.Categorical
Xk = knockoff.get_knockoffs(
    X,
    method="second_order",   # "second_order" | "GAN" | "GAN_torch" | "SCIP"
    rng=rng,
    categorical_method="forest",
)
```

### `categorical_method`

Controls how categorical columns are encoded before knockoffs are generated.

| value | behavior |
|---|---|
| `'forest'` | Fits a ranger random forest per categorical column; uses predicted class-probability logits as a soft numeric encoding |
| `'linear'` | Same, but with logistic regression — lighter and faster |
| `'ohe'` | Hard one-hot-encodes categories as floats; no probability smoothing |
| `'scip'` | For numeric columns, operates on conditional residuals `X_j − E[X_j | X_{-j}]` so knockoffs respect the joint distribution; for categorical columns uses forest-SCIP |

`'scip'` is the most statistically principled approach for mixed data. `'forest'` or `'linear'` are convenient defaults when a quick approximation is acceptable.

### `conditional_expectations`

A `pl.DataFrame` of shape `(n, p_numeric)` giving `E[X_j | X_{-j}]` for each numeric column. Only relevant when `categorical_method='scip'`.

```python
from heteroknockoffpy import rbridge

# compute once, reuse across multiple knockoff draws
ce = rbridge.get_forest_conditional_expectations(X)

Xk1 = knockoff.get_knockoffs(X, method="GAN", rng=rng,
                               categorical_method="scip",
                               conditional_expectations=ce)
Xk2 = knockoff.get_knockoffs(X, method="GAN", rng=rng,
                               categorical_method="scip",
                               conditional_expectations=ce)
```

If `conditional_expectations=None` (the default) and `categorical_method='scip'`, the package computes them internally using R `ranger::ranger`. Pass a pre-computed frame to avoid refitting the forest on every call.

---

## Importances

All importance functions return a `np.ndarray` of length `2p` — scores for `[x_1, …, x_p, x̃_1, …, x̃_p]`. Use `wFromImportances` to convert these to knockoff W-statistics for variable selection.

Both PRISM functions train a single MLP on `[X, Xk]` while sweeping a lambda regularization path. At the end of each lambda stage a snapshot of importances is recorded; the returned importances are the mean across all snapshots. The regularization path defaults to `logspace(1, -2, 50)`; pass `lambda_path` and/or `a_path` to override. `epochs` is distributed evenly across stages.

### PRISM-W — `prismWImportances`

Records first-layer column norms `‖W[:,j]‖₂` at each lambda stage. Fast — no extra forward passes per snapshot.

```python
from heteroknockoffpy.importance import prismWImportances

imp = prismWImportances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",   # "continuous" | "count" | "categorical" | None (inferred)
    model_type="mlp",            # "mlp" | "pairwise" | "additive"
    epochs=500,
)
```

### PRISM-G — `prismGImportances`

Records per-feature output sensitivity `φⱼ = mean|ŷ(x+σeⱼ) − ŷ(x−σeⱼ)| / 2σ` at each lambda stage. More directly tied to the model's predictions than PRISM-W, but requires extra forward passes per snapshot.

```python
from heteroknockoffpy.importance import prismGImportances

imp = prismGImportances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",
    local_grad_method="auto_diff",  # "auto_diff" | "bandwidth"
    bandwidth=None,                 # only used when local_grad_method="bandwidth"
    model_type="mlp",
    epochs=500,
)
```

### `model_type`

| value | behavior |
|---|---|
| `'mlp'` | Standard MLP on `[X, Xk]` with group regularization on first-layer columns |
| `'pairwise'` | Adds a learnable filter that creates convex combinations of `xⱼ` and `x̃ⱼ`, forcing explicit per-feature competition |
| `'additive'` | Feature-wise sub-networks with separate group regularization for `X` and `Xk` channels |

---

## Variable selection

```python
from heteroknockoffpy.importance import wFromImportances, selection_threshold

W = wFromImportances(imp)
threshold = selection_threshold(W, fdr=0.1)
selected = [j for j, w in enumerate(W) if w >= threshold]
```
