# heteroknockoffpy

Knockoffs and MALD importances for heterogeneous (mixed numeric/categorical) data, using conditional residuals and random forests.

---

## Installation

```
pip install heteroknockoffpy
```

Ranger-based methods require R and `rpy2`. Install the `ranger` R package before using `rangerMaldImportances` or `categorical_method='forest'`.

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

### PyTorch / TF model — `maldImportancesFromModel`

```python
from heteroknockoffpy.importance import maldImportancesFromModel

imp = maldImportancesFromModel(
    model=my_torch_model,   # any PredictionModel with autodiff support
    X=X,
    Xk=Xk,
    y=y,
    outcome_type="continuous",       # "continuous" | "count" | "categorical" | None (inferred)
    local_grad_method="auto_diff",   # "auto_diff" | "bandwidth"
    bandwidth=None,                  # only used if local_grad_method="bandwidth"
    exponent=1.0,
)
```

### Built-in feedforward net — `torchMaldImportances`

```python
from heteroknockoffpy.importance import torchMaldImportances

imp = torchMaldImportances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    epochs=500,
)
```

### R ranger forest — `rangerMaldImportances`

```python
from heteroknockoffpy.rbridge import rangerMaldImportances

imp = rangerMaldImportances(
    X=X, Xk=Xk, y=y,
    outcome_type="continuous",
    bandwidth=1.0,
    bandwidth_exponent=0.2,
)
```

### Bandwidth parameters

MALD approximates `∂ŷ/∂x_j` by finite difference:

```
(ŷ(x + h·eⱼ) − ŷ(x)) / h
```

- **`bandwidth`** — the absolute step size `h`. Smaller values give a more local derivative at the cost of higher variance; larger values smooth more but can miss sharp nonlinearities. When `local_grad_method='bandwidth'` in `maldImportancesFromModel`, `None` defaults to `n^{-0.2}`.

- **`bandwidth_exponent`** — used by `rangerMaldImportances` to scale `h` relative to sample size: effective `h = bandwidth × n^{-bandwidth_exponent}`. The default `0.2` implements the standard nonparametric `n^{-1/5}` rule.

When autodiff is available (`local_grad_method='auto_diff'`), both bandwidth parameters are ignored and the exact gradient is used instead.

---

## Variable selection

```python
from heteroknockoffpy.importance import wFromImportances, selection_threshold

W = wFromImportances(imp)
threshold = selection_threshold(W, fdr=0.1)
selected = [j for j, w in enumerate(W) if w >= threshold]
```
