# heteroknockoffpy

Knockoffs and importance measures for heterogeneous (mixed numeric/categorical) data, using conditional residuals and random forests.

Based on the knockoff filter framework ([Candès et al., 2018](https://academic.oup.com/jrsssb/article/80/3/551/7048447)).

---

## Installation

```
pip install heteroknockoffpy
```

`categorical_method='forest'` and `method='SCIP'` require R and `rpy2`. Install the `ranger` and `rangerKnockoff` R packages before using them.

---

## Knockoffs

```python
from heteroknockoffpy import knockoff
import numpy as np

rng = np.random.default_rng(0)

# X is a polars DataFrame; categorical columns must have dtype pl.Categorical
Xk = knockoff.get_knockoffs(
    X,
    method="second_order",   # "second_order" | "GAN_torch" | "SCIP"
    rng=rng,
    categorical_method="forest",
)
```

### `method`

| value | behavior |
|---|---|
| `'second_order'` | Matches the first two moments (mean and covariance) of X. Fast and closed-form via the R `knockoff` package. Works well when the joint distribution is approximately Gaussian; may lose power in strongly non-linear settings. |
| `'GAN_torch'` | Trains a GAN in PyTorch to learn the full joint distribution of X and generate knockoffs that are indistinguishable from it. Slower than second-order but can capture non-Gaussian and non-linear dependence structures. |
| `'SCIP'` | Sorted L1 Penalized Inference knockoffs via the `rangerKnockoff` R package. Fits a ranger random forest per column to estimate conditional distributions, then generates knockoffs from those conditional models. The most statistically principled method for non-parametric joint distributions. |

### `categorical_method`

Controls how categorical columns are encoded before knockoffs are generated. Not applicable when `method='SCIP'` (which handles categoricals natively).

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

Xk1 = knockoff.get_knockoffs(X, method="GAN_torch", rng=rng,
                               categorical_method="scip",
                               conditional_expectations=ce)
Xk2 = knockoff.get_knockoffs(X, method="GAN_torch", rng=rng,
                               categorical_method="scip",
                               conditional_expectations=ce)
```

If `conditional_expectations=None` (the default) and `categorical_method='scip'`, the package computes them internally using R `ranger::ranger`. Pass a pre-computed frame to avoid refitting the forest on every call.

---

## Importances

All importance functions return a `np.ndarray` of length `2p` — scores for `[x_1, …, x_p, x̃_1, …, x̃_p]`. Use `wFromImportances` to convert these to knockoff W-statistics for variable selection.

### PRISM-W — `prismWImportances`

Trains a single MLP on `[X, Xk]` while sweeping a lambda regularization path. Records first-layer column norms `‖W[:,j]‖₂` at the end of each lambda stage; the returned importances are the mean across all snapshots. Fast — no extra forward passes per snapshot.

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

Same training procedure as PRISM-W. Records per-feature output sensitivity `φⱼ = mean|ŷ(x+σeⱼ) − ŷ(x−σeⱼ)| / 2σ` at each lambda stage. More directly tied to the model's predictions than PRISM-W, but requires extra forward passes per snapshot.

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

The regularization path defaults to `logspace(1, -2, 50)`; pass `lambda_path` and/or `a_path` to override. `epochs` is distributed evenly across stages.

### Lasso — `lassoImportances`

Fits a penalized linear model on `[X, Xk]` and uses absolute coefficient values as importances. Cross-validates the regularization strength automatically. Fast and interpretable; best when the outcome-feature relationship is approximately linear.

- continuous: `LassoCV` (sklearn)
- count: `PoissonLassoCV` (L1-penalized Poisson GLM)
- categorical: `LogisticRegressionCV` with L1 penalty (SAGA solver)

```python
from heteroknockoffpy.importance import lassoImportances

imp = lassoImportances(X=X, Xk=Xk, y=y, outcome_type="continuous")
```

### Ridge — `ridgeImportances`

Same as `lassoImportances` but with L2 regularization. Coefficients are shrunk but not zeroed, so all features retain non-zero importance. Useful when many features are expected to have small true effects.

- continuous: `RidgeCV` (sklearn)
- count: `PoissonRegressor` via `GridSearchCV` (neg Poisson deviance scoring)
- categorical: `LogisticRegressionCV` with L2 penalty (LBFGS solver)

```python
from heteroknockoffpy.importance import ridgeImportances

imp = ridgeImportances(X=X, Xk=Xk, y=y, outcome_type="continuous")
```

### Elastic Net — `elasticImportances`

Interpolates between Lasso and Ridge via `l1_ratio` (0 = Ridge, 1 = Lasso). Useful when there are groups of correlated features — the L2 component keeps them together while L1 performs selection.

- continuous: `ElasticNetCV` (sklearn)
- count: `PoissonLassoCV` with `L1_wt=l1_ratio`
- categorical: `LogisticRegressionCV` with `penalty='elasticnet'`

```python
from heteroknockoffpy.importance import elasticImportances

imp = elasticImportances(X=X, Xk=Xk, y=y, outcome_type="continuous", l1_ratio=0.5)
```

Delegates to `lassoImportances` when `l1_ratio=1` and to `ridgeImportances` when `l1_ratio=0`.

### Random Forest Gini — `rangerGiniImportances`

Fits a ranger random forest on `[X, Xk]` and returns variable importances based on mean decrease in node impurity (Gini importance). Non-parametric and robust to non-linearities; no hyperparameter tuning required.

```python
from heteroknockoffpy.importance import rangerGiniImportances

imp = rangerGiniImportances(X=X, Xk=Xk, y=y, outcome_type="continuous")
```

Requires R and `rpy2` with the `ranger` package installed.

### Random Forest MALD — `rangerMaldImportances`

Fits a ranger random forest on `[X, Xk]` and computes Mean Absolute Local Derivative (MALD) importances: for each predictor column, the pointwise sensitivity of the forest's prediction is measured and averaged. This captures how much the output changes when a variable is perturbed, unlike Gini which aggregates split quality.

Predictor handling is native — numeric columns use a bandwidth finite-difference, factor columns sweep over all levels (max-minus-min) — so **no one-hot encoding is applied at any outcome type**.

| outcome type | sensitivity measure |
|---|---|
| `'continuous'` | finite difference of predicted value |
| `'count'` | finite difference of log predicted value |
| `'categorical'` | Mahalanobis norm of log-probability contrasts vs. first class |

Returns a `(2p,)` array of raw importances — first `p` for `X`, last `p` for `Xk`. Pass through `wFromImportances` for knockoff W-statistics.

```python
from heteroknockoffpy.importance import rangerMaldImportances, wFromImportances

imp = rangerMaldImportances(X=X, Xk=Xk, y=y, outcome_type="continuous")
W   = wFromImportances(imp)
```

Accepts any extra keyword arguments (e.g. `bandwidth`, `exponent`, `num_trees`) which are forwarded to `ranger::ranger` or the MALD script. Requires R and `rpy2` with the `ranger` package installed.

### `model_type` (PRISM only)

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
