# heteroknockoffpy

Knockoffs and importance measures for heterogeneous (mixed numeric/categorical) data, using conditional residuals and random forests.

Based on the knockoff filter framework ([Candès et al., 2018](https://academic.oup.com/jrsssb/article/80/3/551/7048447)).

---

## Installation

```
pip install heteroknockoffpy
```

`categorical_method='forest'`, `categorical_method='ranger_scip'`, and `method='ranger_SCIP'` require R and `rpy2`. Install the `ranger` and `rangerKnockoff` R packages before using them. `categorical_method='xgb_scip'` and `method='xgb_SCIP'` use `xgboost` instead and don't require R.

On macOS, `xgboost` requires OpenMP:

```
brew install libomp
```

### macOS: torch + xgboost run in separate processes automatically

`torch` and `xgboost` each load their own copy of `libomp`, and having both in one process
crashes on macOS. The public API handles this transparently: any `heteroknockoffpy.importance`
`xgb*`/`prism*` call, and `knockoff.get_knockoffs(method="torch_GAN")` / `method="xgb_SCIP"` /
`categorical_method="xgb_scip"`, automatically runs in an isolated subprocess whenever it would
introduce the second library into a process that already has the other loaded — so mixing them
in the same script or notebook just works, with no extra setup. Because this can spawn a
subprocess, guard top-level driver code with `if __name__ == "__main__":`, per normal Python
`multiprocessing` requirements.

(If you bypass the public API and import `heteroknockoffpy.heteroknockofftorch.torchImportances`
and `heteroknockoffpy.xgbImportances` directly yourself, this isolation doesn't apply —
`./scripts/fix_xgboost_omp.sh` is available as a fallback for that case.)

---

## Knockoffs

Every knockoff function shares this common core of inputs:

- **`X`** — Original data (numeric + `pl.Categorical` columns).
- **`rng`** — `np.random.Generator` seeding the method's randomness (R/xgboost/OHE-sampling downstream, and, for `second_order`, R's own RNG).
- **`categorical_method`** (`get_second_order`/`get_torchGAN` only) — How categorical columns are encoded before knockoffs are generated: `'forest'`/`'linear'`/`'ohe'`/`'ranger_scip'`/`'xgb_scip'`. See the `categorical_method` table below. Not a parameter of `get_rangerSCIP`/`get_xgbSCIP`, which handle categoricals natively.
- **`conditional_expectations`** — `pl.DataFrame` of `E[X_j | X_{-j}]` per numeric column; only relevant when `categorical_method` is `'ranger_scip'`/`'xgb_scip'`. Computed internally if omitted. See `conditional_expectations` below.
- **`verbose`** / **`verbose_prefix`** — Verbosity level (`0` = silent) and a string prefix for any verbose print output, useful when nesting calls inside a higher-level loop.
- **`weight`** — Optional length-`n` sample weight. `None` (default) fits unweighted. Forwarded to whichever step actually fits something: `'second_order'` uses it as a real weighted mean/covariance; `'forest'`/`'linear'`/`'ranger_scip'`/`'xgb_scip'` (as a `categorical_method` or as `method` itself) use it as native `case.weights`/`sample_weight`; `'ohe'` ignores it (no fit on that path); `'torch_GAN'` accepts it only for signature consistency — the GAN itself is unsupervised and does not use it, though it's still passed through to `categorical_method`'s own fit.

```python
from heteroknockoffpy import knockoff
import numpy as np

rng = np.random.default_rng(0)

# X is a polars DataFrame; categorical columns must have dtype pl.Categorical
Xk = knockoff.get_knockoffs(
    X,
    method="second_order",   # "second_order" | "torch_GAN" | "ranger_SCIP" | "xgb_SCIP"
    rng=rng,
    categorical_method="forest",
)
```

### `method`

| value | behavior |
|---|---|
| `'second_order'` | Matches the first two moments (mean and covariance) of X. Fast and closed-form via the R `knockoff` package. Works well when the joint distribution is approximately Gaussian; may lose power in strongly non-linear settings. |
| `'torch_GAN'` | Trains a GAN in PyTorch to learn the full joint distribution of X and generate knockoffs that are indistinguishable from it. Slower than second-order but can capture non-Gaussian and non-linear dependence structures. |
| `'ranger_SCIP'` | Sequential Conditional Independence Procedure knockoffs via the `rangerKnockoff`/`ranger` R packages. Fits a ranger random forest per column (conditioning on all original columns plus all previously generated knockoffs) to estimate conditional distributions, then generates knockoffs from those conditional models. The most statistically principled method for non-parametric joint distributions. |
| `'xgb_SCIP'` | Same SCIP algorithm as `'ranger_SCIP'`, but each column's conditional model is a sequential [`xgboost.XGBRegressor`/`XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html) instead of an R ranger forest — no R/`rpy2` dependency. See `heteroknockoffpy.xgbScip` for the tuned `model_kwargs` typically used (`max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`, `n_estimators`). |

### `categorical_method`

Controls how categorical columns are encoded before knockoffs are generated. Not applicable when `method` is `'ranger_SCIP'` or `'xgb_SCIP'` (which handle categoricals natively).

| value | behavior |
|---|---|
| `'forest'` | Fits a ranger random forest per categorical column; uses predicted class-probability logits as a soft numeric encoding |
| `'linear'` | Same, but with logistic regression — lighter and faster |
| `'ohe'` | Hard one-hot-encodes categories as floats; no probability smoothing |
| `'ranger_scip'` | For numeric columns, operates on conditional residuals `X_j − E[X_j | X_{-j}]` so knockoffs respect the joint distribution; for categorical columns uses ranger forest-SCIP |
| `'xgb_scip'` | Same as `'ranger_scip'`, but conditional expectations and categorical SCIP knockoffs are computed with sequential [`xgboost`](https://xgboost.readthedocs.io/en/latest/python/python_api.html) models instead of R ranger — no R/`rpy2` dependency |

`'ranger_scip'`/`'xgb_scip'` are the most statistically principled approaches for mixed data. `'forest'` or `'linear'` are convenient defaults when a quick approximation is acceptable.

### `conditional_expectations`

A `pl.DataFrame` of shape `(n, p_numeric)` giving `E[X_j | X_{-j}]` for each numeric column. Only relevant when `categorical_method` is `'ranger_scip'` or `'xgb_scip'`.

```python
from heteroknockoffpy import rbridge

# compute once, reuse across multiple knockoff draws
ce = rbridge.get_forest_conditional_expectations(X)

Xk1 = knockoff.get_knockoffs(X, method="torch_GAN", rng=rng,
                               categorical_method="ranger_scip",
                               conditional_expectations=ce)
Xk2 = knockoff.get_knockoffs(X, method="torch_GAN", rng=rng,
                               categorical_method="ranger_scip",
                               conditional_expectations=ce)
```

If `conditional_expectations=None` (the default) and `categorical_method='ranger_scip'`, the package computes them internally using R `ranger::ranger`
(`rbridge.get_forest_conditional_expectations`). For `categorical_method='xgb_scip'`, it uses sequential `xgboost.XGBRegressor` models instead
(`xgbScip.get_forest_conditional_expectations`). Pass a pre-computed frame to avoid refitting on every call.

---

## Importances

Every importance function shares this common core of inputs:

- **`X`** / **`Xk`** — Original data and its knockoffs (same schema, numeric + `pl.Categorical` columns).
- **`y`** — Outcome; scalar (continuous/count) or categorical Series/DataFrame.
- **`outcome_type`** — `'continuous'`/`'count'`/`'categorical'`; inferred from `y` if omitted.
- **`verbose`** — Verbosity level (`0` = silent).
- **`weight`** — Optional length-`n` sample weight. `None` (default) fits unweighted. For the PRISM-torch family (`prismWImportances`/`prismGImportances`/`prismGWImportances`/`prismWImportancesPerOHE`), it's a per-sample multiplier on the training loss. For `rangerGiniImportances`/`rangerPrismImportances`, it's forwarded to `ranger::ranger`'s `case.weights` (resampling-probability weighting, not a loss multiplier). For `lassoImportances`/`ridgeImportances`/`elasticImportances`, it's forwarded as each underlying sklearn/statsmodels model's `sample_weight`. For the xgboost-based functions (`xgbImportances`/`xgbPrismImportances`/`xgbShapImportances`), it's xgboost's own native `sample_weight`.

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

### PRISM-GW — `prismGWImportances`

Trains the same model as PRISM-G and PRISM-W but in a single pass, producing both sets of importances simultaneously. At each lambda stage the snapshot closure captures PRISM-W group norms as a side effect while returning PRISM-G local-gradient importances as the primary snapshot. Roughly halves the compute cost of running both methods separately.

Returns a 2-tuple `(g_importances, w_importances)`, each of shape `(2p,)`.

```python
from heteroknockoffpy.importance import prismGWImportances

g_imp, w_imp = prismGWImportances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",
    local_grad_method="bandwidth",  # "auto_diff" | "bandwidth"
    epochs=500,
)
```

All parameters are identical to `prismGImportances`. `local_grad_method` is required (it governs the G snapshot; the W snapshot uses group norms and needs no gradient method).

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

### Random Forest PRISM — `rangerPrismImportances`

Fits a ranger random forest on `[X, Xk]` and computes PRISM importances: for each predictor column, the pointwise sensitivity of the forest's prediction is measured and averaged. This captures how much the output changes when a variable is perturbed, unlike Gini which aggregates split quality.

Predictor handling is native — numeric columns use a bandwidth finite-difference, factor columns sweep over all levels (max-minus-min) — so **no one-hot encoding is applied at any outcome type**.

| outcome type | sensitivity measure |
|---|---|
| `'continuous'` | finite difference of predicted value |
| `'count'` | finite difference of log predicted value |
| `'categorical'` | Mahalanobis norm of log-probability contrasts vs. first class |

Returns a `(2p,)` array of raw importances — first `p` for `X`, last `p` for `Xk`. Pass through `wFromImportances` for knockoff W-statistics.

```python
from heteroknockoffpy.importance import rangerPrismImportances, wFromImportances

imp = rangerPrismImportances(X=X, Xk=Xk, y=y, outcome_type="continuous")
W   = wFromImportances(imp)
```

Accepts any extra keyword arguments (e.g. `bandwidth`, `exponent`, `num_trees`) which are forwarded to `ranger::ranger` or the PRISM script. Requires R and `rpy2` with the `ranger` package installed.

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
