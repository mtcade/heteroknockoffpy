# heteroknockoffpy

Knockoffs and importance measures for heterogeneous (mixed numeric/categorical) data

Based on the knockoff filter framework ([Candès et al., 2018](https://academic.oup.com/jrsssb/article/80/3/551/7048447)).

---

## Installation

```
pip install heteroknockoffpy
```

Some features need a working R installation (`rpy2` is installed as a dependency, but R itself is not):

| needs R | doesn't need R |
|---|---|
| `method='second_order'` (R `knockoff`), `method='ranger_SCIP'`, `categorical_method='ranger'`/`'ranger_scip'`, `rangerGiniImportances`, `rangerPrismImportances` (R `ranger`) | `method='torch_GAN'`/`'xgb_SCIP'`/`'glm_SCIP'`, `categorical_method='xgb'`/`'xgb_scip'`/`'linear'`/`'ohe'`, every other importance function |

The R packages `knockoff`, `ranger`, and `arrow` are installed from CRAN automatically the first time an R-backed feature is
used, if they're missing. `xgbShapImportances` needs the optional `shap` extra:

```
pip install 'heteroknockoffpy[shap]'
```

On macOS, `xgboost` requires OpenMP:

```
brew install libomp
```

### macOS: torch, xgboost, and R run in separate processes automatically

`torch` and `xgboost` each bundle their own `libomp`, and R (via `rpy2`) embeds its own native runtime; loading any two of
them into one process can crash on macOS. The public API handles this for you. Every `knockoff` and `importance` call that
uses torch, xgboost, or R runs in a freshly spawned subprocess whenever one of the *other* two is already loaded in the
current process, so you can mix them in the same script or notebook with no extra setup. Because this can spawn a
subprocess, guard top-level driver code with `if __name__ == "__main__":`, as Python `multiprocessing` normally requires.

If you bypass the public API and import `heteroknockoffpy.rbridge`, `heteroknockoffpy.xgbScip`,
`heteroknockoffpy.xgbImportances`, or `heteroknockoffpy.heteroknockofftorch.*` directly, this isolation doesn't apply.

---

## Knockoffs

`knockoff.get_knockoffs(X, method, rng, ...)` picks a generator by name and forwards everything else to it. You can also call
each generator directly (`get_second_order`, `get_torchGAN`, `get_rangerSCIP`, `get_xgbSCIP`, `get_glmSCIP`).

```python
from heteroknockoffpy import knockoff
import numpy as np

rng = np.random.default_rng(0)

# X is a polars DataFrame; categorical columns must have dtype pl.Categorical
Xk = knockoff.get_knockoffs(
    X,
    method="second_order",        # required -- see `method` below
    rng=rng,
    categorical_method="ranger",  # required for second_order/torch_GAN -- see `categorical_method` below
)
```

Every generator takes these inputs:

- **`X`**: original data (numeric + `pl.Categorical` columns).
- **`rng`**: `np.random.Generator` that seeds all of the method's randomness (Python, xgboost, torch, and R's own RNG via `set.seed`).
- **`verbose`** / **`verbose_prefix`**: verbosity level (`0` = silent) and a string prefix for verbose output, useful when nesting calls inside a higher-level loop.
- **`weight`**: optional length-`n` sample weight. `None` (default) fits unweighted. It goes to whichever step actually fits a model:
  - `'second_order'` computes a weighted mean and covariance. `shrink` is not honored on the weighted path; a small diagonal ridge is added instead if the weighted covariance isn't positive-definite.
  - `'ranger'`/`'xgb'`/`'linear'`/`'ranger_scip'`/`'xgb_scip'` (as a `categorical_method`) and `'ranger_SCIP'`/`'xgb_SCIP'`/`'glm_SCIP'` (as a `method`) pass it as native `case.weights`/`sample_weight`.
  - `'ohe'` ignores it, because nothing is fit on that path.
  - `'torch_GAN'` accepts it only for signature consistency. The GAN is unsupervised and doesn't use it, but it is still passed to `categorical_method`'s own fit.

Which other options apply depends on `method`:

| `method` | `categorical_method` | `conditional_expectations` | `residuals_method` | categorical columns |
|---|---|---|---|---|
| `'second_order'` | **required** | used only with `'ranger_scip'`/`'xgb_scip'` | not accepted (R raises `unused argument`) | via `categorical_method` |
| `'torch_GAN'` | **required** | used only with `'ranger_scip'`/`'xgb_scip'` | ignored | via `categorical_method` |
| `'ranger_SCIP'` | not used (ignored, with a warning) | ignored | `'normal'` (default) / `'permute'` | native |
| `'xgb_SCIP'` | not used (ignored, with a warning) | ignored | `'normal'` (default) / `'permute'` / `'zero_inflated'` | native |
| `'glm_SCIP'` | not accepted (`TypeError`) | ignored | not accepted (always zero-inflated) | **not supported** (`ValueError`) |

### `method`

Required; there is no default. `'second_order'` and `'torch_GAN'` generate knockoffs for a *numeric* array, so they need
`categorical_method` to say how categorical columns get in and out of that array. The `*_SCIP` methods model every column
one at a time and handle categoricals themselves.

| value | behavior |
|---|---|
| `'second_order'` | Matches the first two moments (mean and covariance) of the numeric array, using R `knockoff::create.second_order`. Fast and closed-form. Works well when the joint distribution is roughly Gaussian; may lose power in strongly non-linear settings. Extra kwargs: `shrink` (default `True`, shrinks the covariance estimate; worth setting explicitly when `p` is large relative to `n`) and `method` (R's `create.second_order` method, default `'asdp'`). Requires R. |
| `'torch_GAN'` | Trains a GAN in PyTorch to learn the full joint distribution of the numeric array and generate knockoffs from it. Slower than `'second_order'`, but it can capture non-Gaussian and non-linear dependence. Extra kwargs: `niter` (default `2000`), `mb_size` (`128`), `lr` (`1e-4`), `lamda` (`1`), `mu` (`1`), `lam` (`10`), `combined_inner` (`False`); unrecognized kwargs are ignored. |
| `'ranger_SCIP'` | Sequential Conditional Independence Pairs (SCIP) knockoffs with R `ranger` forests. Each column's knockoff comes from a forest that conditions on all original columns plus every knockoff generated so far: a probability forest for categorical columns, and a regression forest plus a resampled residual (`residuals_method`) for numeric columns. Extra kwargs go to `ranger::ranger` (`num_trees`, `mtry`, `min_node_size`, `max_depth`, `sample_fraction`, `num_threads`, `respect_unordered_factors`). Requires R. |
| `'xgb_SCIP'` | The same SCIP algorithm as `'ranger_SCIP'`, but each column's model is an [`xgboost.XGBRegressor`/`XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html). No R/`rpy2` dependency. Pass xgboost hyperparameters directly as kwargs (`max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`, `n_estimators`). |
| `'glm_SCIP'` | SCIP for sparse/zero-inflated, **all-numeric** `X` using regularized linear GLMs: a logistic model for `P(x_j ≠ 0)` plus a Gamma GLM (`nonzero_model='gamma'`, default) or a log-normal Ridge model (`nonzero_model='exponential'`, with `clipped=True` by default) for the nonzero part. Extra kwargs: `nonzero_model`, `clipped`, `logistic_kwargs`, `nonzero_kwargs`. Raises `ValueError` if `X` has any categorical column. |

### `residuals_method` (`'ranger_SCIP'`/`'xgb_SCIP'` only)

How each numeric column's knockoff residual is drawn around its fitted conditional mean. Categorical columns are unaffected.

| value | behavior |
|---|---|
| `'normal'` (default) | Residual drawn from `N(0, sd(residuals))` |
| `'permute'` | Residual is a random permutation of the observed residuals |
| `'zero_inflated'` | `'xgb_SCIP'` only (`'ranger_SCIP'` raises `AssertionError`). For sparse numeric columns: a two-part `P(x_j ≠ 0)` classifier plus `E[x_j | x_j ≠ 0]` regressor; the draw is a Bernoulli gate around a normal-perturbed nonzero value |

### `categorical_method` (`'second_order'`/`'torch_GAN'` only)

Required whenever `method` is `'second_order'` or `'torch_GAN'`; leaving it out raises `KeyError`. There is no default. It
controls how the data is turned into the numeric array that the knockoff generator sees, and how the result is turned back
into a frame with `X`'s schema. The six values fall into two families, and only the second one uses
`conditional_expectations`:

**Encoding family:** `conditional_expectations` is ignored. The generator sees the raw numeric columns plus an encoding of each
categorical column, and its output is decoded back into categories. For all-numeric `X` there is nothing to encode, so the choice
among these four doesn't matter.

| value | behavior |
|---|---|
| `'ranger'` | Fits an R ranger probability forest per categorical column and uses the predicted class-probability logits as a soft numeric encoding. Decoded by softmax sampling. Requires R |
| `'xgb'` | Same as `'ranger'`, but the per-column probability model is an [`xgboost.XGBClassifier`](https://xgboost.readthedocs.io/en/latest/python/python_api.html). No R/`rpy2` dependency. Recognized xgboost hyperparameters (`max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`, `n_estimators`) are picked out of the extra kwargs passed to `get_knockoffs`/`get_second_order`/`get_torchGAN`; everything else goes to the knockoff generator |
| `'linear'` | Same as `'ranger'`, but with logistic regression. Lighter and faster |
| `'ohe'` | Hard one-hot encoding as floats, with no probability smoothing. Decoded by taking the largest dummy |

**Conditional-residual family:** uses `conditional_expectations` (see below). The generator sees only the numeric residuals
`X_j − E[X_j | X_{-j}]`, not the raw columns. Categorical columns never enter the generator; their knockoffs are drawn by
SCIP afterwards. These two values change the result even for all-numeric `X`.

| value | behavior |
|---|---|
| `'ranger_scip'` | Conditional expectations (when not supplied) and the categorical SCIP step use R `ranger` forests. Requires R |
| `'xgb_scip'` | Same as `'ranger_scip'`, but both steps use sequential [`xgboost`](https://xgboost.readthedocs.io/en/latest/python/python_api.html) models. No R/`rpy2` dependency |

`'ranger_scip'`/`'xgb_scip'` are the most statistically principled choices for mixed data. `'ranger'`, `'xgb'`, or `'linear'`
are convenient when a quick approximation is acceptable.

### `conditional_expectations` (`'second_order'`/`'torch_GAN'` with `'ranger_scip'`/`'xgb_scip'` only)

A `pl.DataFrame` of shape `(n, p_numeric)` giving `E[X_j | X_{-j}]` for each **numeric** column of `X`, with rows in the same
order as `X`. A frame you provide must have **exactly the same column names as `X`'s numeric columns**: each column is matched
to the `X` column with the same name, so column order doesn't matter. A missing column, an extra column (including any
categorical column), or a renamed column raises `ValueError`.

It only has an effect when `method` is `'second_order'` or `'torch_GAN'` **and** `categorical_method` is `'ranger_scip'` or
`'xgb_scip'`. In every other combination it is silently ignored: the encoding-family `categorical_method`s discard it, and
the `*_SCIP` methods never receive it. When it applies, the knockoffs are built from the residuals:

1. Residuals: `R = X_numeric − conditional_expectations`.
2. Knockoff residuals: `R̃ =` the `method`'s knockoffs of `R`, so `'second_order'` matches the residuals' mean and covariance, and `'torch_GAN'` learns their joint distribution.
3. Numeric knockoffs: `X̃_numeric = conditional_expectations + R̃`.
4. Categorical knockoffs: drawn one column at a time by SCIP (ranger or xgboost, per `categorical_method`), conditioning on `X` and `X̃_numeric`.

If `conditional_expectations=None` (the default), the package computes it internally on every call: with R `ranger` for
`'ranger_scip'` (`rbridge.get_forest_conditional_expectations`), or with sequential `xgboost.XGBRegressor` models for
`'xgb_scip'` (`xgbScip.get_forest_conditional_expectations`). To avoid refitting on every draw, compute it once and pass it in.
Either function's output works with either `categorical_method`.

```python
from heteroknockoffpy import xgbScip

# compute once, reuse across multiple knockoff draws
ce = xgbScip.get_forest_conditional_expectations(X)   # or rbridge.get_forest_conditional_expectations(X)

Xk1 = knockoff.get_knockoffs(X, method="torch_GAN", rng=rng,
                               categorical_method="xgb_scip",
                               conditional_expectations=ce)
Xk2 = knockoff.get_knockoffs(X, method="second_order", rng=rng,
                               categorical_method="xgb_scip",
                               conditional_expectations=ce)
```

### Combinations that don't work

| combination | result |
|---|---|
| `method='second_order'`/`'torch_GAN'` without `categorical_method` | `KeyError` |
| `method='glm_SCIP'` with any categorical column in `X` | `ValueError` |
| `method='glm_SCIP'` with `categorical_method` (or any unrecognized kwarg) | `TypeError` |
| `method='ranger_SCIP'` with `residuals_method='zero_inflated'` | `AssertionError` (`'xgb_SCIP'` only) |
| `method='second_order'` with `residuals_method` | R error: `unused argument` |
| `conditional_expectations` whose column names aren't exactly `X`'s numeric column names | `ValueError` (column order doesn't matter) |
| `conditional_expectations` with any `method`/`categorical_method` other than `'second_order'`/`'torch_GAN'` + `'ranger_scip'`/`'xgb_scip'` | no error; silently ignored |
| `categorical_method` with `method='ranger_SCIP'`/`'xgb_SCIP'` | no error; ignored (ranger/xgboost print an "unused argument/parameter" warning) |
| `weight` with `method='second_order'` and `shrink=True` | no error; shrinkage not applied (see `weight` above) |

---

## Importances

Every importance function shares this common core of inputs:

- **`X`** / **`Xk`** — Original data and its knockoffs (same schema, numeric + `pl.Categorical` columns).
- **`y`** — Outcome; scalar (continuous/count) or categorical Series/DataFrame.
- **`outcome_type`** — `'continuous'`/`'count'`/`'categorical'`; inferred from `y` if omitted.
- **`verbose`** — Verbosity level (`0` = silent).
- **`rng`** — Optional `np.random.Generator` for reproducibility (seeds torch / xgboost); unseeded if omitted. Not accepted by the ranger or linear-model functions.
- **`weight`** — Optional length-`n` sample weight. `None` (default) fits unweighted. For the PRISM-torch family (`grip2Importances`/`torchPrismImportances`/`prismGrip2Importances`/`grip2ImportancesPerOHE`/`torchPrismLocalGradients`), it's a per-sample multiplier on the training loss. For `rangerGiniImportances`/`rangerPrismImportances`, it's forwarded to `ranger::ranger`'s `case.weights` (resampling-probability weighting, not a loss multiplier). For `lassoImportances`/`ridgeImportances`/`elasticImportances`, it's forwarded as each underlying sklearn/statsmodels model's `sample_weight`. For the xgboost-based functions (`xgbImportances`/`xgbPrismImportances`/`xgbShapImportances`), it's xgboost's own native `sample_weight`.

All importance functions return a `np.ndarray` of length `2p` — scores for `[x_1, …, x_p, x̃_1, …, x̃_p]`. Use `wFromImportances` to convert these to knockoff W-statistics for variable selection.

### GRIP2 — `grip2Importances`

Trains a single MLP on `[X, Xk]` while sweeping a lambda/a regularization path. Records first-layer column norms `‖W[:,j]‖₂` at the end of each stage; the returned importances are the mean across all snapshots. Fast — no extra forward passes per snapshot.

```python
from heteroknockoffpy.importance import grip2Importances

imp = grip2Importances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",   # "continuous" | "count" | "categorical" | None (inferred)
    model_type="mlp",            # "mlp" | "pairwise" | "additive"
    calibrate=True,              # derive lambda_path from GRIP2 Eq. 5's gradient-ratio calibration
    n_blocks=30,
    epochs=500,
)
```

### Torch PRISM — `torchPrismImportances`

Same training procedure as GRIP2. Records per-feature output sensitivity `φⱼ = mean|ŷ(x+σeⱼ) − ŷ(x−σeⱼ)| / 2σ` at each stage. More directly tied to the model's predictions than GRIP2, but requires extra forward passes per snapshot.

```python
from heteroknockoffpy.importance import torchPrismImportances

imp = torchPrismImportances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",
    local_grad_method="bandwidth",  # "auto_diff" | "bandwidth" (default "bandwidth")
    bandwidth=1.0,                  # default 1.0; only used when local_grad_method="bandwidth"
    model_type="mlp",
    calibrate=True,                 # derive lambda_path from GRIP2 Eq. 5's gradient-ratio calibration
    n_blocks=30,
    epochs=500,
)
```

`calibrate` defaults to `False`. `calibrate=True` runs GRIP2 Eq. 5's gradient-ratio calibration to pick `lambda_path`'s range instead of requiring `lambda_min`/`lambda_max` up front. After warmup finishes, it measures `‖∇_W R(θ; λ=1, a)‖ / ‖∇_W L_pred(θ)‖` — the ratio between the group-regularization penalty's gradient norm and the (unregularized) prediction loss's gradient norm, both taken w.r.t. the first-layer group weights, on the held-out warmup validation split (or the full data if none was held out) — averaged over a freshly drawn `a_path` (`n_blocks` values from `Uniform(a_min, a_max)`, default `0.3`–`1.0`). Because the penalty scales linearly in `lambda`, that one ratio (measured at `lambda=1`) is enough to solve directly for the `(lambda_min, lambda_max)` bounds that make the ratio span `[calibrate_rmin, calibrate_rmax]` (default `0.01`–`1.0`) — no candidate lambda values are actually trained during calibration itself. `lambda_path` is then drawn as `n_blocks` values from `LogUniform` over that calibrated range.

With `calibrate=False`, the same two paths are drawn the same way but without the calibration step: `lambda_path` from `LogUniform(lambda_min, lambda_max)` (default `1e-3`–`1e-1`) and `a_path` from `Uniform(a_min, a_max)` (default `0.3`–`1.0`), each of length `n_blocks` (default `30`), both seeded by `rng`.

Either way, training then proceeds in `n_blocks` sequential BSS stages, each stage `b` regularized with that stage's own `(lambda_path[b], a_path[b])` pair. `epochs` is converted to a raw gradient-step budget (`epochs * ceil(n / batch_size)`) and distributed as evenly as possible in raw-step units (not whole epochs) across those stages, so a stage can end mid-epoch. Changing `n_blocks`/`lambda_path` length redistributes this fixed total budget, it never changes it. Pass `total_steps` instead of `epochs` to pin the exact step count directly, independent of `n`/`batch_size`.

### PRISM-GRIP2 — `prismGrip2Importances`

Trains the same model as Torch PRISM and GRIP2 but in a single pass, producing both sets of importances simultaneously. At each stage the snapshot closure captures GRIP2 group norms as a side effect while returning Torch PRISM local-gradient importances as the primary snapshot. Roughly halves the compute cost of running both methods separately.

Returns a 2-tuple `(g_importances, w_importances)`, each of shape `(2p,)`.

```python
from heteroknockoffpy.importance import prismGrip2Importances

g_imp, w_imp = prismGrip2Importances(
    X=X, Xk=Xk, y=y,
    layers=[64, 32],
    outcome_type="continuous",
    local_grad_method="bandwidth",  # "auto_diff" | "bandwidth" (default "bandwidth")
    calibrate=True,                 # derive lambda_path from GRIP2 Eq. 5's gradient-ratio calibration
    n_blocks=30,
    epochs=500,
)
```

All parameters are identical to `torchPrismImportances`, including the `calibrate`/`n_blocks` regularization-path defaults described above. `local_grad_method` defaults to `'bandwidth'` and governs only the G (Torch PRISM) snapshot; the W (GRIP2) snapshot uses group norms and needs no gradient method.

### GRIP2 per dummy — `grip2ImportancesPerOHE`

Same training as `grip2Importances`, but every one-hot dummy of a categorical column is its own regularization group and gets its own score, instead of being grouped back into one score per original variable. Only `model_type='mlp'`/`'pairwise'` are supported. Returns shape `(2·p_ohe,)` (first half for `X`'s one-hot-expanded columns, second half for `Xk`'s), so `wFromImportances` gives one W-statistic per dummy, not per variable.

### Torch PRISM local gradients — `torchPrismLocalGradients`

Trains the same network as `torchPrismImportances` but returns the raw per-sample local-gradient matrix for `X` only, shape `(n, p_numeric + Σ(c_k − 1))`, instead of averaged importances. Continuous/count outcomes only (`'categorical'` raises `NotImplementedError`).

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

Accepts extra keyword arguments: ranger's own (`num_trees`, `mtry`, `min_node_size`, `max_depth`, `sample_fraction`, `num_threads`, `respect_unordered_factors`) go to `ranger::ranger`, and `bandwidth`/`bandwidth_exponent`/`exponent` go to the PRISM script. Requires R and `rpy2` with the `ranger` package installed.

### XGBoost — `xgbImportances`, `xgbPrismImportances`, `xgbShapImportances`

Each fits a single xgboost model on `[X, Xk]` with native categorical handling (no one-hot encoding), so no R is needed. Pass xgboost constructor hyperparameters as `model_kwargs={...}` (`max_depth`, `learning_rate`, `n_estimators`, …).

- `xgbImportances` — split-based importances; `importance_type` is `'gain'` (default), `'weight'`, `'cover'`, `'total_gain'`, or `'total_cover'`.
- `xgbPrismImportances` — the xgboost analogue of `rangerPrismImportances` (same sensitivity measures as the table above). Extra kwargs: `bandwidth` (default `1.0`), `bandwidth_exponent` (default `0.0`), `exponent` (default `1.0`).
- `xgbShapImportances` — mean absolute SHAP contributions via `shap.TreeExplainer`; requires `pip install 'heteroknockoffpy[shap]'`.

```python
from heteroknockoffpy.importance import xgbPrismImportances

imp = xgbPrismImportances(X=X, Xk=Xk, y=y, outcome_type="continuous", rng=rng,
                          model_kwargs={"max_depth": 4, "n_estimators": 200})
```

### `model_type` (torch GRIP2/PRISM family)

| value | behavior |
|---|---|
| `'mlp'` | Standard MLP on `[X, Xk]` with group regularization on first-layer columns |
| `'pairwise'` | Adds a learnable filter that creates convex combinations of `xⱼ` and `x̃ⱼ`, forcing explicit per-feature competition |
| `'additive'` | Feature-wise sub-networks with separate group regularization for `X` and `Xk` channels |

---

## Variable selection

```python
from heteroknockoffpy.importance import wFromImportances, selection_threshold

W = wFromImportances(imp)                    # W_method="difference" (default) | "signed_max"
threshold = selection_threshold(W, fdr=0.1)  # offset=1 (default) controls FDR exactly; offset=0 controls modified FDR
selected = [j for j, w in enumerate(W) if w >= threshold]
```

---

## Example data

The PRISM simulation results (Zenodo DOI [10.5281/zenodo.22085306](https://doi.org/10.5281/zenodo.22085306), ~580 MB) are
available as optional example data. Nothing is downloaded at install time. Install the downloader extra, then fetch it
explicitly or let the first load do it:

```
pip install 'heteroknockoffpy[example-data]'
heteroknockoffpy-fetch-data          # or: python -m heteroknockoffpy.datasets  (--path, --force, --keep-zip, --data-home)
```

The download is SHA-256 verified and cached in the user cache directory (`~/Library/Caches/heteroknockoffpy` on macOS,
`~/.cache/heteroknockoffpy` on Linux). Override it with `data_home=` or `$HETEROKNOCKOFFPY_DATA`, or point `root=` at
an already-extracted `data/` directory (e.g. a PRISMdata checkout).

```python
import numpy as np
import polars as pl
from heteroknockoffpy import datasets

count = datasets.load_prism("count")     # "linear" | "continuous" | "count" | "categorical"; downloads if needed
count.catalog                            # one row per trial: scenario, n, p, covariance, magnitude, importance_methods,
                                         # relevant_features, fdr, ... -- ready for polars group_by/plots
view = count.filter(importance_method="torch_prism", scenario="n_sweep", n=[1024, 4096])
view = view.filter(pl.col("covariance_rho") > 0.5)

trial = view[0]                          # by position, or view["<bundle_id>"]; iterate with `for trial in view:`
trial.X, trial.Xk, trial.y               # design matrix, stored knockoffs, response
trial.importances, trial.selections      # stored results (long polars frames)
trial.relevant_features                  # ground truth from the generating function
trial.selected_features("torch_prism")   # stored selections
trial.selected_features("torch_prism", fdr=0.1)    # re-threshold stored importances, no refit
trial.fdp("torch_prism"), trial.power("torch_prism")

# Recompute with the stored settings (fresh randomness)
rng = np.random.default_rng(0)
Xk = trial.compute_knockoffs(rng)
imp = trial.compute_importances("torch_prism", rng=rng, Xk=Xk)
```

A trial can store more than one importance method (`torch_grip2` and `torch_prism` are read off one shared fit), so
methods that take `method` require it whenever `trial.methods` has more than one entry. Lower-level access is in
`heteroknockoffpy.datasets.prism.raw` (tables in their on-disk shape) and `heteroknockoffpy.datasets.prism.transforms`
(`heteroknockoffpy_kwargs`, `relevant_features`, importance-array helpers).
