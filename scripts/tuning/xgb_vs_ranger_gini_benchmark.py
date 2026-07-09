#
#//  scripts/tuning/xgb_vs_ranger_gini_benchmark.py
#//  heteroknockoffpy
#//
#//  Power/FDR/F1 benchmark: xgb score(weight) importance + xgb prism importance
#//  (from a single shared XGBRegressor fit, using the tuned hyperparameters from
#//  xgb_hparam_search.py) vs. rangerGiniImportances, swept over noise_sd, with
#//  realistic (oracle_knockoffs=False) knockoffs. See
#//  docs/heteroknockoffpy/xgbImportance.html for the full design rationale.
#//

import json
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
import xgboost

sys.path.insert(0, "src")

from heteroknockoffpy import importance
from heteroknockoffpy.utilities import get_ar1_simple_case
from heteroknockoffpy.xgbImportances import _concat_X_Xk


DATA_PARAMS = dict(
    n=2000,
    p_numeric=10,
    p_categorical=5,
    categories=4,
    rho=0.5,
    p_relevant=0.4,
    oracle_knockoffs=False,
)

NOISE_SDS = [1.0, 0.5, 0.25]
N_TRIALS = 10
TARGET_FDR = 0.2

XGB_PARAMS = dict(
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=True,
    max_depth=5,
    learning_rate=0.020709001834846544,
    min_child_weight=14.145217915977055,
    subsample=0.6277268835909049,
    colsample_bytree=0.9738467518239733,
    reg_alpha=0.23524091767522132,
    reg_lambda=1.292204336359651,
    gamma=0.0038791239397924842,
    n_estimators=700,
)


def score_weight_importance(model: xgboost.XGBRegressor, X_all_columns, importance_type: str = "weight") -> np.ndarray:
    score_dict = model.get_booster().get_score(importance_type=importance_type)
    imp = np.zeros(len(X_all_columns))
    for i, col in enumerate(X_all_columns):
        imp[i] = score_dict.get(col, 0.0)
    #
    return imp


def prism_importance_continuous(
    model: xgboost.XGBRegressor,
    X_all_pd: pd.DataFrame,
    bandwidth: float = 1.0,
    bandwidth_exponent: float = 0.2,
    exponent: float = 1.0,
) -> np.ndarray:
    # Mirrors xgbImportances.prism_importances' continuous-outcome branch, applied
    # to an already-fitted model instead of fitting internally.
    n = X_all_pd.shape[0]
    p_all = X_all_pd.shape[1]
    n_factor = n ** bandwidth_exponent

    base_preds = model.predict(X_all_pd)
    importances_pointwise = np.zeros((n, p_all))

    for j, col in enumerate(X_all_pd.columns):
        is_categorical = isinstance(X_all_pd[col].dtype, pd.CategoricalDtype)

        if is_categorical:
            levels = X_all_pd[col].cat.categories
            if len(levels) == 0:
                continue
            #
            preds_mat = np.zeros((n, len(levels)))
            for ki, lev in enumerate(levels):
                X_test = X_all_pd.copy()
                X_test[col] = pd.Categorical([lev] * n, dtype=X_all_pd[col].dtype)
                preds_mat[:, ki] = model.predict(X_test)
            #
            importances_pointwise[:, j] = preds_mat.max(axis=1) - preds_mat.min(axis=1)
        #
        else:
            col_std = X_all_pd[col].std()
            bw = float(col_std) * bandwidth / n_factor
            if bw == 0:
                continue
            #
            X_test = X_all_pd.copy()
            X_test[col] = X_all_pd[col] + bw
            mod_preds = model.predict(X_test)
            importances_pointwise[:, j] = (mod_preds - base_preds) / bw
        #
    #
    return np.mean(np.abs(importances_pointwise) ** exponent, axis=0)


def compute_metrics(W: np.ndarray, cols: list, relevant_vars: list) -> dict:
    relevant_mask = np.array([c in relevant_vars for c in cols])
    categorical_mask = np.array([c.startswith("cat_") for c in cols])

    T = importance.selection_threshold(W, fdr=TARGET_FDR)
    selected_mask = (W >= T) if np.isfinite(T) else np.zeros_like(W, dtype=bool)

    n_relevant = int(relevant_mask.sum())
    n_selected = int(selected_mask.sum())
    tp = int((selected_mask & relevant_mask).sum())
    fp = int((selected_mask & ~relevant_mask).sum())

    power = tp / n_relevant if n_relevant > 0 else float("nan")
    fdr = fp / n_selected if n_selected > 0 else 0.0

    cat_relevant_mask = relevant_mask & categorical_mask
    cat_selected_mask = selected_mask & categorical_mask
    n_cat_relevant = int(cat_relevant_mask.sum())
    n_cat_selected = int(cat_selected_mask.sum())
    cat_tp = int((selected_mask & relevant_mask & categorical_mask).sum())
    cat_fp = int((selected_mask & ~relevant_mask & categorical_mask).sum())

    power_categorical = cat_tp / n_cat_relevant if n_cat_relevant > 0 else float("nan")
    fdr_categorical = cat_fp / n_cat_selected if n_cat_selected > 0 else 0.0

    precision = 1 - fdr
    recall = power
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return dict(
        power=power,
        fdr=fdr,
        power_categorical=power_categorical,
        fdr_categorical=fdr_categorical,
        f1=f1,
        n_selected=n_selected,
        n_cat_selected=n_cat_selected,
    )


def run_trial(noise_sd: float, trial_idx: int) -> dict:
    rng = np.random.default_rng(trial_idx)
    case = get_ar1_simple_case(rng=rng, noise_sd=noise_sd, **DATA_PARAMS)

    cols = case.X.columns
    X_all = _concat_X_Xk(case.X, case.Xk)
    X_all_pd = X_all.to_pandas()

    # -- Single shared XGB fit -> score(weight) + prism importances
    model = xgboost.XGBRegressor(**XGB_PARAMS)
    model.fit(X_all_pd, case.y)

    score_imp = score_weight_importance(model, X_all.columns)
    prism_imp = prism_importance_continuous(model, X_all_pd)

    W_score = importance.wFromImportances(score_imp)
    W_prism = importance.wFromImportances(prism_imp)

    # -- ranger gini (independent fit)
    y_series = pl.Series("y", case.y)
    ranger_imp = importance.rangerGiniImportances(X=case.X, Xk=case.Xk, y=y_series, outcome_type="continuous")
    W_ranger = importance.wFromImportances(ranger_imp)

    return dict(
        score_weight=compute_metrics(W_score, cols, case.relevant_vars),
        prism=compute_metrics(W_prism, cols, case.relevant_vars),
        ranger_gini=compute_metrics(W_ranger, cols, case.relevant_vars),
    )


def main():
    all_results = {}  # noise_sd -> method -> list of per-trial metric dicts

    t_start = time.time()
    for noise_sd in NOISE_SDS:
        all_results[noise_sd] = dict(score_weight=[], prism=[], ranger_gini=[])
        for trial_idx in range(N_TRIALS):
            t0 = time.time()
            trial_result = run_trial(noise_sd, trial_idx)
            dt = time.time() - t0
            for method, metrics in trial_result.items():
                all_results[noise_sd][method].append(metrics)
            #
            print(
                "noise_sd={} trial={} ({:.1f}s): score power={:.2f} fdr={:.2f} | "
                "prism power={:.2f} fdr={:.2f} | ranger power={:.2f} fdr={:.2f}".format(
                    noise_sd, trial_idx, dt,
                    trial_result["score_weight"]["power"], trial_result["score_weight"]["fdr"],
                    trial_result["prism"]["power"], trial_result["prism"]["fdr"],
                    trial_result["ranger_gini"]["power"], trial_result["ranger_gini"]["fdr"],
                )
            )
        #
    #
    total_wall = time.time() - t_start
    print("Total wall time: {:.1f}s".format(total_wall))

    # -- Summarize mean/std per (noise_sd, method, metric)
    summary = {}
    metric_names = ["power", "fdr", "power_categorical", "fdr_categorical", "f1"]
    for noise_sd in NOISE_SDS:
        summary[noise_sd] = {}
        for method in ["score_weight", "prism", "ranger_gini"]:
            trials = all_results[noise_sd][method]
            summary[noise_sd][method] = {}
            for metric in metric_names:
                vals = np.array([t[metric] for t in trials], dtype=float)
                summary[noise_sd][method][metric] = dict(
                    mean=float(np.nanmean(vals)),
                    std=float(np.nanstd(vals)),
                )
            #
        #
    #

    out = dict(
        data_params=DATA_PARAMS,
        noise_sds=NOISE_SDS,
        n_trials=N_TRIALS,
        target_fdr=TARGET_FDR,
        xgb_params=XGB_PARAMS,
        total_wall_seconds=total_wall,
        raw=all_results,
        summary=summary,
    )

    out_path = "scripts/tuning/xgb_vs_ranger_gini_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    #
    print("Wrote results to {}".format(out_path))

    print("\n=== Summary (mean +/- std over {} trials) ===".format(N_TRIALS))
    for noise_sd in NOISE_SDS:
        print("\n-- noise_sd={} --".format(noise_sd))
        for method in ["score_weight", "prism", "ranger_gini"]:
            print("  {}:".format(method))
            for metric in metric_names:
                s = summary[noise_sd][method][metric]
                print("    {:<20s} {:.3f} +/- {:.3f}".format(metric, s["mean"], s["std"]))
            #
        #
    #


if __name__ == "__main__":
    main()
