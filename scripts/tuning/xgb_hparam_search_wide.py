#
#//  scripts/tuning/xgb_hparam_search_wide.py
#//  heteroknockoffpy
#//
#//  Redo of xgb_hparam_search.py at a larger, sparser scale: p_numeric=60,
#//  p_categorical=30 (6x the original), p_relevant=0.15 (vs. 0.4 originally).
#//  Same search objective/space/procedure as the original -- only DATA_PARAMS
#//  differs. See docs/heteroknockoffpy/xgbImportance.html for the full design
#//  rationale (both the original scale and this redo).
#//

import json
import sys
import time

import numpy as np
import optuna
import xgboost
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, "src")

from heteroknockoffpy.utilities import get_ar1_simple_case
from heteroknockoffpy.xgbImportances import _concat_X_Xk


DATA_PARAMS = dict(
    n=2000,
    p_numeric=60,
    p_categorical=30,
    categories=4,
    rho=0.5,
    p_relevant=0.15,
    noise_sd=1.0,
    oracle_knockoffs=True,
)

N_TRIALS = 200
PRIMARY_SEED = 0
REVALIDATION_SEEDS = [1, 2, 3]
TOP_K_REVALIDATE = 5


def build_case(seed: int):
    rng = np.random.default_rng(seed)
    case = get_ar1_simple_case(rng=rng, **DATA_PARAMS)
    X_all = _concat_X_Xk(case.X, case.Xk)
    X_all_pd = X_all.to_pandas()
    return case, X_all_pd


def score_trial_params(params: dict, X_all_pd, y: np.ndarray, oracle: np.ndarray) -> float:
    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        **params,
    )
    model.fit(X_all_pd, y)
    pred = model.predict(X_all_pd)
    return float(mean_squared_error(oracle, pred))


def suggest_params(trial: optuna.Trial) -> dict:
    return dict(
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1e-2, 20, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 5, log=True),
        n_estimators=trial.suggest_int("n_estimators", 50, 2000, log=True),
    )


def main():
    print("Building primary dataset (seed={})...".format(PRIMARY_SEED))
    case, X_all_pd = build_case(PRIMARY_SEED)
    y = case.y
    oracle = case.oracle
    print("  X_all shape: {}".format(X_all_pd.shape))
    print("  n_relevant_vars: {}".format(len(case.relevant_vars)))
    print("  relevant_vars: {}".format(case.relevant_vars))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        return score_trial_params(params, X_all_pd, y, oracle)

    sampler = optuna.samplers.TPESampler(seed=PRIMARY_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    search_wall_seconds = time.time() - t0
    print("Search done in {:.1f}s. Best primary MSE: {:.5f}".format(search_wall_seconds, study.best_value))

    # -- Revalidate top-K trials on fresh seeds
    top_trials = sorted(study.trials, key=lambda t: t.value)[:TOP_K_REVALIDATE]

    revalidation_cases = {seed: build_case(seed) for seed in REVALIDATION_SEEDS}

    revalidation_results = []
    for rank, trial in enumerate(top_trials, start=1):
        params = trial.params
        fresh_scores = []
        for seed, (rcase, rX_all_pd) in revalidation_cases.items():
            mse = score_trial_params(params, rX_all_pd, rcase.y, rcase.oracle)
            fresh_scores.append(mse)
        #
        mean_fresh = float(np.mean(fresh_scores))
        revalidation_results.append(dict(
            rank=rank,
            primary_mse=trial.value,
            params=params,
            fresh_seed_mses=dict(zip(REVALIDATION_SEEDS, fresh_scores)),
            mean_fresh_mse=mean_fresh,
        ))
        print("Trial rank {}: primary_mse={:.5f} mean_fresh_mse={:.5f} params={}".format(
            rank, trial.value, mean_fresh, params
        ))
    #

    best_by_fresh = min(revalidation_results, key=lambda r: r["mean_fresh_mse"])

    # -- Baseline: xgboost defaults, for comparison
    default_mse_primary = score_trial_params({}, X_all_pd, y, oracle)
    default_fresh = [
        score_trial_params({}, rX_all_pd, rcase.y, rcase.oracle)
        for seed, (rcase, rX_all_pd) in revalidation_cases.items()
    ]
    default_mean_fresh = float(np.mean(default_fresh))

    # -- Oracle variance, for R^2-style context
    oracle_var_primary = float(np.var(oracle))

    results = dict(
        data_params=DATA_PARAMS,
        n_trials=N_TRIALS,
        primary_seed=PRIMARY_SEED,
        revalidation_seeds=REVALIDATION_SEEDS,
        search_wall_seconds=search_wall_seconds,
        n_relevant_vars=len(case.relevant_vars),
        best_primary_mse=study.best_value,
        best_primary_params=study.best_params,
        oracle_var_primary=oracle_var_primary,
        default_mse_primary=default_mse_primary,
        default_mean_fresh_mse=default_mean_fresh,
        revalidation_results=revalidation_results,
        chosen=best_by_fresh,
    )

    out_path = "scripts/tuning/xgb_hparam_search_wide_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    #
    print("Wrote results to {}".format(out_path))
    print("Chosen params (best mean_fresh_mse={:.5f}):".format(best_by_fresh["mean_fresh_mse"]))
    print(json.dumps(best_by_fresh["params"], indent=2))


if __name__ == "__main__":
    main()
