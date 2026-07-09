#
#//  scripts/tuning/xgb_measure_separation_search.py
#//  heteroknockoffpy
#//
#//  One-at-a-time data-parameter screen looking for settings where the two
#//  xgb-derived importance measures (score_<importance_type> vs prism, from a
#//  single shared fit) diverge from each other. IMPORTANCE_TYPE selects which
#//  of xgboost's split-based importance types (weight/gain/cover/...) is used
#//  for the "score" side. Reuses the fit/metric machinery from
#//  xgb_vs_ranger_gini_benchmark.py; ranger_gini is skipped here for speed
#//  since only the xgb-vs-xgb separation is of interest.
#//

import json
import sys
import time

import numpy as np
import xgboost

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/tuning")

from heteroknockoffpy import importance
from heteroknockoffpy.utilities import get_ar1_simple_case
from heteroknockoffpy.xgbImportances import _concat_X_Xk

from xgb_vs_ranger_gini_benchmark import (
    XGB_PARAMS,
    compute_metrics,
    prism_importance_continuous,
    score_weight_importance,
)

IMPORTANCE_TYPE = "gain"
SCORE_METHOD_NAME = "score_" + IMPORTANCE_TYPE


BASELINE = dict(
    n=2000,
    p_numeric=10,
    p_categorical=5,
    categories=4,
    rho=0.5,
    p_relevant=0.4,
    noise_sd=0.5,
    oracle_knockoffs=False,
)

N_TRIALS = 5

# One-at-a-time sweeps: (param_name, list_of_values)
SWEEPS = [
    ("categories", [2, 3, 4, 8]),
    ("rho", [0.0, 0.3, 0.7, 0.9]),
    ("p_numeric", [10, 20, 40]),
    ("p_categorical", [2, 5, 10]),
    ("n", [300, 1000, 2000]),
    ("noise_sd", [1.0, 0.5, 0.25, 0.1]),
]


def run_trial(params: dict, trial_idx: int) -> dict:
    rng = np.random.default_rng(trial_idx)
    case = get_ar1_simple_case(rng=rng, **params)

    cols = case.X.columns
    X_all = _concat_X_Xk(case.X, case.Xk)
    X_all_pd = X_all.to_pandas()

    model = xgboost.XGBRegressor(**XGB_PARAMS)
    model.fit(X_all_pd, case.y)

    score_imp = score_weight_importance(model, X_all.columns, importance_type=IMPORTANCE_TYPE)
    prism_imp = prism_importance_continuous(model, X_all_pd)

    W_score = importance.wFromImportances(score_imp)
    W_prism = importance.wFromImportances(prism_imp)

    return {
        SCORE_METHOD_NAME: compute_metrics(W_score, cols, case.relevant_vars),
        "prism": compute_metrics(W_prism, cols, case.relevant_vars),
    }


def summarize(trials: list) -> dict:
    metric_names = ["power", "fdr", "power_categorical", "fdr_categorical", "f1"]
    out = {}
    for method in [SCORE_METHOD_NAME, "prism"]:
        out[method] = {}
        for metric in metric_names:
            vals = np.array([t[method][metric] for t in trials], dtype=float)
            out[method][metric] = dict(mean=float(np.nanmean(vals)), std=float(np.nanstd(vals)))
        #
    #
    return out


def separation(summary: dict) -> dict:
    return {
        metric: summary[SCORE_METHOD_NAME][metric]["mean"] - summary["prism"][metric]["mean"]
        for metric in ["power", "fdr", "power_categorical", "fdr_categorical", "f1"]
    }


def main():
    results = []  # list of dicts: {param, value, params, summary, separation, max_abs_sep}

    t_start = time.time()

    # -- baseline itself, once
    settings_to_run = [("baseline", "baseline", dict(BASELINE))]
    for param, values in SWEEPS:
        for v in values:
            params = dict(BASELINE)
            params[param] = v
            if param == "categories" and BASELINE["categories"] == v:
                continue  # already covered by baseline
            if param == "rho" and BASELINE["rho"] == v:
                continue
            if param == "p_numeric" and BASELINE["p_numeric"] == v:
                continue
            if param == "p_categorical" and BASELINE["p_categorical"] == v:
                continue
            if param == "n" and BASELINE["n"] == v:
                continue
            if param == "noise_sd" and BASELINE["noise_sd"] == v:
                continue
            settings_to_run.append((param, v, params))
        #
    #

    for param, value, params in settings_to_run:
        t0 = time.time()
        trials = [run_trial(params, trial_idx) for trial_idx in range(N_TRIALS)]
        dt = time.time() - t0
        summ = summarize(trials)
        sep = separation(summ)
        max_abs_sep = max(abs(v) for v in sep.values())
        results.append(dict(
            param=param, value=value, params=params,
            summary=summ, separation=sep, max_abs_sep=max_abs_sep,
        ))
        print(
            "{:<15s}={:<8s} ({:.1f}s, {} trials): "
            "sep power={:+.2f} fdr={:+.2f} power_cat={:+.2f} fdr_cat={:+.2f} f1={:+.2f}".format(
                str(param), str(value), dt, N_TRIALS,
                sep["power"], sep["fdr"], sep["power_categorical"], sep["fdr_categorical"], sep["f1"],
            )
        )
    #

    total_wall = time.time() - t_start
    print("\nTotal wall time: {:.1f}s".format(total_wall))

    out_path = "scripts/tuning/xgb_measure_separation_search_results_{}.json".format(IMPORTANCE_TYPE)
    with open(out_path, "w") as f:
        json.dump(dict(baseline=BASELINE, n_trials=N_TRIALS, results=results, total_wall_seconds=total_wall), f, indent=2, default=float)
    #
    print("Wrote results to {}".format(out_path))

    print("\n=== Ranked by max |separation| (any metric) ===")
    for r in sorted(results, key=lambda r: -r["max_abs_sep"]):
        print("{:<15s}={:<8s} max_abs_sep={:.3f}  sep={}".format(
            str(r["param"]), str(r["value"]), r["max_abs_sep"],
            {k: round(v, 3) for k, v in r["separation"].items()},
        ))
    #


if __name__ == "__main__":
    main()
