#
#//  scripts/tuning/synth_sweep_2_audit.py
#//  heteroknockoffpy
#//
#//  Audit + aggregate the synth_sweep_2 silverknockoff bundle: ranger_gini vs.
#//  xgb_prism vs. xgb_score(gain) at p=30 (20 numeric + 10 categorical, k=4),
#//  n up to 8000, sweeping magnitude/rho/n independently (5 points x 50 trials
#//  each). Follow-up to synth_sweep_1 with denser interaction-only signal (6
#//  relevant vars, all in pairwise multiplicative interactions, no standalone
#//  main effects) and freshly-tuned XGB hyperparameters for this data size.
#//  Real silverknockoff/silverbrain pipeline output (real second-order OHE
#//  knockoffs, real selection_threshold-based FDR selection), read directly
#//  from the bundle's parquet files -- no silverknockoff import needed. See
#//  docs/heteroknockoffpy/xgbImportance.html for narrative context.
#//

import json
from pathlib import Path

import polars as pl

BUNDLE = Path(
    "/Users/evanmason/Library/Application Support/silverknockoff/bundles/synth_sweep_2"
)
OUT_JSON = Path(__file__).parent / "synth_sweep_2_audit_results.json"

RELEVANT_NUMERIC = {2, 7, 13, 18}
RELEVANT_CATEGORICAL = {23, 27}
RELEVANT = RELEVANT_NUMERIC | RELEVANT_CATEGORICAL
CATEGORICAL_MIN_IDX = 20  # feature_idx 20-29 are the 10 categorical columns

TRIALS_PER_CELL = 50
N_SWEEPS = 3
N_POINTS = 5

# (sweep idn, point column, kept point values, point order)
SWEEPS = [
    ("magnitude_sweep", "magnitude", [8.0, 16.0, 32.0, 64.0, 128.0]),
    ("rho_sweep", "rho", [0.1, 0.3, 0.5, 0.7, 0.9]),
    ("n_sweep", "n", [256.0, 512.0, 1024.0, 2048.0, 4096.0]),
]


def _nu_val(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Pivot one par_name's scalar value (whichever par_val_* column is non-null) per bundle."""
    row = df.filter(pl.col("par_name") == name)
    val = (
        pl.coalesce(
            pl.col("par_val_float64").cast(pl.Float64),
            pl.col("par_val_int64").cast(pl.Float64),
            pl.col("par_val_string").cast(pl.Float64, strict=False),
        )
        if name not in ("idn", "method", "importance_type", "x_bundle_idn_hex")
        else pl.col("par_val_string")
    )
    return row.select("par_bundle_idn", val.alias(name))


def main() -> None:
    log: list[str] = []

    def p(msg: str) -> None:
        print(msg)
        log.append(msg)

    index = pl.read_parquet(BUNDLE / "data" / "index.parquet")
    params = pl.read_parquet(BUNDLE / "data" / "parameters_table.parquet")
    importances = pl.read_parquet(BUNDLE / "data" / "importances" / "*.parquet")
    selections = pl.read_parquet(BUNDLE / "data" / "selections" / "*.parquet")

    p(f"index.parquet: {index.shape[0]} rows, par_stage counts: "
      f"{index.group_by('par_stage').len().sort('par_stage').to_dicts()}")

    trial_idx = index.filter(pl.col("par_stage") == "trial")
    trial_idx_dedup = trial_idx.unique(subset=["par_bundle_idn"])
    p(f"trial index rows: {trial_idx.shape[0]} -> {trial_idx_dedup.shape[0]} after dedup "
      f"by par_bundle_idn ({trial_idx.shape[0] - trial_idx_dedup.shape[0]} duplicates dropped)")

    x_idx = index.filter(pl.col("par_stage") == "X").select("par_bundle_idn")

    # --- trial_covariates: sweep, magnitude, method live directly on the trial bundle ---
    trial_params = params.filter(pl.col("par_bundle_idn").is_in(trial_idx_dedup["par_bundle_idn"]))

    sweep_idn = _nu_val(trial_params.filter(pl.col("par_group") == "y"), "idn").rename({"idn": "sweep"})
    magnitude = _nu_val(trial_params.filter(pl.col("par_group") == "y"), "magnitude")
    method_raw = _nu_val(trial_params.filter(pl.col("par_group") == "importances"), "method")
    importance_type = _nu_val(trial_params.filter(pl.col("par_group") == "importances"), "importance_type")
    x_hex = _nu_val(trial_params.filter(pl.col("par_group") == "_meta"), "x_bundle_idn_hex")

    # --- rho, n live on the X-stage bundle, reached via the _meta hex hop ---
    x_params = params.filter(pl.col("par_bundle_idn").is_in(x_idx["par_bundle_idn"]))
    x_rho = _nu_val(x_params.filter(pl.col("par_group") == "X"), "covariance_rho").rename({"covariance_rho": "rho"})
    x_n = _nu_val(x_params.filter(pl.col("par_group") == "X"), "n")
    x_covariates = (
        x_idx.with_columns(pl.col("par_bundle_idn").bin.encode("hex").alias("x_bundle_idn_hex"))
        .join(x_rho.rename({"par_bundle_idn": "x_bidn"}), left_on="par_bundle_idn", right_on="x_bidn", how="left")
        .join(x_n.rename({"par_bundle_idn": "x_bidn"}), left_on="par_bundle_idn", right_on="x_bidn", how="left")
        .select("x_bundle_idn_hex", "rho", "n")
    )

    trial_covariates = (
        trial_idx_dedup.select("par_bundle_idn")
        .join(sweep_idn, on="par_bundle_idn", how="left")
        .join(magnitude, on="par_bundle_idn", how="left")
        .join(method_raw, on="par_bundle_idn", how="left")
        .join(importance_type, on="par_bundle_idn", how="left")
        .join(x_hex.rename({"x_bundle_idn_hex": "hex_join"}), on="par_bundle_idn", how="left")
        .join(x_covariates, left_on="hex_join", right_on="x_bundle_idn_hex", how="left")
        .drop("hex_join")
        .with_columns(
            pl.when(pl.col("method") == "xgb_score")
            .then(pl.col("method") + pl.lit("_") + pl.col("importance_type"))
            .otherwise(pl.col("method"))
            .alias("method")
        )
    )

    # importance_type is only set for xgb_score bundles, so it's excluded from
    # the null check (it's folded into `method` already and no longer needed).
    n_null = trial_covariates.drop("importance_type").null_count().to_dicts()[0]
    p(f"trial_covariates: {trial_covariates.shape[0]} rows, null counts per column: {n_null}")
    n_methods = trial_covariates["method"].n_unique()
    expected_total = N_SWEEPS * N_POINTS * TRIALS_PER_CELL * n_methods
    p(f"detected {n_methods} methods: {sorted(trial_covariates['method'].unique().to_list())}")
    p(f"expected total trial-bundles (no staleness): {expected_total}")
    assert all(v == 0 for v in n_null.values()), f"unexpected nulls in trial_covariates: {n_null}"

    # --- stale-trial detection: recalibration leftovers (if any) ---
    by_sweep_mag = trial_covariates.group_by(["sweep", "magnitude"]).len().sort(["sweep", "magnitude"])
    p(f"rows by (sweep, magnitude):\n{by_sweep_mag}")

    kept = trial_covariates.filter(
        ((pl.col("sweep") == "magnitude_sweep") & (pl.col("magnitude").is_in([8.0, 16.0, 32.0, 64.0, 128.0])))
        | ((pl.col("sweep") != "magnitude_sweep") & (pl.col("magnitude") == 64.0))
    )
    n_stale = trial_covariates.shape[0] - kept.shape[0]
    p(f"dropped {n_stale} stale trial-bundles (recalibration leftovers, if any); {kept.shape[0]} remain")

    point_col = {"magnitude_sweep": "magnitude", "rho_sweep": "rho", "n_sweep": "n"}
    kept = kept.with_columns(
        pl.when(pl.col("sweep") == "magnitude_sweep").then(pl.col("magnitude"))
        .when(pl.col("sweep") == "rho_sweep").then(pl.col("rho"))
        .otherwise(pl.col("n"))
        .alias("point_value")
    )
    cell_counts = kept.group_by(["sweep", "point_value", "method"]).len().sort(["sweep", "point_value", "method"])
    bad_cells = cell_counts.filter(pl.col("len") != TRIALS_PER_CELL)
    p(f"cells with != {TRIALS_PER_CELL} trials: {bad_cells.to_dicts() if bad_cells.shape[0] else 'none'}")
    p(f"cell_counts shape: {cell_counts.shape[0]} (expected {N_SWEEPS * N_POINTS * n_methods})")

    # --- importances/selections sanity ---
    for name, df in [("importances", importances), ("selections", selections)]:
        dupe = df.group_by(["par_bundle_idn", "importance_method", "feature_idx"]).len().filter(pl.col("len") > 1)
        idx_range = df.select(pl.col("feature_idx").min().alias("min"), pl.col("feature_idx").max().alias("max"))
        p(f"{name}: {df.shape[0]} rows, feature_idx range {idx_range.to_dicts()[0]}, "
          f"{dupe.shape[0]} duplicate (bundle,method,feature) groups")
        assert dupe.shape[0] == 0
        assert idx_range["min"][0] == 0 and idx_range["max"][0] == 29

    fdr_targets = selections.select(pl.col("fdr_target").unique())
    p(f"selections.fdr_target unique values: {fdr_targets['fdr_target'].to_list()}")
    assert fdr_targets.shape[0] == 1

    sel_bundles = set(selections["par_bundle_idn"].unique().to_list())
    kept_bundles = set(kept["par_bundle_idn"].unique().to_list())
    p(f"kept bundles missing from selections: {len(kept_bundles - sel_bundles)}; "
      f"selections bundles not in kept set (expected, includes stale/imports): {len(sel_bundles - kept_bundles)}")
    assert not (kept_bundles - sel_bundles), "some kept trial bundles have no selections file"

    # --- metrics ---
    sel = selections.filter(pl.col("par_bundle_idn").is_in(kept_bundles)).join(
        kept.select("par_bundle_idn", "sweep", "point_value", "method"), on="par_bundle_idn", how="inner"
    )

    per_trial = sel.group_by(["sweep", "point_value", "method", "par_bundle_idn"]).agg(
        n_selected=pl.col("selected").sum(),
        n_tp=(pl.col("selected") & pl.col("feature_idx").is_in(sorted(RELEVANT))).sum(),
        n_selected_cat=(pl.col("selected") & (pl.col("feature_idx") >= CATEGORICAL_MIN_IDX)).sum(),
        n_tp_cat=(pl.col("selected") & pl.col("feature_idx").is_in(sorted(RELEVANT_CATEGORICAL))).sum(),
    ).with_columns(
        (pl.col("n_tp") / len(RELEVANT)).alias("power"),
        pl.when(pl.col("n_selected") > 0).then(1 - pl.col("n_tp") / pl.col("n_selected")).otherwise(0.0).alias("fdr"),
        (pl.col("n_tp_cat") / len(RELEVANT_CATEGORICAL)).alias("power_categorical"),
        pl.when(pl.col("n_selected_cat") > 0)
        .then(1 - pl.col("n_tp_cat") / pl.col("n_selected_cat")).otherwise(0.0).alias("fdr_categorical"),
    ).with_columns(
        pl.when((1 - pl.col("fdr") + pl.col("power")) > 0)
        .then(2 * (1 - pl.col("fdr")) * pl.col("power") / (1 - pl.col("fdr") + pl.col("power")))
        .otherwise(0.0)
        .alias("f1")
    )

    p(f"per_trial shape: {per_trial.shape[0]} (expected {kept.shape[0]})")
    assert per_trial.shape[0] == kept.shape[0]

    agg = per_trial.group_by(["sweep", "point_value", "method"]).agg(
        power_mean=pl.col("power").mean(), power_std=pl.col("power").std(),
        fdr_mean=pl.col("fdr").mean(), fdr_std=pl.col("fdr").std(),
        power_cat_mean=pl.col("power_categorical").mean(), power_cat_std=pl.col("power_categorical").std(),
        fdr_cat_mean=pl.col("fdr_categorical").mean(), fdr_cat_std=pl.col("fdr_categorical").std(),
        f1_mean=pl.col("f1").mean(), f1_std=pl.col("f1").std(),
    )

    method_order = sorted(trial_covariates["method"].unique().to_list())

    def emit_table_html(sweep_idn: str, points: list[float], point_label: str) -> str:
        lines = [
            "<table>",
            f'<tr><th>{point_label}</th><th>method</th><th>power</th><th>fdr</th>'
            "<th>power_categorical</th><th>fdr_categorical</th><th>f1</th></tr>",
        ]
        for point in points:
            cell = agg.filter((pl.col("sweep") == sweep_idn) & (pl.col("point_value") == point))
            cell_by_method = {r["method"]: r for r in cell.to_dicts()}
            for i, meth in enumerate(method_order):
                r = cell_by_method[meth]
                td_point = f'<td rowspan="{n_methods}">{point:g}</td>' if i == 0 else ""
                lines.append(
                    f"<tr>{td_point}<td>{meth}</td>"
                    f'<td>{r["power_mean"]:.3f} ± {r["power_std"]:.3f}</td>'
                    f'<td>{r["fdr_mean"]:.3f} ± {r["fdr_std"]:.3f}</td>'
                    f'<td>{r["power_cat_mean"]:.3f} ± {r["power_cat_std"]:.3f}</td>'
                    f'<td>{r["fdr_cat_mean"]:.3f} ± {r["fdr_cat_std"]:.3f}</td>'
                    f'<td>{r["f1_mean"]:.3f} ± {r["f1_std"]:.3f}</td></tr>'
                )
        lines.append("</table>")
        return "\n".join(lines)

    tables = {}
    for sweep_idn, label, points in SWEEPS:
        html = emit_table_html(sweep_idn, points, label)
        tables[sweep_idn] = html
        p(f"\n--- {sweep_idn} table ---\n{html}")

    overall = per_trial.group_by("method").agg(
        power_mean=pl.col("power").mean(),
        fdr_mean=pl.col("fdr").mean(),
        power_cat_mean=pl.col("power_categorical").mean(),
        f1_mean=pl.col("f1").mean(),
    ).sort("method")
    p(f"\n--- overall (across all {cell_counts.shape[0]} cells) ---\n{overall}")

    OUT_JSON.write_text(json.dumps({
        "log": log,
        "tables_html": tables,
        "agg": agg.sort(["sweep", "point_value", "method"]).to_dicts(),
        "overall": overall.to_dicts(),
    }, indent=2))
    p(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
