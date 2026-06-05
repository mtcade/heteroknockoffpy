#
#//  tests/test_knockoff_categories.py
#//  heteroknockoffpy
#//
#//  Verifies that knockoff generation produces correct categories — both as
#//  literal pl.Categorical values and as consistent one-hot encodings — even
#//  when the knockoff procedure never selects every possible category value.
#
#  Note on polars Categorical vocabulary:
#  Polars 1.x uses a process-wide global string cache, so cat.get_categories()
#  on any series reflects all strings ever registered, not just those in that
#  column.  Tests therefore verify consistency through OHE output shapes and
#  values rather than relying on cat.get_categories().
#

import numpy as np
import polars as pl
import pytest

from heteroknockoffpy.utilities import (
    collapse_ohe,
    get_ohe_df,
    get_ohe_np,
    get_oheDict,
    makeChoices_ohe,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cat_series(name: str, values: list[str]) -> pl.Series:
    return pl.Series(name=name, values=values, dtype=pl.Utf8).cast(pl.Categorical)


def _x_three_cats(n: int = 12) -> pl.DataFrame:
    """X with one categorical column cycling through '0','1','2'."""
    vals = ["0", "1", "2"] * (n // 3) + ["0", "1", "2"][: n % 3]
    return pl.DataFrame([_cat_series("col", vals)])


def _x_cats_from(categories: list[str], n: int = 12) -> pl.DataFrame:
    """X with one categorical column uniformly cycling through given categories."""
    vals = (categories * ((n // len(categories)) + 1))[:n]
    return pl.DataFrame([_cat_series("col", vals)])


def _ohe_always_picks(category_index: int, n: int, n_cats: int, drop_first: bool) -> np.ndarray:
    """OHE weight matrix that always selects the given category index.

    With drop_first=True the first category is implicit; the array has
    n_cats-1 columns representing categories 1..n_cats-1.  Set the column
    for category_index (adjusted for drop_first) to a large weight.
    """
    n_cols = n_cats - 1 if drop_first else n_cats
    arr = np.zeros((n, n_cols))
    col = category_index - 1 if drop_first else category_index
    if 0 <= col < n_cols:
        arr[:, col] = 1.0
    return arr


def _x_categories(X: pl.DataFrame) -> dict[str, list[str]]:
    """Return a categories_override dict from X's actual present values."""
    return {
        col: sorted(X[col].cast(pl.Utf8).unique().drop_nulls().to_list())
        for col in X.columns
        if X.schema[col] == pl.Categorical
    }


# ── makeChoices_ohe: literal values ──────────────────────────────────────────

class TestMakeChoicesOheLiterals:
    def _categories(self) -> pl.Series:
        return pl.Series(["0", "1", "2"], dtype=pl.Utf8)

    def test_all_values_are_valid_categories(self):
        cats = self._categories()
        weights = np.array([[0.1, 0.7, 0.2]] * 8)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=False)
        valid = set(cats.to_list())
        assert all(v in valid for v in result.cast(pl.Utf8).to_list())

    def test_dtype_is_categorical(self):
        cats = self._categories()
        weights = np.array([[0.1, 0.7, 0.2]] * 5)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=False)
        assert result.dtype == pl.Categorical

    def test_max_picks_highest_weight_column(self):
        cats = self._categories()
        # Always pick index 2 ("2")
        weights = np.array([[0.0, 0.0, 1.0]] * 6)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=False)
        assert all(v == "2" for v in result.cast(pl.Utf8).to_list())

    def test_max_with_drop_first_picks_correct_category(self):
        cats = self._categories()
        # drop_first=True: input has 2 cols representing cats 1 and 2.
        # Setting col 0 high picks cat 1.
        weights = np.array([[1.0, 0.0]] * 6)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=True)
        assert all(v == "1" for v in result.cast(pl.Utf8).to_list())

    def test_max_drop_first_first_cat_recoverable(self):
        cats = self._categories()
        # drop_first=True: both cols near 0 → 1 - sum ≈ 1 → cat 0 wins
        weights = np.array([[0.0, 0.0]] * 6)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=True)
        assert all(v == "0" for v in result.cast(pl.Utf8).to_list())

    def test_length_matches_input_rows(self):
        cats = self._categories()
        n = 10
        weights = np.random.default_rng(0).random((n, 3))
        result = makeChoices_ohe(weights, cats, method="max", drop_first=False)
        assert len(result) == n

    def test_name_applied_to_series(self):
        cats = self._categories()
        weights = np.array([[0.1, 0.7, 0.2]] * 4)
        result = makeChoices_ohe(weights, cats, name="mycol", method="max", drop_first=False)
        assert result.name == "mycol"

    def test_when_some_categories_never_selected_values_are_still_valid(self):
        cats = self._categories()
        # Always pick "2" — "0" and "1" never chosen
        weights = np.array([[0.0, 0.0, 1.0]] * 10)
        result = makeChoices_ohe(weights, cats, method="max", drop_first=False)
        valid = set(cats.to_list())
        assert all(v in valid for v in result.cast(pl.Utf8).to_list())
        assert all(v == "2" for v in result.cast(pl.Utf8).to_list())


# ── collapse_ohe: literal categories ─────────────────────────────────────────

class TestCollapseOheLiterals:
    def test_categorical_values_are_valid_categories_from_x(self):
        X = _x_three_cats(12)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        valid = set(X["col"].cast(pl.Utf8).unique().to_list())
        assert all(v in valid for v in Xk["col"].cast(pl.Utf8).to_list())

    def test_result_has_categorical_dtype(self):
        X = _x_three_cats(9)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        assert Xk["col"].dtype == pl.Categorical

    def test_result_schema_matches_x(self):
        X = _x_three_cats(9)
        Xk_ohe = _ohe_always_picks(2, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        assert Xk.schema == X.schema

    def test_result_row_count_matches_x(self):
        X = _x_three_cats(15)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        assert len(Xk) == len(X)

    def test_correct_category_selected_by_max(self):
        X = _x_three_cats(6)
        # Always pick category "2" (index 2, col 1 with drop_first=True)
        Xk_ohe = _ohe_always_picks(2, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        assert all(v == "2" for v in Xk["col"].cast(pl.Utf8).to_list())

    def test_when_category_never_chosen_values_are_still_valid(self):
        """Even when Xk never picks '0', all values should be valid X categories."""
        X = _x_three_cats(12)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)
        valid = set(X["col"].cast(pl.Utf8).unique().to_list())
        assert all(v in valid for v in Xk["col"].cast(pl.Utf8).to_list())
        assert "0" not in set(Xk["col"].cast(pl.Utf8).to_list())


# ── OHE consistency via categories_override ──────────────────────────────────
#
# When Xk is missing some categories, callers must provide X's category list
# via categories_override so that the OHE arrays have the same column layout.
# These tests verify that pattern works correctly.

class TestOheConsistency:
    def test_ohe_np_same_width_with_override(self):
        X = _x_three_cats(12)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)

        override = _x_categories(X)
        x_width  = get_ohe_np(X,  drop_first=True).shape[1]
        xk_width = get_ohe_np(Xk, drop_first=True, categories_override=override).shape[1]
        assert x_width == xk_width

    def test_ohe_np_width_without_override_reflects_only_present_categories(self):
        """Without override, Xk only produces columns for its present categories."""
        X = _x_three_cats(12)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)

        # Xk only has "1", so without override: 1 cat, drop_first drops the only
        # dummy → 0 columns.  This is expected behaviour when no override given.
        xk_width_no_override = get_ohe_np(Xk, drop_first=True).shape[1]
        assert xk_width_no_override == 0

    def test_ohedict_same_tuple_length_with_override(self):
        X = _x_three_cats(12)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)

        x_cats = _x_categories(X)
        Xk_renamed = Xk.rename({"col": "col~"})
        X_all = pl.concat([X, Xk_renamed], how="horizontal")

        override = {**x_cats, **{"col~": x_cats["col"]}}
        d = get_oheDict(X_all, drop_first=True, categories_override=override)
        assert len(d["col"]) == len(d["col~"])

    def test_ohe_values_correct_when_xk_always_picks_one_category(self):
        """With override, OHE of Xk should be all-zero except for the chosen category."""
        X = _x_three_cats(6)
        Xk_ohe_in = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe_in, method="max", drop_first=True)

        override = _x_categories(X)
        Xk_ohe_out = get_ohe_df(Xk, drop_first=True, categories_override=override)
        # drop_first drops "col_0"; remaining cols are "col_1" and "col_2"
        assert "col_1" in Xk_ohe_out.columns
        assert "col_2" in Xk_ohe_out.columns
        assert all(v == 1 for v in Xk_ohe_out["col_1"].to_list())
        assert all(v == 0 for v in Xk_ohe_out["col_2"].to_list())

    def test_ohe_zero_columns_for_categories_absent_from_xk(self):
        """Dummy columns for categories absent from Xk should be all zeros."""
        X = _x_three_cats(12)
        Xk_ohe_in = _ohe_always_picks(2, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe_in, method="max", drop_first=True)

        override = _x_categories(X)
        Xk_ohe_out = get_ohe_df(Xk, drop_first=True, categories_override=override)
        # "col_0" is dropped; "col_1" is absent from Xk → all zeros
        assert "col_1" in Xk_ohe_out.columns
        assert all(v == 0 for v in Xk_ohe_out["col_1"].to_list())
        # "col_2" is present in Xk → all ones
        assert "col_2" in Xk_ohe_out.columns
        assert all(v == 1 for v in Xk_ohe_out["col_2"].to_list())

    def test_x_all_ohedict_indices_contiguous_across_x_and_xk(self):
        X = _x_three_cats(9)
        Xk_ohe = _ohe_always_picks(1, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)

        x_cats = _x_categories(X)
        X_all = pl.concat([X, Xk.rename({"col": "col~"})], how="horizontal")
        override = {**x_cats, **{"col~": x_cats["col"]}}
        d = get_oheDict(X_all, drop_first=True, categories_override=override)

        x_indices  = set(d["col"])
        xk_indices = set(d["col~"])
        assert x_indices.isdisjoint(xk_indices)

    def test_override_order_in_ohe_matches_sorted_category_order(self):
        """Dummy columns should appear in sorted category order."""
        X = _x_cats_from(["0", "1", "2"], n=9)
        Xk_ohe = _ohe_always_picks(2, len(X), 3, drop_first=True)
        Xk = collapse_ohe(X, Xk_ohe, method="max", drop_first=True)

        override = {"col": ["0", "1", "2"]}
        result = get_ohe_df(Xk, drop_first=True, categories_override=override)
        # After dropping "col_0" (first lexical), remaining should be "col_1","col_2"
        assert result.columns == ["col_1", "col_2"]
