#
#//  tests/test_ohe.py
#//  heteroknockoffpy
#//
#//  Tests for get_ohe_df, get_ohe_np, and get_oheDict: one-hot encoding of
#//  pl.Categorical columns, always dropping the first lexical (sorted) category.
#

import numpy as np
import polars as pl
import pytest

from heteroknockoffpy.utilities import get_ohe_df, get_ohe_np, get_oheDict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cat_series(name: str, values: list[str]) -> pl.Series:
    return pl.Series(name=name, values=values, dtype=pl.Utf8).cast(pl.Categorical)


def _make_df(**cols: list[str]) -> pl.DataFrame:
    return pl.DataFrame({name: _cat_series(name, vals) for name, vals in cols.items()})


def _make_mixed_df(cat_cols: dict[str, list[str]], num_cols: dict[str, list[float]]) -> pl.DataFrame:
    series = [_cat_series(name, vals) for name, vals in cat_cols.items()]
    series += [pl.Series(name=name, values=vals) for name, vals in num_cols.items()]
    return pl.DataFrame(series)


# ── get_ohe_df: drop_first=True ───────────────────────────────────────────────

class TestGetOheDfDropFirst:
    def test_binary_drops_first_lexical(self):
        df = _make_df(col=["a", "b", "a", "b"])
        result = get_ohe_df(df, drop_first=True)
        assert "col_a" not in result.columns
        assert "col_b" in result.columns

    def test_three_categories_drops_first_lexical(self):
        df = _make_df(col=["c", "a", "b", "c", "a"])
        result = get_ohe_df(df, drop_first=True)
        assert "col_a" not in result.columns
        assert "col_b" in result.columns
        assert "col_c" in result.columns

    def test_numeric_string_categories_drops_zero(self):
        # Categories "0", "1", "2" — first lexical is "0"
        df = _make_df(col=["0", "1", "2", "0", "1"])
        result = get_ohe_df(df, drop_first=True)
        assert "col_0" not in result.columns
        assert "col_1" in result.columns
        assert "col_2" in result.columns

    def test_column_count_is_n_categories_minus_one(self):
        df = _make_df(col=["x", "y", "z", "x", "z"])
        result = get_ohe_df(df, drop_first=True)
        assert result.shape[1] == 2  # 3 categories - 1

    def test_row_count_preserved(self):
        df = _make_df(col=["a", "b", "a", "c", "b"])
        result = get_ohe_df(df, drop_first=True)
        assert result.shape[0] == 5

    def test_values_are_binary(self):
        df = _make_df(col=["a", "b", "c", "a"])
        result = get_ohe_df(df, drop_first=True)
        for col in result.columns:
            assert set(result[col].to_list()).issubset({0, 1})

    def test_first_lexical_regardless_of_insertion_order(self):
        # Values added in reverse lexical order: "c", "b", "a"
        df = _make_df(col=["c", "b", "a", "c", "b"])
        result = get_ohe_df(df, drop_first=True)
        assert "col_a" not in result.columns
        assert "col_b" in result.columns
        assert "col_c" in result.columns

    def test_multiple_categorical_columns(self):
        df = _make_df(x=["a", "b", "a"], y=["1", "2", "1"])
        result = get_ohe_df(df, drop_first=True)
        assert "x_a" not in result.columns
        assert "x_b" in result.columns
        assert "y_1" not in result.columns
        assert "y_2" in result.columns

    def test_numeric_and_categorical_mixed(self):
        df = _make_mixed_df(
            cat_cols={"cat": ["a", "b", "c", "a"]},
            num_cols={"num": [1.0, 2.0, 3.0, 4.0]},
        )
        result = get_ohe_df(df, drop_first=True)
        assert "num" in result.columns
        assert "cat_a" not in result.columns
        assert "cat_b" in result.columns
        assert "cat_c" in result.columns
        assert result["num"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_correct_dummy_values_for_rows(self):
        df = _make_df(col=["b", "a", "c", "b"])
        result = get_ohe_df(df, drop_first=True)
        # "a" dropped; "b"=1 for rows 0,3; "c"=1 for row 2
        assert result["col_b"].to_list() == [1, 0, 0, 1]
        assert result["col_c"].to_list() == [0, 0, 1, 0]

    def test_single_category_produces_no_columns(self):
        # After dropping the only category, nothing remains for that variable
        df = _make_df(col=["a", "a", "a"])
        result = get_ohe_df(df, drop_first=True)
        assert result.shape[1] == 0

    def test_large_numeric_string_keys_lexical_order(self):
        # Lexical sort: "0" < "1" < "10" < "2" — "0" should be dropped, not "1" or "10"
        df = _make_df(col=["0", "1", "2", "10", "0", "1", "2", "10"])
        result = get_ohe_df(df, drop_first=True)
        assert "col_0" not in result.columns
        assert "col_1" in result.columns
        assert "col_2" in result.columns
        assert "col_10" in result.columns


# ── get_ohe_df: drop_first=False ─────────────────────────────────────────────

class TestGetOheDfNoDropFirst:
    def test_all_categories_present(self):
        df = _make_df(col=["a", "b", "c", "a"])
        result = get_ohe_df(df, drop_first=False)
        assert "col_a" in result.columns
        assert "col_b" in result.columns
        assert "col_c" in result.columns

    def test_column_count_equals_n_categories(self):
        df = _make_df(col=["x", "y", "z", "x"])
        result = get_ohe_df(df, drop_first=False)
        assert result.shape[1] == 3

    def test_no_categorical_columns_returns_unchanged(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = get_ohe_df(df, drop_first=False)
        assert result.equals(df)


# ── get_ohe_np ────────────────────────────────────────────────────────────────

class TestGetOheNp:
    def test_returns_numpy_array(self):
        df = _make_df(col=["a", "b", "c", "a"])
        result = get_ohe_np(df, drop_first=True)
        assert isinstance(result, np.ndarray)

    def test_shape_with_drop_first(self):
        df = _make_df(col=["a", "b", "c", "a"])
        result = get_ohe_np(df, drop_first=True)
        assert result.shape == (4, 2)  # 4 rows, 3 cats - 1

    def test_shape_without_drop_first(self):
        df = _make_df(col=["a", "b", "c", "a"])
        result = get_ohe_np(df, drop_first=False)
        assert result.shape == (4, 3)

    def test_mixed_shape(self):
        df = _make_mixed_df(
            cat_cols={"cat": ["a", "b", "c", "a"]},
            num_cols={"num": [1.0, 2.0, 3.0, 4.0]},
        )
        result = get_ohe_np(df, drop_first=True)
        # 1 numeric col + 2 dummy cols (3 cats - 1)
        assert result.shape == (4, 3)


# ── get_oheDict ───────────────────────────────────────────────────────────────

class TestGetOheDict:
    def test_categorical_maps_to_tuple(self):
        df = _make_df(col=["a", "b", "c", "a"])
        d = get_oheDict(df, drop_first=True)
        assert isinstance(d["col"], tuple)
        assert len(d["col"]) == 2  # 3 cats - 1

    def test_numeric_maps_to_int(self):
        df = _make_mixed_df(cat_cols={}, num_cols={"x": [1.0, 2.0]})
        d = get_oheDict(df, drop_first=True)
        assert isinstance(d["x"], int)

    def test_column_indices_are_contiguous(self):
        df = _make_mixed_df(
            cat_cols={"cat": ["a", "b", "c", "a"]},
            num_cols={"num": [1.0, 2.0, 3.0, 4.0]},
        )
        d = get_oheDict(df, drop_first=True)
        # cat → (0, 1), num → 2
        assert d["cat"] == (0, 1)
        assert d["num"] == 2

    def test_drop_first_false_includes_all_cats(self):
        df = _make_df(col=["a", "b", "c", "a"])
        d = get_oheDict(df, drop_first=False)
        assert len(d["col"]) == 3

    def test_ohedict_width_matches_ohe_np_width(self):
        df = _make_mixed_df(
            cat_cols={"cat": ["a", "b", "c", "a"]},
            num_cols={"num": [1.0, 2.0, 3.0, 4.0]},
        )
        ohe_np = get_ohe_np(df, drop_first=True)
        d = get_oheDict(df, drop_first=True)
        expected_width = sum(
            len(v) if isinstance(v, tuple) else 1 for v in d.values()
        )
        assert ohe_np.shape[1] == expected_width

    def test_two_categoricals_indices_non_overlapping(self):
        df = _make_df(x=["a", "b", "a"], y=["1", "2", "1"])
        d = get_oheDict(df, drop_first=True)
        x_indices = set(d["x"])
        y_indices = set(d["y"])
        assert x_indices.isdisjoint(y_indices)
