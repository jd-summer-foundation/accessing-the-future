import pandas as pd
import pytest

from scripts.build_model_inputs import build_model_inputs

from scripts.pipeline_utils import (
    AGE_COL,
    DEFAULT_DERIVATION_CONFIG,
    DEFAULT_HOUSING_MOBILITY_WORKBOOK,
    DEFAULT_LEGACY_ORACLE,
    INMOVER_COL,
    TENURE_COLUMNS,
    load_model_inputs,
)

HOUSING_MOBILITY_AGE_LABELS = {
    "15-24": "15 to 24",
    "25-34": "25 to 34",
    "35-44": "35 to 44",
    "45-54": "45 to 54",
    "55-64": "55 to 64",
    "65-74": "65 to 74",
    "75+": "75 and over",
}


def _load_housing_mobility_rows() -> dict[str, pd.Series]:
    table = pd.read_excel(DEFAULT_HOUSING_MOBILITY_WORKBOOK, sheet_name="Table 2.2", header=None)
    rows: dict[str, pd.Series] = {}
    labels = table.iloc[:, 0].astype(str).str.strip()
    for age_bracket, raw_label in HOUSING_MOBILITY_AGE_LABELS.items():
        matches = table.loc[labels == raw_label]
        assert not matches.empty, f"Missing housing mobility row for {raw_label}"
        rows[age_bracket] = matches.iloc[0]
    return rows


def test_housing_mobility_age_rows_are_present() -> None:
    rows = _load_housing_mobility_rows()
    assert list(rows.keys()) == list(HOUSING_MOBILITY_AGE_LABELS.keys())


def test_tenure_columns_match_legacy_oracle() -> None:
    generated = build_model_inputs(DEFAULT_DERIVATION_CONFIG)
    oracle = load_model_inputs(DEFAULT_LEGACY_ORACLE)
    for column in TENURE_COLUMNS:
        pd.testing.assert_series_equal(generated[column], oracle[column], check_names=False)


def test_inmover_distribution_is_derived_from_lt1_counts() -> None:
    generated = build_model_inputs(DEFAULT_DERIVATION_CONFIG)
    rows = _load_housing_mobility_rows()

    counts = {
        age_bracket: (float(row.iloc[1]) / 100.0) * float(row.iloc[9])
        for age_bracket, row in rows.items()
    }
    total = sum(counts.values())
    expected = {age_bracket: count / total for age_bracket, count in counts.items()}

    for age_bracket, expected_value in expected.items():
        actual_value = float(generated.loc[generated[AGE_COL] == age_bracket, INMOVER_COL].iloc[0])
        assert actual_value == pytest.approx(expected_value, rel=0, abs=1e-12)
