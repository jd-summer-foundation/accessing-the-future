"""Tests for the appendix trend-schedule table exporter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import au_housing_disability_monte_carlo as eng
from scripts.pipeline_utils import DEFAULT_BASELINE_CONFIG, DEFAULT_MODEL_INPUT_CSV
from scripts.trend_schedule_table import generate_trend_tables

HORIZON_YEARS = 20
START_YEAR = 2022

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_INPUT_CSV.exists(),
    reason="processed model inputs not built",
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    reports_dir = tmp_path_factory.mktemp("reports")
    generate_trend_tables(DEFAULT_BASELINE_CONFIG, None, reports_dir)
    return reports_dir


@pytest.fixture(scope="module")
def schedule_df(generated: Path) -> pd.DataFrame:
    return pd.read_csv(generated / "tables" / "table_a2_trend_schedules.csv")


def test_all_artifacts_written(generated: Path) -> None:
    tables_dir = generated / "tables"
    for name in [
        "table_a1_trend_parameters.csv",
        "table_a1_trend_parameters.md",
        "table_a2_trend_schedules.csv",
        "table_a2_trend_schedules.md",
    ]:
        assert (tables_dir / name).exists(), name
    assert (generated / "trend_tables_manifest.json").exists()


def test_schedule_covers_all_trends_years_brackets(schedule_df: pd.DataFrame) -> None:
    assert set(schedule_df["trend"]) == set(eng.TREND_TYPES)
    assert len(schedule_df) == len(eng.TREND_TYPES) * (HORIZON_YEARS + 1) * len(eng.BRACKETS)
    for trend in eng.TREND_TYPES:
        subset = schedule_df.loc[schedule_df["trend"] == trend]
        assert sorted(subset["year_offset"].unique()) == list(range(HORIZON_YEARS + 1))
        assert set(subset["age_bracket"]) == set(eng.BRACKETS)
    assert (schedule_df["calendar_year"] == START_YEAR + schedule_df["year_offset"]).all()


def test_year_zero_matches_model_input_base_rates(schedule_df: pd.DataFrame) -> None:
    inputs = pd.read_csv(DEFAULT_MODEL_INPUT_CSV)
    age_col = "Age of Reference Person (Years)"
    year_zero = schedule_df.loc[schedule_df["year_offset"] == 0]
    for _, row in inputs.loc[inputs[age_col].isin(eng.BRACKETS)].iterrows():
        bracket_rows = year_zero.loc[year_zero["age_bracket"] == row[age_col]]
        assert len(bracket_rows) == len(eng.TREND_TYPES)
        assert bracket_rows["any_disability_rate"].tolist() == pytest.approx(
            [float(row["DIP_any"]) / 100.0] * len(eng.TREND_TYPES)
        )
        assert bracket_rows["physical_disability_rate"].tolist() == pytest.approx(
            [float(row["DIP_physical"]) / 100.0] * len(eng.TREND_TYPES)
        )


def test_none_trend_rates_constant_over_horizon(schedule_df: pd.DataFrame) -> None:
    subset = schedule_df.loc[schedule_df["trend"] == "none"]
    per_bracket = subset.groupby("age_bracket")[["any_disability_rate", "physical_disability_rate"]].nunique()
    assert (per_bracket == 1).all().all()


def test_physical_never_exceeds_any(schedule_df: pd.DataFrame) -> None:
    assert (
        schedule_df["physical_disability_rate"] <= schedule_df["any_disability_rate"] + 1e-9
    ).all()


def test_schedule_matches_engine_projection(schedule_df: pd.DataFrame, generated: Path) -> None:
    """The exported rates must equal build_trend_schedule output exactly."""
    from run_from_excel import _extract_survey_rates_any, prepare_simulation_inputs
    from scripts.pipeline_utils import HIST_SURVEY_YEARS, load_model_inputs

    df = load_model_inputs(DEFAULT_MODEL_INPUT_CSV)
    df_raw = pd.read_csv(DEFAULT_MODEL_INPUT_CSV)
    prepared = prepare_simulation_inputs(df)
    historical_any = _extract_survey_rates_any(df_raw, HIST_SURVEY_YEARS)

    for trend in eng.TREND_TYPES:
        schedule = eng.build_trend_schedule(
            prepared["rates"],
            historical_any[START_YEAR],
            trend,
            horizon_years=HORIZON_YEARS,
            historical_any=historical_any,
        )
        subset = schedule_df.loc[schedule_df["trend"] == trend]
        for snapshot in schedule:
            offset = int(snapshot.time_year)
            year_rows = subset.loc[subset["year_offset"] == offset].set_index("age_bracket")
            for bracket in eng.BRACKETS:
                assert year_rows.loc[bracket, "any_disability_rate"] == pytest.approx(
                    snapshot.rates.any_dis.by_bracket[bracket]
                )
                assert year_rows.loc[bracket, "physical_disability_rate"] == pytest.approx(
                    snapshot.rates.motor_phys.by_bracket[bracket]
                )
