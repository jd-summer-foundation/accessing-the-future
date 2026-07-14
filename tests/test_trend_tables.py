import numpy as np
import pandas as pd
import pytest

import au_housing_disability_monte_carlo as eng
from scripts.pipeline_utils import AGE_COL, HIST_RATE_ANY_COL_PREFIX, RATE_ANY_COL, RATE_MOTOR_COL
from scripts.trend_tables import build_trend_table


def _synthetic_inputs() -> pd.DataFrame:
    rows = []
    for index, bracket in enumerate(eng.BRACKETS):
        base = 0.10 + 0.08 * index
        rows.append(
            {
                AGE_COL: bracket,
                RATE_ANY_COL: (base + 0.10) * 100.0,
                RATE_MOTOR_COL: (base + 0.05) * 100.0,
                f"{HIST_RATE_ANY_COL_PREFIX}2003": base - 0.02,
                f"{HIST_RATE_ANY_COL_PREFIX}2015": base - 0.01,
                f"{HIST_RATE_ANY_COL_PREFIX}2022": base,
            }
        )
    return pd.DataFrame(rows)


def test_relative_changes_match_engine_schedule() -> None:
    df = _synthetic_inputs()
    table = build_trend_table(df).set_index("age_bracket")

    hh_any = {b: float(df.loc[df[AGE_COL] == b, RATE_ANY_COL].iloc[0]) / 100.0 for b in eng.BRACKETS}
    hh_motor = {b: float(df.loc[df[AGE_COL] == b, RATE_MOTOR_COL].iloc[0]) / 100.0 for b in eng.BRACKETS}
    person_any = {
        year: {b: float(df.loc[df[AGE_COL] == b, f"{HIST_RATE_ANY_COL_PREFIX}{year}"].iloc[0]) for b in eng.BRACKETS}
        for year in (2003, 2015, 2022)
    }
    rates = eng.AllRates(any_dis=eng.Rates(hh_any), motor_phys=eng.Rates(hh_motor))

    for trend, (prior_year, _window) in eng._LINEAR_TREND_WINDOWS.items():
        schedule = eng.build_trend_schedule(
            rates,
            person_any[2022],
            trend,
            horizon_years=2,
            historical_any=person_any,
        )
        column = f"annual_change_{prior_year}_2022_pct_per_yr"
        for bracket in eng.BRACKETS:
            engine_increment = (
                schedule[1].rates.any_dis.by_bracket[bracket]
                - schedule[0].rates.any_dis.by_bracket[bracket]
            )
            expected_relative = engine_increment / hh_any[bracket] * 100.0
            assert table.loc[bracket, column] == pytest.approx(expected_relative, abs=1e-9)


def test_level_columns_are_percentages() -> None:
    df = _synthetic_inputs()
    table = build_trend_table(df).set_index("age_bracket")
    for bracket in eng.BRACKETS:
        person_2022 = float(df.loc[df[AGE_COL] == bracket, f"{HIST_RATE_ANY_COL_PREFIX}2022"].iloc[0])
        assert table.loc[bracket, "any_2022_pct"] == pytest.approx(person_2022 * 100.0)
        assert table.loc[bracket, "hh_any_2022_pct"] == pytest.approx(
            float(df.loc[df[AGE_COL] == bracket, RATE_ANY_COL].iloc[0])
        )


def test_missing_history_column_raises() -> None:
    df = _synthetic_inputs().drop(columns=[f"{HIST_RATE_ANY_COL_PREFIX}2015"])
    with pytest.raises(ValueError, match="2015"):
        build_trend_table(df)
