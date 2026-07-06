import argparse
from pathlib import Path

import au_housing_disability_monte_carlo as eng
import pandas as pd
import pytest
from run_from_excel import _build_runtime, _build_scaled_rates, _uncertainty_cases, prepare_simulation_inputs
from scripts.pipeline_utils import AGE_COL, RATE_ANY_COL, RATE_MOTOR_COL, TENURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _args(config: Path, *, horizon_years: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        input=ROOT / "data/processed/model_inputs.csv",
        sheet=None,
        outdir=None,
        n_props=None,
        seed=None,
        horizon_years=horizon_years,
        verbose=False,
    )


def test_runtime_uses_engine_default_when_config_omits_horizon(tmp_path: Path) -> None:
    config = tmp_path / "no_horizon.yaml"
    config.write_text(
        """
run:
  name: no_horizon
  seed: 123
  n_props: 10
""".lstrip(),
        encoding="utf-8",
    )

    runtime = _build_runtime(_args(config))

    assert runtime["horizon_years"] == eng.DEFAULT_HORIZON_YEARS


def test_runtime_horizon_cli_override_wins(tmp_path: Path) -> None:
    config = tmp_path / "configured_horizon.yaml"
    config.write_text(
        """
run:
  name: configured_horizon
  seed: 123
  n_props: 10
  horizon_years: 20
""".lstrip(),
        encoding="utf-8",
    )

    runtime = _build_runtime(_args(config, horizon_years=50))

    assert runtime["horizon_years"] == 50


def test_runtime_reads_age_transition_mode(tmp_path: Path) -> None:
    config = tmp_path / "annual_mode.yaml"
    config.write_text(
        """
run:
  name: annual_mode
  seed: 123
  n_props: 10
  age_transition_mode: annual_interpolated
""".lstrip(),
        encoding="utf-8",
    )

    runtime = _build_runtime(_args(config))

    assert runtime["age_transition_mode"] == "annual_interpolated"


def test_uncertainty_cases_expand_only_when_moes_are_available() -> None:
    assert _uncertainty_cases({}) == ["base"]
    assert _uncertainty_cases({RATE_ANY_COL: {"15-24": 0.039}}) == ["low", "base", "high"]


def test_scaled_rates_apply_low_and_high_moe_bounds() -> None:
    base_maps = {
        RATE_ANY_COL: {bracket: 0.235 for bracket in eng.BRACKETS},
        RATE_MOTOR_COL: {bracket: 0.136 for bracket in eng.BRACKETS},
    }
    moe_maps = {
        RATE_ANY_COL: {bracket: 0.039 for bracket in eng.BRACKETS},
        RATE_MOTOR_COL: {bracket: 0.039 for bracket in eng.BRACKETS},
    }

    low = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="low")
    base = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="base")
    high = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="high")

    assert low.any_dis.by_bracket["15-24"] == pytest.approx(0.196)
    assert base.any_dis.by_bracket["15-24"] == pytest.approx(0.235)
    assert high.any_dis.by_bracket["15-24"] == pytest.approx(0.274)


def test_prepare_simulation_inputs_converts_sub_one_percent_tenure_values() -> None:
    base_row = {
        RATE_ANY_COL: 10.0,
        RATE_MOTOR_COL: 5.0,
        "Of those who have length of tenure <1 year, what proportion are in each age bucket?": 1.0 / len(eng.BRACKETS),
    }
    base_row.update({column: 0.0 for column in TENURE_COLUMNS})

    rows = []
    for bracket in eng.BRACKETS:
        row = {AGE_COL: bracket, **base_row}
        if bracket == "25-34":
            row.update(
                {
                    "Length of tenure: less than 1 year  (% of all households, not count)": 26.0,
                    "Length of tenure: 1-2 years": 19.7,
                    "Length of tenure: 2-3 years": 17.4,
                    "Length of tenure: 3-4 years": 20.9,
                    "Length of tenure: 5-9 years": 13.0,
                    "Length of tenure: 10-19 years": 2.4,
                    "Length of tenure: 20+ years": 0.5,
                }
            )
        else:
            row["Length of tenure: less than 1 year  (% of all households, not count)"] = 100.0
        rows.append(row)

    df = pd.DataFrame(rows)

    prepared = prepare_simulation_inputs(df)

    assert prepared["tenure"].probs_by_bracket["25-34"][-1] == pytest.approx(0.005005005005005005)
