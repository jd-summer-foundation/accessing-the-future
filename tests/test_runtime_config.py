import argparse
from pathlib import Path

import au_housing_disability_monte_carlo as eng
import pytest
from run_from_excel import _build_runtime, _build_scaled_rates, _uncertainty_cases
from scripts.pipeline_utils import RATE_ANY_COL, RATE_MOTOR_COL, RATE_PHYS2_COL, RATE_SEVERE_COL

ROOT = Path(__file__).resolve().parents[1]


def _args(config: Path, *, horizon_years: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        input=ROOT / "data/processed/model_inputs.csv",
        excel=None,
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


def test_uncertainty_cases_expand_only_when_moes_are_available() -> None:
    assert _uncertainty_cases({}) == ["base"]
    assert _uncertainty_cases({RATE_ANY_COL: {"15-24": 0.039}}) == ["low", "base", "high"]


def test_scaled_rates_apply_low_and_high_moe_bounds() -> None:
    base_maps = {
        RATE_ANY_COL: {bracket: 0.235 for bracket in eng.BRACKETS},
        RATE_MOTOR_COL: {bracket: 0.136 for bracket in eng.BRACKETS},
        RATE_SEVERE_COL: {bracket: 0.068 for bracket in eng.BRACKETS},
        RATE_PHYS2_COL: {bracket: 0.396 for bracket in eng.BRACKETS},
    }
    moe_maps = {
        RATE_ANY_COL: {bracket: 0.039 for bracket in eng.BRACKETS},
        RATE_MOTOR_COL: {bracket: 0.039 for bracket in eng.BRACKETS},
        RATE_SEVERE_COL: {bracket: 0.024 for bracket in eng.BRACKETS},
        RATE_PHYS2_COL: {bracket: 0.047 for bracket in eng.BRACKETS},
    }

    low = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="low")
    base = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="base")
    high = _build_scaled_rates(base_maps, {"rate_scale": 1.0}, rate_moe_maps=moe_maps, uncertainty_case="high")

    assert low.any_dis.by_bracket["15-24"] == pytest.approx(0.196)
    assert base.any_dis.by_bracket["15-24"] == pytest.approx(0.235)
    assert high.any_dis.by_bracket["15-24"] == pytest.approx(0.274)
