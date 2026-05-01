import argparse
from pathlib import Path

import au_housing_disability_monte_carlo as eng
from run_from_excel import _build_runtime

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
