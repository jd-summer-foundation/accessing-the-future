#!/usr/bin/env python3
"""Time from build to first disability occupancy, for cashflow/retrofit-cost analysis.

Runs the central specification (in-mover seeding, no rate trend, unchanged
tenure) and reports, for each disability category and uncertainty case:

  - the probability the dwelling is ever occupied by such a household in the
    horizon (consistency check against the headline results)
  - years from build to the FIRST such occupancy, conditional on it happening
    within the horizon (mean, median, quartiles)
  - the split between "household already had the condition when it moved in"
    and "a resident household acquired it in place"
  - a year-by-year cumulative distribution: share of all dwellings whose first
    disability occupancy has happened by the end of year t

The cumulative distribution is the input a discounted-cashflow comparison of
build-in vs retrofit actually needs; the conditional mean alone is censored at
the horizon and excludes the ~40% of dwellings with no such occupancy.

Outputs:
  results/first_occupancy/first_occupancy_summary.csv
  results/first_occupancy/first_occupancy_cdf.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import au_housing_disability_monte_carlo as eng
from run_from_excel import (
    _build_scaled_rates,
    _extract_survey_rates_any,
    _uncertainty_cases,
    prepare_simulation_inputs,
)
from scripts.pipeline_utils import (
    DEFAULT_BASELINE_CONFIG,
    HIST_SURVEY_YEARS,
    load_model_inputs,
    load_yaml,
    resolve_input_path,
)

CENTRAL_SCENARIO = {"name": "main_inmover_first", "disabled_tenure_factor": 1.0, "rate_scale": 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--input", type=Path, default=None, help="Processed model inputs CSV.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/first_occupancy")
    parser.add_argument("--n-props", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon-years", type=int, default=None)
    return parser.parse_args()


def _first_occupancy_stats(times: np.ndarray, moved_in: np.ndarray, horizon: int) -> dict:
    occurred = np.isfinite(times)
    t = times[occurred]
    return {
        "p_ever": float(occurred.mean()),
        "mean_years_to_first": float(t.mean()),
        "median_years_to_first": float(np.median(t)),
        "p25_years_to_first": float(np.percentile(t, 25)),
        "p75_years_to_first": float(np.percentile(t, 75)),
        "share_first_at_build": float((t == 0.0).mean()),
        "share_first_moved_in_with": float(moved_in[occurred].mean()),
        "share_first_acquired_in_place": float(1.0 - moved_in[occurred].mean()),
        "horizon_years": int(horizon),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) if args.config and args.config.exists() else {}
    run_cfg = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}

    input_path = resolve_input_path(args.input or ROOT / str(run_cfg.get("input_path", "data/processed/model_inputs.csv")))
    n_props = args.n_props if args.n_props is not None else int(run_cfg.get("n_props", 50_000))
    seed = args.seed if args.seed is not None else int(run_cfg.get("seed", 123))
    horizon = args.horizon_years if args.horizon_years is not None else int(run_cfg.get("horizon_years", eng.DEFAULT_HORIZON_YEARS))
    start_year = int(run_cfg.get("start_year", 2022))
    age_transition_mode = str(run_cfg.get("age_transition_mode", "bracket_boundary"))

    df = load_model_inputs(input_path, sheet=0)
    df_raw = pd.read_csv(input_path)
    prepared = prepare_simulation_inputs(df)
    historical_any = _extract_survey_rates_any(df_raw, HIST_SURVEY_YEARS)

    params = eng.SimParams(
        n_props=n_props,
        seed=seed,
        horizon_years=horizon,
        disabled_tenure_factor=1.0,
        age_transition_mode=age_transition_mode,
    )
    print(f"age_transition_mode: {age_transition_mode}")

    summary_rows = []
    cdf_rows = []
    for uncertainty_case in _uncertainty_cases(prepared["rate_moe_maps"]):
        scenario_rates = _build_scaled_rates(
            prepared["base_rate_maps"],
            CENTRAL_SCENARIO,
            rate_moe_maps=prepared["rate_moe_maps"],
            uncertainty_case=uncertainty_case,
        )
        prebuilt = eng.build_trend_schedule(
            scenario_rates,
            historical_any[start_year],
            "none",
            horizon_years=horizon,
            historical_any=historical_any,
        )
        result = eng.run_sim(
            params,
            scenario_rates,
            prepared["tenure"],
            prepared["inmovers"],
            return_time_stats=True,
            prebuilt_schedule=prebuilt,
            return_first_occupancy=True,
        )
        first = result["first_occupancy"]

        for category, time_key, moved_key in (
            ("physical", "phys_time", "phys_moved_in"),
            ("any", "any_time", "any_moved_in"),
        ):
            times = first[time_key]
            stats = _first_occupancy_stats(times, first[moved_key], horizon)
            summary_rows.append({
                "scenario": f"{CENTRAL_SCENARIO['name']}_{uncertainty_case}",
                "uncertainty_case": uncertainty_case,
                "age_transition_mode": age_transition_mode,
                "category": category,
                **stats,
            })
            for year in range(horizon + 1):
                cdf_rows.append({
                    "uncertainty_case": uncertainty_case,
                    "category": category,
                    "year": year,
                    "cum_share_first_occupancy": float(np.mean(times <= year)),
                })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    cdf_df = pd.DataFrame(cdf_rows)
    summary_df.to_csv(outdir / "first_occupancy_summary.csv", index=False)
    cdf_df.to_csv(outdir / "first_occupancy_cdf.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))
    print()
    base_cdf = cdf_df[cdf_df["uncertainty_case"] == "base"].pivot(index="year", columns="category", values="cum_share_first_occupancy")
    print("Cumulative share of dwellings whose first disability occupancy has happened by end of year t (base case):")
    print(base_cdf.to_string())
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
