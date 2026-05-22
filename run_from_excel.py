#!/usr/bin/env python3
"""Config-driven runner for the housing disability Monte Carlo model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import au_housing_disability_monte_carlo as eng
from scripts.pipeline_utils import (
    AGE_COL,
    DEFAULT_BASELINE_CONFIG,
    HIST_RATE_ANY_COL_PREFIX,
    HIST_SURVEY_YEARS,
    INMOVER_COL,
    RATE_COLUMNS,
    RATE_MOE_SUFFIX,
    RATE_ANY_COL,
    RATE_MOTOR_COL,
    TENURE_BUCKETS,
    TENURE_COLUMNS,
    artifact_checksums,
    dependency_versions,
    git_commit,
    load_model_inputs,
    load_yaml,
    manifest_path,
    resolve_input_path,
    serialise_for_json,
    sha256_file,
    utc_now_iso,
    write_json,
)


def _to_rate(value: object) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    rate = float(numeric)
    if rate > 1.0:
        rate /= 100.0
    return rate


def _normalise_dist(data: Dict[str, float], name: str) -> Dict[str, float]:
    values = np.array([float(data[bracket]) for bracket in eng.BRACKETS], dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{name}: distribution sums to 0 after cleaning: {data}")
    values = values / total
    return {bracket: float(values[index]) for index, bracket in enumerate(eng.BRACKETS)}


def _extract_bracketed_series(
    valid_rows: pd.DataFrame,
    age_col: str,
    series_name: str,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for bracket in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[age_col] == bracket]
        if row.empty or series_name not in valid_rows.columns:
            out[bracket] = float(fallback[bracket])
            continue
        value = _to_rate(row.iloc[0][series_name])
        out[bracket] = float(fallback[bracket]) if not np.isfinite(value) else float(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Australian housing disability Monte Carlo from processed model inputs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG, help="Path to YAML run configuration.")
    parser.add_argument("--input", type=Path, default=None, help="Processed input CSV or Excel workbook.")
    parser.add_argument("--sheet", default=None, help="Sheet name or index when reading Excel inputs.")
    parser.add_argument("--outdir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--n-props", type=int, default=None, help="Override number of dwellings to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument("--horizon-years", type=int, default=None, help="Override simulation horizon in years.")
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    return parser.parse_args()


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else SCRIPT_DIR / path


def _default_scenarios() -> List[Dict[str, object]]:
    return [
        {"name": "main_inmover_first", "disabled_tenure_factor": 1.0, "rate_scale": 1.0},
        {"name": "sens_disabled_tenure_1p2", "disabled_tenure_factor": 1.2, "rate_scale": 1.0},
        {"name": "selection_scale_0p9", "disabled_tenure_factor": 1.0, "rate_scale": 0.9},
    ]


def _scenario_title(name: str) -> str:
    return name.replace("_", " ")


def _build_runtime(args: argparse.Namespace) -> Dict[str, object]:
    config = load_yaml(args.config) if args.config and args.config.exists() else {}
    run_cfg = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}
    input_cfg = config.get("input", {}) if isinstance(config.get("input", {}), dict) else {}

    configured_input = _resolve_path(run_cfg.get("input_path")) if run_cfg else None
    input_path = resolve_input_path(args.input or configured_input)
    output_dir = args.outdir or _resolve_path(run_cfg.get("output_dir")) or SCRIPT_DIR / "results/baseline"

    runtime = {
        "config": config,
        "config_path": args.config,
        "input_path": input_path,
        "sheet": args.sheet if args.sheet is not None else input_cfg.get("sheet", 0),
        "output_dir": output_dir,
        "run_name": run_cfg.get("name", output_dir.name),
        "n_props": args.n_props if args.n_props is not None else int(run_cfg.get("n_props", 50_000)),
        "seed": args.seed if args.seed is not None else int(run_cfg.get("seed", 123)),
        "horizon_years": args.horizon_years
        if args.horizon_years is not None
        else int(run_cfg.get("horizon_years", eng.DEFAULT_HORIZON_YEARS)),
        "return_time_stats": bool(run_cfg.get("return_time_stats", True)),
        "scenarios": config.get("scenarios") or _default_scenarios(),
        "transition_model_config": config.get("transition_model"),
        "start_year": int(run_cfg.get("start_year", 2022)),
        "verbose": args.verbose,
        "cli_args": {key: value for key, value in vars(args).items() if value is not None and key != "verbose"},
    }
    return runtime


def prepare_simulation_inputs(df: pd.DataFrame) -> Dict[str, object]:
    valid_rows = df.loc[df[AGE_COL].isin(eng.BRACKETS)].copy()

    inmover_raw = {bracket: float(valid_rows.loc[valid_rows[AGE_COL] == bracket, INMOVER_COL].iloc[0]) for bracket in eng.BRACKETS}
    inmover_probs = _normalise_dist(inmover_raw, "inmover_probs")
    inmovers = eng.InMoverDist(inmover_probs)

    probs_by_bracket: Dict[str, List[float]] = {}
    for bracket in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[AGE_COL] == bracket]
        if row.empty:
            raise ValueError(f"Missing tenure row for bracket {bracket}")
        values = [_to_rate(row.iloc[0][column]) for column in TENURE_COLUMNS]
        arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
        total = float(arr.sum())
        if total <= 0:
            raise ValueError(f"Tenure probabilities sum to 0 for {bracket}: {values}")
        probs_by_bracket[bracket] = (arr / total).tolist()

    tenure = eng.TenureDist(buckets=TENURE_BUCKETS, probs_by_bracket=probs_by_bracket)

    fallback_any = {"15-24": 0.06, "25-34": 0.08, "35-44": 0.11, "45-54": 0.15, "55-64": 0.21, "65-74": 0.30, "75+": 0.48}
    fallback_motor = {"15-24": 0.01, "25-34": 0.015, "35-44": 0.02, "45-54": 0.04, "55-64": 0.06, "65-74": 0.10, "75+": 0.18}

    rates_any = _extract_bracketed_series(valid_rows, AGE_COL, RATE_ANY_COL, fallback_any)
    rates_motor = _extract_bracketed_series(valid_rows, AGE_COL, RATE_MOTOR_COL, fallback_motor)

    for bracket in eng.BRACKETS:
        rates_motor[bracket] = min(float(rates_motor[bracket]), float(rates_any[bracket]))

    base_rate_maps = {
        RATE_ANY_COL: rates_any,
        RATE_MOTOR_COL: rates_motor,
    }
    rate_moe_maps: Dict[str, Dict[str, float]] = {}
    for column_name in RATE_COLUMNS:
        moe_column = f"{column_name}{RATE_MOE_SUFFIX}"
        if moe_column not in valid_rows.columns:
            continue
        rate_moe_maps[column_name] = _extract_bracketed_series(
            valid_rows,
            AGE_COL,
            moe_column,
            {bracket: 0.0 for bracket in eng.BRACKETS},
        )
    rates = eng.AllRates(
        any_dis=eng.Rates(rates_any),
        motor_phys=eng.Rates(rates_motor),
    )

    inputs_df = pd.DataFrame(
        {
            "age_bracket": eng.BRACKETS,
            "any_rate_input": [rates_any[bracket] for bracket in eng.BRACKETS],
            "motor_phys_rate_input": [rates_motor[bracket] for bracket in eng.BRACKETS],
            "inmover_dist": [inmover_probs[bracket] for bracket in eng.BRACKETS],
        }
    )
    for column_name, moe_map in rate_moe_maps.items():
        inputs_df[f"{column_name}_moe_input"] = [moe_map[bracket] for bracket in eng.BRACKETS]
    for index, bucket in enumerate(TENURE_BUCKETS):
        inputs_df[f"tenure_prob_{bucket}"] = [tenure.probs_by_bracket[bracket][index] for bracket in eng.BRACKETS]

    profiles = eng.make_profiles(rates)
    profiles_df = pd.DataFrame(
        {
            "age_bracket": eng.BRACKETS,
            "adj_any": [profiles["adj_any"][bracket] for bracket in eng.BRACKETS],
            "cond_motor_phys_given_any": [profiles["cond_phys"][bracket] for bracket in eng.BRACKETS],
        }
    )

    return {
        "rates": rates,
        "base_rate_maps": base_rate_maps,
        "rate_moe_maps": rate_moe_maps,
        "tenure": tenure,
        "inmovers": inmovers,
        "inputs_df": inputs_df,
        "profiles_df": profiles_df,
    }


def _uncertainty_cases(rate_moe_maps: Dict[str, Dict[str, float]]) -> List[str]:
    return ["low", "base", "high"] if rate_moe_maps else ["base"]


def _apply_uncertainty_case(
    base_rate_maps: Dict[str, Dict[str, float]],
    rate_moe_maps: Dict[str, Dict[str, float]],
    uncertainty_case: str,
) -> Dict[str, Dict[str, float]]:
    if uncertainty_case not in {"low", "base", "high"}:
        raise ValueError(f"Unknown uncertainty case: {uncertainty_case}")
    if uncertainty_case == "base":
        return {
            column_name: {bracket: float(value) for bracket, value in values.items()}
            for column_name, values in base_rate_maps.items()
        }

    direction = -1.0 if uncertainty_case == "low" else 1.0
    adjusted: Dict[str, Dict[str, float]] = {}
    for column_name, values in base_rate_maps.items():
        moe_values = rate_moe_maps.get(column_name, {})
        adjusted[column_name] = {
            bracket: float(np.clip(float(value) + direction * float(moe_values.get(bracket, 0.0)), 0.0, 1.0))
            for bracket, value in values.items()
        }
    return adjusted


def _build_scaled_rates(
    base_rate_maps: Dict[str, Dict[str, float]],
    scenario: Dict[str, object],
    *,
    rate_moe_maps: Dict[str, Dict[str, float]] | None = None,
    uncertainty_case: str = "base",
) -> eng.AllRates:
    global_scale = float(scenario.get("rate_scale", 1.0))
    category_scales = scenario.get("rate_scales", {})
    if not isinstance(category_scales, dict):
        category_scales = {}
    case_rate_maps = _apply_uncertainty_case(base_rate_maps, rate_moe_maps or {}, uncertainty_case)

    def scaled(column_name: str) -> Dict[str, float]:
        scale = float(category_scales.get(column_name, global_scale))
        return {
            bracket: float(np.clip(scale * float(case_rate_maps[column_name][bracket]), 0.0, 1.0))
            for bracket in eng.BRACKETS
        }

    return eng.AllRates(
        any_dis=eng.Rates(scaled(RATE_ANY_COL)),
        motor_phys=eng.Rates(scaled(RATE_MOTOR_COL)),
    )


def _extract_survey_rates_any(
    df: pd.DataFrame,
    survey_years: List[int],
) -> Dict[int, Dict[str, float]]:
    """Extract historical any-disability rates from model inputs DataFrame."""
    result: Dict[int, Dict[str, float]] = {}
    for year in survey_years:
        col = f"{HIST_RATE_ANY_COL_PREFIX}{year}"
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in model inputs. "
                f"Run 'make build-data' to regenerate model_inputs.csv with historical rates."
            )
        bracket_rates: Dict[str, float] = {}
        for bracket in eng.BRACKETS:
            row = df.loc[df[AGE_COL] == bracket]
            if row.empty:
                raise ValueError(f"Missing bracket {bracket!r} in inputs")
            bracket_rates[bracket] = _to_rate(row.iloc[0][col])
        result[year] = bracket_rates
    return result


def _write_manifest(runtime: Dict[str, object], scenario_summaries: pd.DataFrame) -> None:
    input_path = runtime["input_path"]
    config_path = runtime["config_path"]
    output_dir = runtime["output_dir"]
    output_paths = [
        Path(output_dir) / "scenario_summaries.csv",
        Path(output_dir) / "inputs_used.csv",
        Path(output_dir) / "profiles_used.csv",
    ]

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "run_name": runtime["run_name"],
        "git_commit": git_commit(SCRIPT_DIR),
        "python_version": sys.version.split()[0],
        "dependency_versions": dependency_versions(),
        "input": {
            "path": input_path,
            "sha256": sha256_file(input_path),
            "sheet": runtime["sheet"],
        },
        "config": {
            "path": config_path,
            "sha256": sha256_file(config_path) if isinstance(config_path, Path) and config_path.exists() else None,
            "values": serialise_for_json(runtime["config"]),
        },
        "runtime": {
            "output_dir": output_dir,
            "n_props": runtime["n_props"],
            "seed": runtime["seed"],
            "horizon_years": runtime["horizon_years"],
            "return_time_stats": runtime["return_time_stats"],
        },
        "cli_overrides": runtime["cli_args"],
        "scenarios": serialise_for_json(runtime["scenarios"]),
        "resolved_scenarios": serialise_for_json(runtime.get("resolved_scenarios", [])),
        "outputs": {
            "directory": output_dir,
            "rows": int(len(scenario_summaries)),
            "scenario_summaries": {
                "path": manifest_path(output_paths[0], relative_to=Path(output_dir)),
                "sha256": sha256_file(output_paths[0]),
            },
            "inputs_used": {
                "path": manifest_path(output_paths[1], relative_to=Path(output_dir)),
                "sha256": sha256_file(output_paths[1]),
            },
            "profiles_used": {
                "path": manifest_path(output_paths[2], relative_to=Path(output_dir)),
                "sha256": sha256_file(output_paths[2]),
            },
            "artifact_checksums": artifact_checksums(output_paths, relative_to=Path(output_dir)),
        },
    }
    write_json(Path(output_dir) / "run_manifest.json", manifest)


def main() -> None:
    args = parse_args()
    runtime = _build_runtime(args)
    output_dir = Path(runtime["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if runtime["verbose"]:
        print(f"Reading model inputs from: {runtime['input_path']}")

    input_path = Path(runtime["input_path"])
    df = load_model_inputs(input_path, sheet=runtime["sheet"])
    df_raw = pd.read_csv(input_path) if input_path.suffix == ".csv" else pd.read_excel(input_path, sheet_name=runtime["sheet"])
    prepared = prepare_simulation_inputs(df)
    prepared["inputs_df"].to_csv(output_dir / "inputs_used.csv", index=False)
    prepared["profiles_df"].to_csv(output_dir / "profiles_used.csv", index=False)

    summaries = []
    resolved_scenarios = []
    for scenario in runtime["scenarios"]:
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario entries must be mappings, found {type(scenario).__name__}")

        scenario_name = str(scenario["name"])
        scenario_transition_config = (
            scenario["transition_model"] if "transition_model" in scenario else runtime["transition_model_config"]
        )
        if not isinstance(scenario_transition_config, dict) or scenario_transition_config.get("type") != "trend":
            raise ValueError("transition_model.type must be 'trend'")

        base_year = int(scenario_transition_config.get("base_year", 2022))
        if base_year != int(runtime["start_year"]):
            raise ValueError(
                f"transition_model.base_year ({base_year}) must equal "
                f"run.start_year ({int(runtime['start_year'])})"
            )
        trend = str(scenario.get("trend", scenario_transition_config.get("trend", "none")))
        historical_any = _extract_survey_rates_any(df_raw, HIST_SURVEY_YEARS)
        any_base = historical_any[base_year]

        params = eng.SimParams(
            n_props=int(runtime["n_props"]),
            seed=int(runtime["seed"]),
            horizon_years=int(runtime["horizon_years"]),
            disabled_tenure_factor=float(scenario.get("disabled_tenure_factor", 1.0)),
        )
        for uncertainty_case in _uncertainty_cases(prepared["rate_moe_maps"]):
            scenario_rates = _build_scaled_rates(
                prepared["base_rate_maps"],
                scenario,
                rate_moe_maps=prepared["rate_moe_maps"],
                uncertainty_case=uncertainty_case,
            )
            output_scenario_name = f"{scenario_name}_{uncertainty_case}"

            prebuilt = eng.build_trend_schedule(
                scenario_rates,
                any_base,
                trend,
                horizon_years=int(runtime["horizon_years"]),
                historical_any=historical_any,
            )

            print(f"\n=== Running scenario: {_scenario_title(output_scenario_name)} ===")
            summary = eng.run_sim(
                params,
                scenario_rates,
                prepared["tenure"],
                prepared["inmovers"],
                return_time_stats=bool(runtime["return_time_stats"]),
                prebuilt_schedule=prebuilt,
            )
            summaries.append({
                "scenario": output_scenario_name,
                "uncertainty_case": uncertainty_case,
                "trend": trend,
                **summary,
            })
            resolved_scenarios.append(
                {
                    "name": output_scenario_name,
                    "base_scenario": scenario_name,
                    "uncertainty_case": uncertainty_case,
                    "trend": trend,
                    "disabled_tenure_factor": params.disabled_tenure_factor,
                    "rate_scale": float(scenario.get("rate_scale", 1.0)),
                    "rate_scales": scenario.get("rate_scales", {}),
                    "transition_model": {"type": "trend", "trend": trend, "base_year": base_year},
                }
            )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "scenario_summaries.csv", index=False)
    runtime["resolved_scenarios"] = resolved_scenarios
    _write_manifest(runtime, summary_df)

    for scenario in summaries:
        title = f"Scenario: {scenario['scenario']}"
        print(f"\n{title}\n" + "-" * len(title))
        for key, value in scenario.items():
            if key in {"scenario", "uncertainty_case", "trend"}:
                continue
            print(f"{key:26s}: {value:.6f}")

    if runtime["verbose"]:
        print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
