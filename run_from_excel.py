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
    GENPOP_COL,
    INMOVER_COL,
    RATE_ANY_COL,
    RATE_MOTOR_COL,
    RATE_PHYS2_COL,
    RATE_SEVERE_COL,
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
    parser.add_argument("--excel", type=Path, default=None, help="Deprecated alias for --input.")
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
        {"name": "main_inmover_first", "first_draw_source": "inmover", "disabled_tenure_factor": 1.0, "rate_scale": 1.0},
        {"name": "sens_general_first", "first_draw_source": "general", "disabled_tenure_factor": 1.0, "rate_scale": 1.0},
        {"name": "sens_disabled_tenure_1p2", "first_draw_source": "inmover", "disabled_tenure_factor": 1.2, "rate_scale": 1.0},
        {"name": "selection_scale_0p9", "first_draw_source": "inmover", "disabled_tenure_factor": 1.0, "rate_scale": 0.9},
    ]


def _scenario_title(name: str) -> str:
    return name.replace("_", " ")


def _build_runtime(args: argparse.Namespace) -> Dict[str, object]:
    config = load_yaml(args.config) if args.config and args.config.exists() else {}
    run_cfg = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}
    input_cfg = config.get("input", {}) if isinstance(config.get("input", {}), dict) else {}

    configured_input = _resolve_path(run_cfg.get("input_path")) if run_cfg else None
    input_path = resolve_input_path(args.input or args.excel or configured_input)
    output_dir = args.outdir or _resolve_path(run_cfg.get("output_dir")) or SCRIPT_DIR / "results/baseline"

    runtime = {
        "config": config,
        "config_path": args.config,
        "input_path": input_path,
        "sheet": args.sheet if args.sheet is not None else input_cfg.get("sheet", 0),
        "output_dir": output_dir,
        "run_name": run_cfg.get("name", output_dir.name),
        "n_props": args.n_props if args.n_props is not None else int(run_cfg.get("n_props", 44_346)),
        "seed": args.seed if args.seed is not None else int(run_cfg.get("seed", 123)),
        "horizon_years": args.horizon_years if args.horizon_years is not None else int(run_cfg.get("horizon_years", 50)),
        "return_time_stats": bool(run_cfg.get("return_time_stats", True)),
        "scenarios": config.get("scenarios") or _default_scenarios(),
        "transition_model_config": config.get("transition_model"),
        "verbose": args.verbose,
        "cli_args": {key: value for key, value in vars(args).items() if value is not None and key != "verbose"},
    }
    return runtime


def prepare_simulation_inputs(df: pd.DataFrame) -> Dict[str, object]:
    valid_rows = df.loc[df[AGE_COL].isin(eng.BRACKETS)].copy()

    inmover_raw = {bracket: float(valid_rows.loc[valid_rows[AGE_COL] == bracket, INMOVER_COL].iloc[0]) for bracket in eng.BRACKETS}
    inmover_probs = _normalise_dist(inmover_raw, "inmover_probs")
    inmovers = eng.InMoverDist(inmover_probs)

    gen_raw: Dict[str, float] = {}
    for bracket in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[AGE_COL] == bracket]
        value = pd.to_numeric(row.iloc[0][GENPOP_COL], errors="coerce") if not row.empty else float("nan")
        gen_raw[bracket] = 0.0 if pd.isna(value) else float(value)
    gen_probs = _normalise_dist(gen_raw, "general_pop_probs")

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
    fallback_severe = {"15-24": 0.02, "25-34": 0.02, "35-44": 0.03, "45-54": 0.05, "55-64": 0.07, "65-74": 0.11, "75+": 0.20}
    fallback_motor = {"15-24": 0.01, "25-34": 0.015, "35-44": 0.02, "45-54": 0.04, "55-64": 0.06, "65-74": 0.10, "75+": 0.18}

    rates_any = _extract_bracketed_series(valid_rows, AGE_COL, RATE_ANY_COL, fallback_any)
    rates_severe = _extract_bracketed_series(valid_rows, AGE_COL, RATE_SEVERE_COL, fallback_severe)
    rates_motor = _extract_bracketed_series(valid_rows, AGE_COL, RATE_MOTOR_COL, fallback_motor)
    rates_phys2 = _extract_bracketed_series(valid_rows, AGE_COL, RATE_PHYS2_COL, fallback_motor)

    for bracket in eng.BRACKETS:
        rates_severe[bracket] = min(float(rates_severe[bracket]), float(rates_any[bracket]))
        rates_motor[bracket] = min(float(rates_motor[bracket]), float(rates_any[bracket]))

    base_rate_maps = {
        RATE_ANY_COL: rates_any,
        RATE_SEVERE_COL: rates_severe,
        RATE_MOTOR_COL: rates_motor,
        RATE_PHYS2_COL: rates_phys2,
    }
    rates = eng.AllRates(
        any_dis=eng.Rates(rates_any),
        severe_prof=eng.Rates(rates_severe),
        motor_phys=eng.Rates(rates_motor),
        phys2=eng.Rates(rates_phys2),
    )

    inputs_df = pd.DataFrame(
        {
            "age_bracket": eng.BRACKETS,
            "any_rate_input": [rates_any[bracket] for bracket in eng.BRACKETS],
            "severe_rate_input": [rates_severe[bracket] for bracket in eng.BRACKETS],
            "motor_phys_rate_input": [rates_motor[bracket] for bracket in eng.BRACKETS],
            "phys2_rate_input": [rates_phys2[bracket] for bracket in eng.BRACKETS],
            "inmover_dist": [inmover_probs[bracket] for bracket in eng.BRACKETS],
            "general_pop_dist": [gen_probs[bracket] for bracket in eng.BRACKETS],
        }
    )
    for index, bucket in enumerate(TENURE_BUCKETS):
        inputs_df[f"tenure_prob_{bucket}"] = [tenure.probs_by_bracket[bracket][index] for bracket in eng.BRACKETS]

    profiles = eng.make_profiles(rates)
    profiles_df = pd.DataFrame(
        {
            "age_bracket": eng.BRACKETS,
            "adj_any": [profiles["adj_any"][bracket] for bracket in eng.BRACKETS],
            "cond_severe_given_any": [profiles["cond_severe"][bracket] for bracket in eng.BRACKETS],
            "cond_motor_phys_given_any": [profiles["cond_phys"][bracket] for bracket in eng.BRACKETS],
            "adj_phys2": [profiles["adj_phys2"][bracket] for bracket in eng.BRACKETS],
        }
    )

    return {
        "rates": rates,
        "base_rate_maps": base_rate_maps,
        "tenure": tenure,
        "inmovers": inmovers,
        "general_pop_probs": gen_probs,
        "inputs_df": inputs_df,
        "profiles_df": profiles_df,
    }


def _build_scaled_rates(base_rate_maps: Dict[str, Dict[str, float]], scenario: Dict[str, object]) -> eng.AllRates:
    global_scale = float(scenario.get("rate_scale", 1.0))
    category_scales = scenario.get("rate_scales", {})
    if not isinstance(category_scales, dict):
        category_scales = {}

    def scaled(column_name: str) -> Dict[str, float]:
        scale = float(category_scales.get(column_name, global_scale))
        return {bracket: scale * float(base_rate_maps[column_name][bracket]) for bracket in eng.BRACKETS}

    return eng.AllRates(
        any_dis=eng.Rates(scaled(RATE_ANY_COL)),
        severe_prof=eng.Rates(scaled(RATE_SEVERE_COL)),
        motor_phys=eng.Rates(scaled(RATE_MOTOR_COL)),
        phys2=eng.Rates(scaled(RATE_PHYS2_COL)),
    )


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

    df = load_model_inputs(Path(runtime["input_path"]), sheet=runtime["sheet"])
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
        scenario_transition_model = eng.transition_model_from_config(scenario_transition_config)
        params = eng.SimParams(
            n_props=int(runtime["n_props"]),
            seed=int(runtime["seed"]),
            first_draw_source=str(scenario.get("first_draw_source", "inmover")),
            horizon_years=int(runtime["horizon_years"]),
            disabled_tenure_factor=float(scenario.get("disabled_tenure_factor", 1.0)),
        )
        scenario_rates = _build_scaled_rates(prepared["base_rate_maps"], scenario)

        print(f"\n=== Running scenario: {_scenario_title(scenario_name)} ===")
        summary = eng.run_sim(
            params,
            scenario_rates,
            prepared["tenure"],
            prepared["inmovers"],
            transition_model=scenario_transition_model,
            general_pop_probs=prepared["general_pop_probs"] if params.first_draw_source == "general" else None,
            return_time_stats=bool(runtime["return_time_stats"]),
        )
        summaries.append({"scenario": scenario_name, **summary})
        resolved_scenarios.append(
            {
                "name": scenario_name,
                "first_draw_source": params.first_draw_source,
                "disabled_tenure_factor": params.disabled_tenure_factor,
                "rate_scale": float(scenario.get("rate_scale", 1.0)),
                "rate_scales": scenario.get("rate_scales", {}),
                "transition_model": eng.transition_model_to_config(scenario_transition_model),
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
            if key == "scenario":
                continue
            print(f"{key:26s}: {value:.6f}")

    if runtime["verbose"]:
        print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
