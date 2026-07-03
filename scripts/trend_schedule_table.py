#!/usr/bin/env python3
"""Export the projected disability-rate schedules as appendix-ready tables.

For each trend treatment ("none", "sdac_2003_2022_trend", "sdac_2015_2022_trend")
this script writes out the age-bracket disability rates the simulation applies in
every simulation year, plus the per-bracket trend parameters derived from the
historical SDAC series. The schedules are built with the same functions the model
run uses (prepare_simulation_inputs + build_trend_schedule), so the tables show
exactly the rates that enter the Monte Carlo.

Outputs (under <reports-dir>/tables/):
  - table_a1_trend_parameters.csv / .md   per-bracket trend derivation
  - table_a2_trend_schedules.csv          long-format schedule (machine-readable)
  - table_a2_trend_schedules.md           per-trend, per-category tables to paste
                                          into the manuscript appendix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import au_housing_disability_monte_carlo as eng
from run_from_excel import _extract_survey_rates_any, prepare_simulation_inputs
from scripts.pipeline_utils import (
    DEFAULT_BASELINE_CONFIG,
    DEFAULT_REPORTS_DIR,
    HIST_SURVEY_YEARS,
    artifact_checksums,
    load_model_inputs,
    load_yaml,
    manifest_path,
    read_tabular_data,
    resolve_input_path,
    sha256_file,
    utc_now_iso,
    write_json,
)

PARAMS_NAME = "table_a1_trend_parameters"
SCHEDULE_NAME = "table_a2_trend_schedules"

TREND_LABELS = {
    "none": "No trend (rates fixed at 2022 levels)",
    "sdac_2003_2022_trend": "Trend 2003–2022",
    "sdac_2015_2022_trend": "Trend 2015–2022",
}
CATEGORY_LABELS = {
    "any_disability": "Any disability",
    "physical_disability": "Physical disability",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_BASELINE_CONFIG, help="Run configuration YAML (for horizon and start year).")
    parser.add_argument("--input", type=Path, default=None, help="Processed model inputs CSV (defaults to the config/canonical path).")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory where tables will be written.")
    return parser.parse_args()


def _trend_parameter_rows(
    historical_any: Dict[int, Dict[str, float]],
    rates_2022: eng.AllRates,
    start_year: int,
) -> pd.DataFrame:
    """Replicate the trend derivation per bracket so it can be reported.

    Mirrors _build_linear_trend_schedule: the absolute annual increment in the
    person-level any-disability series is converted to a relative annual rate of
    change, which is then applied to the household-level 2022 base rates.
    """
    any_base = historical_any[start_year]
    rows: List[Dict[str, object]] = []
    for trend, (prior_year, window_years) in eng._LINEAR_TREND_WINDOWS.items():
        any_prior = historical_any[prior_year]
        for bracket in eng.BRACKETS:
            person_prior = float(any_prior[bracket])
            person_2022 = float(any_base[bracket])
            increment = (person_2022 - person_prior) / window_years
            relative = increment / person_2022 if person_2022 > 0.0 else 0.0
            hh_any = float(rates_2022.any_dis.by_bracket[bracket])
            hh_phys = float(rates_2022.motor_phys.by_bracket[bracket])
            rows.append(
                {
                    "trend": trend,
                    "age_bracket": bracket,
                    "window_start_year": prior_year,
                    "window_years": window_years,
                    "person_any_rate_window_start": person_prior,
                    f"person_any_rate_{start_year}": person_2022,
                    "annual_increment_person_pp": increment,
                    "relative_annual_change": relative,
                    f"household_any_rate_{start_year}": hh_any,
                    f"household_physical_rate_{start_year}": hh_phys,
                    "annual_increment_household_any_pp": hh_any * relative,
                    "annual_increment_household_physical_pp": hh_phys * relative,
                }
            )
    return pd.DataFrame(rows)


def build_schedule_table(
    rates_2022: eng.AllRates,
    historical_any: Dict[int, Dict[str, float]],
    horizon_years: int,
    start_year: int,
) -> pd.DataFrame:
    """Long-format table of applied rates: one row per trend, year and bracket."""
    any_base = historical_any[start_year]
    rows: List[Dict[str, object]] = []
    for trend in eng.TREND_TYPES:
        schedule = eng.build_trend_schedule(
            rates_2022,
            any_base,
            trend,
            horizon_years=horizon_years,
            historical_any=historical_any,
        )
        for snapshot in schedule:
            offset = int(snapshot.time_year)
            for bracket in eng.BRACKETS:
                rows.append(
                    {
                        "trend": trend,
                        "year_offset": offset,
                        "calendar_year": start_year + offset,
                        "age_bracket": bracket,
                        "any_disability_rate": snapshot.rates.any_dis.by_bracket[bracket],
                        "physical_disability_rate": snapshot.rates.motor_phys.by_bracket[bracket],
                    }
                )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in df.columns) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def _schedule_markdown(schedule_df: pd.DataFrame, start_year: int) -> str:
    """Render the schedules as copy-paste tables: rows = years, columns = brackets."""
    sections: List[str] = [
        "# Appendix: projected disability rates by trend scenario",
        "",
        "Simulated household-level disability prevalence (%) by age of household",
        f"reference person, for each simulation year, as applied by the model. Rates in",
        f"year 0 ({start_year}) are the SDAC {start_year} household-level base rates; later years",
        "apply the per-bracket relative annual trend derived from the SDAC person-level",
        "any-disability series (see the trend-parameters table). Physical-disability",
        "rates are capped at the any-disability rate in the same bracket and year.",
    ]
    rate_columns = {
        "any_disability": "any_disability_rate",
        "physical_disability": "physical_disability_rate",
    }
    for trend in eng.TREND_TYPES:
        subset = schedule_df.loc[schedule_df["trend"] == trend]
        sections += ["", f"## {TREND_LABELS[trend]}"]
        if trend == "none":
            sections += [
                "",
                "Rates are held fixed at their base-year values for the whole horizon,",
                "so every simulation year applies the year-0 row shown below.",
            ]
        for category, column in rate_columns.items():
            wide = subset.pivot(index="calendar_year", columns="age_bracket", values=column)
            wide = wide.loc[:, eng.BRACKETS]
            if trend == "none":
                wide = wide.iloc[:1]
            display = wide.map(lambda value: f"{value * 100:.1f}").reset_index()
            display = display.rename(columns={"calendar_year": "Year"})
            sections += ["", f"### {CATEGORY_LABELS[category]} (%)", "", _markdown_table(display)]
    return "\n".join(sections) + "\n"


def _parameters_markdown(params_df: pd.DataFrame) -> str:
    sections: List[str] = [
        "# Appendix: trend parameters by age bracket",
        "",
        "Derivation of the annual rate changes applied under each historical-trend",
        "window. Person-level columns are the SDAC time-series any-disability rates",
        "used to estimate the trend; the relative annual change is applied to the",
        "household-level 2022 base rates (any and physical) each simulation year.",
        "All rate and increment columns are percentages (increments in percentage",
        "points per year).",
    ]
    percent_columns = [
        column for column in params_df.columns
        if column not in {"trend", "age_bracket", "window_start_year", "window_years", "relative_annual_change"}
    ]
    for trend, group in params_df.groupby("trend", sort=False):
        display = group.drop(columns="trend").copy()
        for column in percent_columns:
            decimals = 3 if column.startswith("annual_increment") else 1
            display[column] = display[column].map(lambda value: f"{value * 100:+.3f}" if decimals == 3 else f"{value * 100:.1f}")
        display["relative_annual_change"] = group["relative_annual_change"].map(lambda value: f"{value * 100:+.2f}%")
        sections += ["", f"## {TREND_LABELS[str(trend)]}", "", _markdown_table(display)]
    return "\n".join(sections) + "\n"


def generate_trend_tables(
    config_path: Path,
    input_override: Path | None,
    reports_dir: Path,
) -> Dict[str, object]:
    config = load_yaml(config_path) if config_path.exists() else {}
    run_cfg = config.get("run", {}) if isinstance(config.get("run", {}), dict) else {}

    configured_input = run_cfg.get("input_path")
    configured_path = None
    if configured_input is not None:
        configured_path = Path(configured_input)
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
    input_path = resolve_input_path(input_override or configured_path)
    horizon_years = int(run_cfg.get("horizon_years", eng.DEFAULT_HORIZON_YEARS))
    start_year = int(run_cfg.get("start_year", 2022))
    sheet = config.get("input", {}).get("sheet", 0) if isinstance(config.get("input", {}), dict) else 0

    df = load_model_inputs(input_path, sheet=sheet)
    df_raw = read_tabular_data(input_path, sheet=sheet)
    prepared = prepare_simulation_inputs(df)
    historical_any = _extract_survey_rates_any(df_raw, HIST_SURVEY_YEARS)
    rates_2022 = prepared["rates"]

    schedule_df = build_schedule_table(rates_2022, historical_any, horizon_years, start_year)
    params_df = _trend_parameter_rows(historical_any, rates_2022, start_year)

    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    schedule_csv = tables_dir / f"{SCHEDULE_NAME}.csv"
    schedule_md = tables_dir / f"{SCHEDULE_NAME}.md"
    params_csv = tables_dir / f"{PARAMS_NAME}.csv"
    params_md = tables_dir / f"{PARAMS_NAME}.md"

    schedule_df.to_csv(schedule_csv, index=False)
    schedule_md.write_text(_schedule_markdown(schedule_df, start_year), encoding="utf-8")
    params_df.to_csv(params_csv, index=False)
    params_md.write_text(_parameters_markdown(params_df), encoding="utf-8")

    generated_paths = [params_csv, params_md, schedule_csv, schedule_md]
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "config": {
            "path": config_path,
            "sha256": sha256_file(config_path) if config_path.exists() else None,
        },
        "input": {
            "path": input_path,
            "sha256": sha256_file(input_path),
        },
        "horizon_years": horizon_years,
        "start_year": start_year,
        "trends": list(eng.TREND_TYPES),
        "artifacts": [manifest_path(path, relative_to=reports_dir) for path in generated_paths],
        "artifact_checksums": artifact_checksums(generated_paths, relative_to=reports_dir),
    }
    write_json(reports_dir / "trend_tables_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate_trend_tables(args.config, args.input, args.reports_dir)
    print(f"Generated {len(manifest['artifacts'])} trend-table artifacts in {args.reports_dir}")


if __name__ == "__main__":
    main()
