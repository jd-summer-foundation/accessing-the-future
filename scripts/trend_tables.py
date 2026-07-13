#!/usr/bin/env python3
"""Emit the manuscript's Appendix B trend-parameter table.

For each model age bracket the table reports:

- the person-level "any disability" prevalence from the SDAC time-series cube
  for the 2003, 2015 and 2022 surveys (aggregated to the model brackets in
  ``data/processed/model_inputs.csv``);
- the household-level 2022 base rates (any disability; physical) from the
  SDAC 2022 custom extract;
- the relative annual rate of change for each historical trend window,
  computed exactly as the simulation engine does: the absolute annual
  increment over the window divided by the 2022 person-level rate.

Run with ``make trend-tables``. Outputs go to
``reports/tables/appendix_b_trend_parameters.{csv,md}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import au_housing_disability_monte_carlo as eng
from scripts.pipeline_utils import (
    AGE_COL,
    DEFAULT_MODEL_INPUT_CSV,
    DEFAULT_REPORTS_DIR,
    HIST_RATE_ANY_COL_PREFIX,
    RATE_ANY_COL,
    RATE_MOTOR_COL,
    read_tabular_data,
)

TABLE_NAME = "appendix_b_trend_parameters"

# Survey years shown as level columns in Appendix B: the endpoints of the two
# trend windows plus the 2022 base year they share.
LEVEL_YEARS = [2003, 2015, 2022]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Appendix B trend-parameter table.")
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL_INPUT_CSV, help="Processed model inputs CSV.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory where tables will be written.")
    return parser.parse_args()


def _bracket_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.loc[df[AGE_COL].isin(eng.BRACKETS)].copy()
    missing = [bracket for bracket in eng.BRACKETS if bracket not in rows[AGE_COL].tolist()]
    if missing:
        raise ValueError(f"Model inputs missing brackets: {missing}")
    return rows.set_index(AGE_COL).loc[eng.BRACKETS]


def build_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = _bracket_rows(df)

    hist: Dict[int, pd.Series] = {}
    for year in LEVEL_YEARS:
        column = f"{HIST_RATE_ANY_COL_PREFIX}{year}"
        if column not in rows.columns:
            raise ValueError(
                f"Column '{column}' not found in model inputs. "
                "Run 'make build-data' to regenerate model_inputs.csv with historical rates."
            )
        hist[year] = rows[column].astype(float)

    out = pd.DataFrame({"age_bracket": eng.BRACKETS})
    for year in LEVEL_YEARS:
        out[f"any_{year}_pct"] = (hist[year] * 100.0).to_numpy()
    # Household-level 2022 base rates are stored as percentages already.
    out["hh_any_2022_pct"] = rows[RATE_ANY_COL].astype(float).to_numpy()
    out["hh_physical_2022_pct"] = rows[RATE_MOTOR_COL].astype(float).to_numpy()

    # Relative annual change per window, matching the engine's derivation:
    # ((rate_2022 - rate_prior) / window_years) / rate_2022.
    for trend, (prior_year, window_years) in eng._LINEAR_TREND_WINDOWS.items():
        label = f"annual_change_{prior_year}_2022_pct_per_yr"
        increment = (hist[2022] - hist[prior_year]) / float(window_years)
        out[label] = (increment / hist[2022] * 100.0).to_numpy()
    return out


def _write_markdown(table: pd.DataFrame, path: Path) -> None:
    header = [
        "Age bracket",
        "Any 2003 (%)",
        "Any 2015 (%)",
        "Any 2022 (%)",
        "HH any 2022 (%)",
        "HH physical 2022 (%)",
        "Annual change, 2003–2022 (%/yr)",
        "Annual change, 2015–2022 (%/yr)",
    ]
    lines = [
        "# Appendix B. Trend parameters by age bracket",
        "",
        "Person-level columns are SDAC time-series any-disability rates (SDACDC01,",
        "aggregated to the model brackets by population-weighted averaging). Each",
        "window's relative annual change is the absolute annual increment over the",
        "window divided by the 2022 person-level rate; the simulation applies that",
        "relative rate to the household-level 2022 base rates as a constant annual",
        "increment.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for _, row in table.iterrows():
        cells = [str(row["age_bracket"])]
        for column in ["any_2003_pct", "any_2015_pct", "any_2022_pct", "hh_any_2022_pct", "hh_physical_2022_pct"]:
            cells.append(f"{row[column]:.2f}")
        for column in ["annual_change_2003_2022_pct_per_yr", "annual_change_2015_2022_pct_per_yr"]:
            cells.append(f"{row[column]:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = read_tabular_data(args.input)
    table = build_trend_table(df)

    tables_dir = args.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{TABLE_NAME}.csv"
    md_path = tables_dir / f"{TABLE_NAME}.md"
    table.to_csv(csv_path, index=False)
    _write_markdown(table, md_path)
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
