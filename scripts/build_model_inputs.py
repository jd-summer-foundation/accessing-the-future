#!/usr/bin/env python3
"""Build the canonical processed model input from the raw source workbooks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict

_ABS_FOOTNOTE_RE = re.compile(r"^\([a-z]\)")  # strips leading markers like (f), (i)

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_utils import (
    AGE_COL,
    DEFAULT_DERIVATION_CONFIG,
    DEFAULT_HOUSING_MOBILITY_WORKBOOK,
    DEFAULT_MODEL_INPUT_CSV,
    DEFAULT_RAW_WORKBOOK,
    DEFAULT_SDACDC01_XLSX,
    HIST_RATE_ANY_COL_PREFIX,
    HIST_RATE_COLUMNS,
    HIST_SURVEY_YEARS,
    INMOVER_COL,
    MODEL_INPUT_COLUMNS,
    RATE_COLUMNS,
    RATE_MOE_COLUMNS,
    RATE_MOE_SUFFIX,
    RATE_ANY_COL,
    RATE_MOTOR_COL,
    TENURE_COLUMNS,
    compare_model_frames,
    load_model_inputs,
    load_yaml,
    validate_model_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed model inputs from raw source tables.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DERIVATION_CONFIG, help="Path to derivation YAML.")
    parser.add_argument("--raw-workbook", type=Path, default=None, help="Override raw workbook path.")
    parser.add_argument("--mobility-workbook", type=Path, default=None, help="Override housing mobility workbook path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Override processed CSV output path.")
    parser.add_argument("--output-excel", type=Path, default=None, help="Optional processed Excel output path.")
    parser.add_argument("--oracle", type=Path, default=None, help="Optional comparison workbook for full-frame checking.")
    parser.add_argument("--sdacdc01", type=Path, default=None, help="Override path to SDACDC01.xlsx.")
    return parser.parse_args()


def _load_raw_table(workbook: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(workbook, sheet_name=sheet_name, header=None)


def _row_by_label(table: pd.DataFrame, label: str, label_column_index: int) -> pd.Series:
    labels = table.iloc[:, label_column_index].astype(str).str.strip()
    matches = table.loc[labels == label]
    if matches.empty:
        raise KeyError(f"Could not find raw label {label!r}")
    return matches.iloc[0]


def _extract_value(row: pd.Series, spec: Dict[str, Any]) -> float | int:
    value = row.iloc[int(spec["column_index"])]
    if isinstance(value, str):
        value = value.replace("#", "").replace(",", "").strip()
    numeric = float(value)
    if spec.get("rounding") == "nearest_int":
        return int(np.floor(numeric + 0.5))
    return numeric


def _resolve_configured_path(config: Dict[str, Any], key: str, default: Path) -> Path:
    configured = config.get(key, str(default.relative_to(ROOT)))
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


def _load_housing_mobility_profiles(
    config: Dict[str, Any],
    *,
    mobility_workbook: Path | None = None,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    workbook = mobility_workbook or _resolve_configured_path(config, "housing_mobility_workbook", DEFAULT_HOUSING_MOBILITY_WORKBOOK)
    table_spec = config["housing_mobility_table"]
    age_mapping = table_spec["age_brackets"]
    label_column_index = int(table_spec.get("raw_label_column_index", 0))
    less_than_one_year_column = int(table_spec["less_than_one_year_column_index"])
    estimated_households_column = int(table_spec["estimated_households_column_index"])
    table = _load_raw_table(workbook, table_spec["tenure_sheet"])

    tenure_by_bracket: Dict[str, Dict[str, float]] = {}
    inmover_counts: Dict[str, float] = {}

    for age_bracket, raw_label in age_mapping.items():
        row = _row_by_label(table, raw_label, label_column_index)
        tenure_values: Dict[str, float] = {}
        for column_name, column_spec in table_spec["tenure_columns"].items():
            tenure_values[column_name] = float(_extract_value(row, column_spec))

        lt1_share = float(_extract_value(row, {"column_index": less_than_one_year_column}))
        estimated_households = float(_extract_value(row, {"column_index": estimated_households_column}))
        inmover_counts[age_bracket] = (lt1_share / 100.0) * estimated_households
        tenure_by_bracket[age_bracket] = tenure_values

    total_inmovers = float(sum(inmover_counts.values()))
    if total_inmovers <= 0:
        raise ValueError("Derived in-mover counts sum to zero; check housing mobility configuration.")

    inmover_distribution = {
        age_bracket: inmover_counts[age_bracket] / total_inmovers
        for age_bracket in age_mapping
    }
    return tenure_by_bracket, inmover_distribution


_FINE_TO_MODEL_BRACKET: Dict[str, str] = {
    "15-24": "15-24",
    "25-34": "25-34",
    "35-44": "35-44",
    "45-54": "45-54",
    "55-59": "55-64",
    "60-64": "55-64",
    "65-69": "65-74",
    "70-74": "65-74",
    "75-79": "75+",
    "80-84": "75+",
    "85-89": "75+",
    "90+":   "75+",
}

# Fine-grained age groups shared by Table 1.3 and Table 3.1 in SDACDC01.xlsx.
# Both tables use the same 12 groups in the same row order (Excel rows 44–55 / 43–54
# for Table 1.3 "All persons"; Excel rows 43–54 for Table 3.1 population weights).
_SDACDC01_FINE_GROUPS: list[str] = [
    "15-24", "25-34", "35-44", "45-54",
    "55-59", "60-64", "65-69", "70-74",
    "75-79", "80-84", "85-89", "90+",
]
_SDACDC01_POP_ROW_START = 42   # 0-indexed; Excel rows 43–54 (Table 3.1)
_SDACDC01_POP_COL = 11         # 0-indexed column L

_SDACDC01_TABLE_1_3_SHEET = "Table 1.3 Proportions"
_SDACDC01_HIST_ROW_START = 43  # 0-indexed; Excel rows 44–55 (Table 1.3 "All persons", 15-24→90+)
_SDACDC01_HIST_YEAR_COLS = [1, 2, 3, 4, 5, 6]  # 0-indexed columns for 2003,2009,2012,2015,2018,2022


def _load_sdacdc01_population_weights(xlsx_path: Path) -> Dict[str, float]:
    """
    Read population counts from SDACDC01.xlsx, Table 3.1 Estimates.

    Returns a mapping from fine-grained age group to population count (thousands).
    Used to compute population-weighted averages when combining fine groups into
    model brackets.
    """
    table = pd.read_excel(xlsx_path, sheet_name="Table 3.1 Estimates", header=None)
    pop_values = table.iloc[
        _SDACDC01_POP_ROW_START : _SDACDC01_POP_ROW_START + len(_SDACDC01_FINE_GROUPS),
        _SDACDC01_POP_COL,
    ]
    weights: Dict[str, float] = {}
    for group, raw in zip(_SDACDC01_FINE_GROUPS, pop_values):
        weights[group] = float(raw)
    if any(w <= 0 for w in weights.values()):
        raise ValueError("SDACDC01.xlsx Table 3.1: non-positive population weight encountered")
    return weights


def _load_historical_rates_from_sdacdc01(
    xlsx_path: Path,
    population_weights: Dict[str, float],
) -> Dict[int, Dict[str, float]]:
    """
    Read historical 'any disability' rates from SDACDC01.xlsx Table 1.3 Proportions.

    Reads the "All persons" block (Excel rows 44–55, 0-indexed rows 43–54),
    columns B–G (2003, 2009, 2012, 2015, 2018, 2022). Aggregates 12 fine age
    groups into 7 model brackets via population-weighted averaging. Returns
    fractions (0–1).
    """
    table = pd.read_excel(xlsx_path, sheet_name=_SDACDC01_TABLE_1_3_SHEET, header=None)
    rate_block = table.iloc[
        _SDACDC01_HIST_ROW_START : _SDACDC01_HIST_ROW_START + len(_SDACDC01_FINE_GROUPS),
        _SDACDC01_HIST_YEAR_COLS,
    ]

    from au_housing_disability_monte_carlo import BRACKETS

    result: Dict[int, Dict[str, float]] = {}
    for year_idx, year in enumerate(HIST_SURVEY_YEARS):
        weighted_sums: Dict[str, float] = {b: 0.0 for b in BRACKETS}
        weight_totals: Dict[str, float] = {b: 0.0 for b in BRACKETS}
        for group_idx, fine in enumerate(_SDACDC01_FINE_GROUPS):
            model_bracket = _FINE_TO_MODEL_BRACKET[fine]
            raw = rate_block.iloc[group_idx, year_idx]
            val = float(_ABS_FOOTNOTE_RE.sub("", str(raw)).strip())
            w = population_weights[fine]
            weighted_sums[model_bracket] += val * w
            weight_totals[model_bracket] += w
        result[year] = {b: (weighted_sums[b] / weight_totals[b]) / 100.0 for b in BRACKETS}

    return result


def build_model_inputs(
    config_path: Path,
    *,
    raw_workbook: Path | None = None,
    mobility_workbook: Path | None = None,
    sdacdc01_workbook: Path | None = None,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    workbook = raw_workbook or _resolve_configured_path(config, "raw_workbook", DEFAULT_RAW_WORKBOOK)
    tables = config["raw_tables"]
    label_column_index = int(tables.get("raw_label_column_index", 0))

    estimates = _load_raw_table(workbook, tables["estimates_sheet"])
    proportions = _load_raw_table(workbook, tables["proportions_sheet"])
    proportion_moes_sheet = tables.get("proportion_moes_sheet")
    proportion_moes = _load_raw_table(workbook, proportion_moes_sheet) if proportion_moes_sheet else None
    tenure_by_bracket, inmover_distribution = _load_housing_mobility_profiles(
        config,
        mobility_workbook=mobility_workbook,
    )

    rows = []
    for age_bracket, raw_label in config["age_brackets"].items():
        estimate_row = _row_by_label(estimates, raw_label, label_column_index)
        proportion_row = _row_by_label(proportions, raw_label, label_column_index)
        proportion_moe_row = (
            _row_by_label(proportion_moes, raw_label, label_column_index)
            if proportion_moes is not None
            else None
        )

        row: Dict[str, float | int | str] = {AGE_COL: age_bracket}
        for column_name, spec in config["column_mappings"].items():
            source_row = estimate_row if spec["sheet"] == "estimates" else proportion_row
            row[column_name] = _extract_value(source_row, spec)
            if spec["sheet"] == "proportions" and proportion_moe_row is not None:
                row[f"{column_name}{RATE_MOE_SUFFIX}"] = _extract_value(proportion_moe_row, spec)

        row[INMOVER_COL] = float(inmover_distribution[age_bracket])
        for column in TENURE_COLUMNS:
            row[column] = float(tenure_by_bracket[age_bracket][column])
        rows.append(row)

    optional_moe_columns = [column for column in RATE_MOE_COLUMNS if any(column in row for row in rows)]
    df = pd.DataFrame(rows, columns=[*MODEL_INPUT_COLUMNS, *optional_moe_columns])
    for column in [*RATE_COLUMNS, *optional_moe_columns]:
        df[column] = df[column].astype(float)
    df = validate_model_inputs(df)

    sdacdc01_path = sdacdc01_workbook or DEFAULT_SDACDC01_XLSX
    population_weights = _load_sdacdc01_population_weights(sdacdc01_path)
    hist_rates = _load_historical_rates_from_sdacdc01(sdacdc01_path, population_weights)
    for year in HIST_SURVEY_YEARS:
        col = f"{HIST_RATE_ANY_COL_PREFIX}{year}"
        from au_housing_disability_monte_carlo import BRACKETS
        df[col] = [float(hist_rates[year][b]) for b in BRACKETS]

    return df


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    df = build_model_inputs(
        args.config,
        raw_workbook=args.raw_workbook,
        mobility_workbook=args.mobility_workbook,
        sdacdc01_workbook=args.sdacdc01,
    )

    output_csv = args.output_csv or ROOT / config.get("processed_csv", str(DEFAULT_MODEL_INPUT_CSV.relative_to(ROOT)))
    output_excel = args.output_excel
    oracle = args.oracle

    if oracle is not None and oracle.exists():
        oracle_df = load_model_inputs(oracle)
        diffs = compare_model_frames(df, oracle_df)
        if diffs:
            raise ValueError("Generated model inputs do not match the comparison workbook:\n- " + "\n- ".join(diffs[:10]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if output_excel is not None:
        output_excel.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_excel) as writer:
            df.to_excel(writer, sheet_name="data", index=False)

    print(f"Wrote processed CSV: {output_csv}")
    if output_excel is not None:
        print(f"Wrote processed Excel: {output_excel}")


if __name__ == "__main__":
    main()
