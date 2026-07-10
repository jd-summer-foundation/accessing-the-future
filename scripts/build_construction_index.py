#!/usr/bin/env python3
"""Build the processed house-construction cost index from the raw ABS PPI workbook.

Source: ABS 6427.0 Producer Price Indexes, Australia, Table 17
("Output of the Construction industries, subdivision and class index numbers"),
class 3011 "House construction" for New South Wales and Western Australia.
Series are quarterly; the retrofit cost analysis works in yearly steps, so we
keep the March quarter of each year as that year's index level.
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

from scripts.pipeline_utils import (
    CONSTRUCTION_INDEX_COLUMNS,
    DEFAULT_CONSTRUCTION_INDEX_CSV,
    DEFAULT_PPI_WORKBOOK,
    PPI_SERIES_IDS,
)

DATA_SHEET = "Data1"
SERIES_ID_LABEL = "Series ID"
MARCH = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed construction cost index from ABS PPI Table 17.")
    parser.add_argument("--ppi-workbook", type=Path, default=DEFAULT_PPI_WORKBOOK, help="Path to raw ABS 6427017.xlsx.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CONSTRUCTION_INDEX_CSV, help="Processed CSV output path.")
    return parser.parse_args()


def _series_columns(raw: pd.DataFrame) -> Dict[str, int]:
    """Locate the workbook columns for each state by ABS Series ID (robust to column order)."""
    id_rows = raw.index[raw.iloc[:, 0] == SERIES_ID_LABEL].tolist()
    if len(id_rows) != 1:
        raise ValueError(f"Expected exactly one '{SERIES_ID_LABEL}' row in {DATA_SHEET}, found {len(id_rows)}")
    id_row = raw.iloc[id_rows[0]]
    columns: Dict[str, int] = {}
    for state, series_id in PPI_SERIES_IDS.items():
        matches = [index for index, value in enumerate(id_row) if value == series_id]
        if len(matches) != 1:
            raise ValueError(f"Series ID {series_id} ({state}) matched {len(matches)} columns; expected 1")
        columns[state] = matches[0]
    return columns


def build_construction_index(ppi_workbook: Path) -> pd.DataFrame:
    raw = pd.read_excel(ppi_workbook, sheet_name=DATA_SHEET, header=None)
    columns = _series_columns(raw)

    # The observation column mixes metadata labels with datetime cells; keep the datetimes.
    dates = pd.to_datetime(raw.iloc[:, 0].where(raw.iloc[:, 0].map(lambda v: isinstance(v, pd.Timestamp) or hasattr(v, "month"))), errors="coerce")
    march_rows = raw.loc[dates.notna() & (dates.dt.month == MARCH)]
    march_dates = dates.loc[march_rows.index]

    out = pd.DataFrame(
        {
            "year": march_dates.dt.year.astype(int).to_numpy(),
            "nsw_index": pd.to_numeric(march_rows.iloc[:, columns["nsw"]], errors="coerce").to_numpy(),
            "wa_index": pd.to_numeric(march_rows.iloc[:, columns["wa"]], errors="coerce").to_numpy(),
        }
    )
    # Both state series start later than the national one; keep only years with both observed.
    out = out.dropna().reset_index(drop=True)

    if out["year"].duplicated().any():
        raise ValueError("Duplicate years in the March-quarter index series")
    if not out["year"].is_monotonic_increasing:
        raise ValueError("March-quarter index years are not sorted")
    if (out[["nsw_index", "wa_index"]] <= 0).any().any():
        raise ValueError("Index values must be positive")

    return out[CONSTRUCTION_INDEX_COLUMNS]


def main() -> None:
    args = parse_args()
    df = build_construction_index(args.ppi_workbook)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows ({int(df['year'].min())}-{int(df['year'].max())}) to {args.output_csv}")


if __name__ == "__main__":
    main()
