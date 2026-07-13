#!/usr/bin/env python3
"""Build the new-dwelling structure-type mix from the raw ABS Building Activity cube.

Source: ABS 8752.0 Building Activity, Australia, building-activity data cube
(87520_activity.xlsb). The cube is a long/tidy table (one row per Type of
Building x Type of Work x Reference Quarter x Sector of Ownership x State).

For NSW and WA, the script keeps new-dwelling commencements ("Dwelling units
commenced") for the three structure types that map onto the CIE DRIS cost
archetypes — separate houses, townhouses and apartments — sums them over the
most recent 12 quarters (3 years, to smooth apartment-project lumpiness), and
converts them to shares of the three-type total.

Correctness note: "Sector of Ownership" has Private Sector, Public Sector AND
Total Sectors rows; Total Sectors already equals Private + Public, so only
Total Sectors rows are kept (summing all sector rows would double-count every
dwelling).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_utils import (
    BUILDING_TYPE_MAP,
    DEFAULT_BUILDING_ACTIVITY_WORKBOOK,
    DEFAULT_DWELLING_MIX_CSV,
    DWELLING_MIX_COLUMNS,
    DWELLING_MIX_STATES,
    DWELLING_TYPES,
)

DATA_SHEET = "87520_activity"
TYPE_OF_WORK = "New"
SECTOR = "Total Sectors"
MEASURE = "Dwelling units commenced (Number)"
WINDOW_QUARTERS = 12
# Excel serial date epoch used by the .xlsb cube.
EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the NSW/WA new-dwelling structure-type mix from ABS 8752.0.")
    parser.add_argument("--activity-workbook", type=Path, default=DEFAULT_BUILDING_ACTIVITY_WORKBOOK, help="Path to raw ABS 87520_activity.xlsb.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_DWELLING_MIX_CSV, help="Processed CSV output path.")
    parser.add_argument("--window-quarters", type=int, default=WINDOW_QUARTERS, help="Number of most recent quarters to average over.")
    return parser.parse_args()


def build_dwelling_mix(activity_workbook: Path, *, window_quarters: int = WINDOW_QUARTERS) -> pd.DataFrame:
    raw = pd.read_excel(
        activity_workbook,
        sheet_name=DATA_SHEET,
        engine="pyxlsb",
        usecols=["Type of Building", "Type of Work", "Reference Quarter", "Sector of Ownership", "State/Territory", MEASURE],
    )

    sectors = set(raw["Sector of Ownership"].dropna().unique())
    if SECTOR not in sectors:
        raise ValueError(f"Sector value {SECTOR!r} not found in cube (found {sorted(sectors)})")

    kept = raw.loc[
        (raw["Type of Work"] == TYPE_OF_WORK)
        & (raw["Sector of Ownership"] == SECTOR)
        & raw["State/Territory"].isin(DWELLING_MIX_STATES.values())
        & raw["Type of Building"].isin(BUILDING_TYPE_MAP)
    ].copy()
    if kept.empty:
        raise ValueError("No rows survived the filters; the cube layout may have changed")

    # Reference Quarter arrives as an Excel date serial; footnote rows are non-numeric.
    quarter_serial = pd.to_numeric(kept["Reference Quarter"], errors="coerce")
    kept = kept.loc[quarter_serial.notna()]
    kept["quarter"] = EXCEL_EPOCH + pd.to_timedelta(quarter_serial.loc[kept.index].astype(int), unit="D")
    kept["dwelling_type"] = kept["Type of Building"].map(BUILDING_TYPE_MAP)
    kept["state"] = kept["State/Territory"].map({label: state for state, label in DWELLING_MIX_STATES.items()})
    kept["commenced"] = pd.to_numeric(kept[MEASURE], errors="coerce").fillna(0.0)

    quarters = sorted(kept["quarter"].unique())
    if len(quarters) < window_quarters:
        raise ValueError(f"Cube has only {len(quarters)} quarters; {window_quarters} requested")
    window = quarters[-window_quarters:]
    windowed = kept.loc[kept["quarter"].isin(window)]

    # Every state/type must be observed in every window quarter, or the shares
    # would silently mix windows of different lengths.
    counts = windowed.groupby(["state", "dwelling_type"])["quarter"].nunique()
    if (counts != window_quarters).any():
        raise ValueError(f"Incomplete quarter coverage in window: {counts[counts != window_quarters].to_dict()}")

    totals = (
        windowed.groupby(["state", "dwelling_type"], as_index=False)["commenced"].sum()
    )
    totals["share"] = totals["commenced"] / totals.groupby("state")["commenced"].transform("sum")
    totals["window_start_quarter"] = pd.Timestamp(window[0]).strftime("%Y-%m")
    totals["window_end_quarter"] = pd.Timestamp(window[-1]).strftime("%Y-%m")

    type_order = {dwelling_type: rank for rank, dwelling_type in enumerate(DWELLING_TYPES)}
    state_order = {state: rank for rank, state in enumerate(DWELLING_MIX_STATES)}
    totals = totals.sort_values(
        ["state", "dwelling_type"],
        key=lambda col: col.map(state_order if col.name == "state" else type_order),
    ).reset_index(drop=True)
    totals["commenced"] = totals["commenced"].round().astype(int)

    share_sums = totals.groupby("state")["share"].sum()
    if not ((share_sums - 1.0).abs() < 1e-9).all():
        raise ValueError(f"Shares do not sum to 1 per state: {share_sums.to_dict()}")

    return totals[DWELLING_MIX_COLUMNS]


def main() -> None:
    args = parse_args()
    df = build_dwelling_mix(args.activity_workbook, window_quarters=args.window_quarters)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    window = f"{df['window_start_quarter'].iloc[0]}..{df['window_end_quarter'].iloc[0]}"
    print(f"Wrote {len(df)} rows ({window}) to {args.output_csv}")
    for state, group in df.groupby("state"):
        shares = ", ".join(f"{row.dwelling_type} {row.share:.1%}" for row in group.itertuples())
        print(f"  {state}: {shares} (commenced {group['commenced'].sum():,})")


if __name__ == "__main__":
    main()
