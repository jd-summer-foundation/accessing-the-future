#!/usr/bin/env python3
"""Validate processed model inputs against schema and raw derivation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_model_inputs import build_model_inputs

from scripts.pipeline_utils import (
    DEFAULT_DERIVATION_CONFIG,
    DEFAULT_HOUSING_MOBILITY_WORKBOOK,
    DEFAULT_MODEL_INPUT_CSV,
    DEFAULT_RAW_WORKBOOK,
    compare_model_frames,
    load_model_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed model inputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_MODEL_INPUT_CSV, help="Processed CSV or workbook to validate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_DERIVATION_CONFIG, help="Path to derivation YAML.")
    parser.add_argument("--raw-workbook", type=Path, default=DEFAULT_RAW_WORKBOOK, help="Override SDAC workbook path.")
    parser.add_argument("--mobility-workbook", type=Path, default=DEFAULT_HOUSING_MOBILITY_WORKBOOK, help="Override housing mobility workbook path.")
    parser.add_argument("--sheet", default=0, help="Sheet to read for Excel inputs.")
    parser.add_argument("--skip-raw-check", action="store_true", help="Skip comparison to the raw-derived build output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_df = load_model_inputs(args.input, sheet=args.sheet)

    if not args.skip_raw_check:
        expected_df = build_model_inputs(
            args.config,
            raw_workbook=args.raw_workbook,
            mobility_workbook=args.mobility_workbook,
        )
        diffs = compare_model_frames(input_df, expected_df)
        if diffs:
            raise ValueError("Validated input differs from raw-derived model inputs:\n- " + "\n- ".join(diffs[:10]))

    print(f"Validated model input: {args.input}")
    print(f"Rows: {len(input_df)}")
    print(f"Columns: {len(input_df.columns)}")


if __name__ == "__main__":
    main()
