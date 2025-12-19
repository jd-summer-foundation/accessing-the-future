#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Excel-driven runner for the housing disability Monte Carlo model.

Defaults assume this repo layout:

./inputs/Data for modelling.xlsx
./outputs/   (created if missing)

You can override paths via CLI flags (recommended for reproducibility / batch runs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure imports work even when running from another directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import au_housing_disability_monte_carlo as eng


# -----------------------------
# Helpers
# -----------------------------

def _find_column(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        raise KeyError(f"Missing expected column: {name!r}. Columns found: {df.columns.tolist()}")
    return name


def _to_rate(x) -> float:
    """Interpret Excel values as either proportions (0-1) or percentages (0-100)."""
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return float("nan")
    v = float(v)
    if v > 1.0:
        v = v / 100.0
    return float(v)


def _normalise_dist(d: Dict[str, float], name: str) -> Dict[str, float]:
    vals = np.array([float(d[b]) for b in eng.BRACKETS], dtype=float)
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= 0:
        raise ValueError(f"{name}: distribution sums to 0 after cleaning: {d}")
    vals = vals / s
    return {b: float(vals[i]) for i, b in enumerate(eng.BRACKETS)}


def _extract_bracketed_series(
    valid_rows: pd.DataFrame,
    age_col: str,
    series_name: str,
    fallback: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[age_col] == b]
        if row.empty or series_name not in valid_rows.columns:
            out[b] = float(fallback[b])
            continue
        val = _to_rate(row.iloc[0][series_name])
        out[b] = float(fallback[b]) if (not np.isfinite(val)) else float(val)
    return out


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Australian housing disability Monte Carlo from Excel inputs.")
    ap.add_argument(
        "--excel",
        type=Path,
        default=Path("inputs/Data for modelling.xlsx"),
        help="Path to input Excel workbook.",
    )
    ap.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index (default: 0).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Output directory (will be created if missing).",
    )
    ap.add_argument("--n-props", type=int, default=376_000, help="Number of dwellings to simulate.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument("--horizon-years", type=int, default=50, help="Simulation horizon in years.")
    ap.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Load Excel ---
    if args.verbose:
        print(f"Reading Excel: {args.excel}")
    df = pd.read_excel(args.excel, sheet_name=args.sheet)

    # Column names (paper-linked data)
    age_col = "Age of Reference Person (Years)"
    inmover_col = "Of those who have length of tenure <1 year, what proportion are in each age bucket?"
    genpop_col = "376k households"

    tenure_cols = [
        "Length of tenure: less than 1 year  (% of all households, not count)",
        "Length of tenure: 1-2 years",
        "Length of tenure: 2-3 years",
        "Length of tenure: 3-4 years",
        "Length of tenure: 5-9 years",
        "Length of tenure: 10-19 years",
        "Length of tenure: 20+ years",
    ]
    buckets = ["<1", "1-2", "2-3", "3-4", "5-9", "10-19", "20+"]

    rate_any_col = "DIP_any"
    rate_motor_col = "DIP_physical"
    rate_severe_col = "DIP_severe"
    rate_phys2_col = "DIP_physical2"

    # Validate columns exist
    _find_column(df, age_col)
    _find_column(df, inmover_col)
    _find_column(df, genpop_col)
    for c in tenure_cols + [rate_any_col, rate_motor_col, rate_severe_col, rate_phys2_col]:
        _find_column(df, c)

    # Filter to the 7 age brackets and only use these rows from now on
    valid_rows = df.loc[df[age_col].isin(eng.BRACKETS)].copy()
    if args.verbose:
        print(f"Bracket rows found: {len(valid_rows)} (expected {len(eng.BRACKETS)})")

    # --- In-mover distribution ---
    inmover_raw: Dict[str, float] = {}
    for b in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[age_col] == b]
        val = _to_rate(row.iloc[0][inmover_col]) if not row.empty else float("nan")
        inmover_raw[b] = 0.0 if not np.isfinite(val) else float(val)
    inmover_probs = _normalise_dist(inmover_raw, "inmover_probs")
    inmovers = eng.InMoverDist(inmover_probs)

    # --- General population distribution ---
    gen_raw: Dict[str, float] = {}
    for b in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[age_col] == b]
        val = pd.to_numeric(row.iloc[0][genpop_col], errors="coerce") if not row.empty else float("nan")
        gen_raw[b] = 0.0 if pd.isna(val) else float(val)
    gen_probs = _normalise_dist(gen_raw, "general_pop_probs")

    # --- Tenure distribution per bracket ---
    probs_by_bracket: Dict[str, List[float]] = {}
    for b in eng.BRACKETS:
        row = valid_rows.loc[valid_rows[age_col] == b]
        if row.empty:
            raise ValueError(f"Missing tenure row for bracket {b}")
        vals: List[float] = []
        for c in tenure_cols:
            v = _to_rate(row.iloc[0][c])
            vals.append(0.0 if not np.isfinite(v) else float(v))
        arr = np.clip(np.asarray(vals, dtype=float), 0.0, None)
        s = float(arr.sum())
        if s <= 0:
            raise ValueError(f"Tenure probs sum to 0 for {b}: {vals}")
        probs_by_bracket[b] = (arr / s).tolist()

    tenure = eng.TenureDist(buckets=buckets, probs_by_bracket=probs_by_bracket)

    # --- Rates (fallbacks only used if Excel has NaNs) ---
    dummy_any = {"15-24": 0.06, "25-34": 0.08, "35-44": 0.11, "45-54": 0.15, "55-64": 0.21, "65-74": 0.30, "75+": 0.48}
    dummy_sev = {"15-24": 0.02, "25-34": 0.02, "35-44": 0.03, "45-54": 0.05, "55-64": 0.07, "65-74": 0.11, "75+": 0.20}
    dummy_mot = {"15-24": 0.01, "25-34": 0.015, "35-44": 0.02, "45-54": 0.04, "55-64": 0.06, "65-74": 0.10, "75+": 0.18}

    rates_any = _extract_bracketed_series(valid_rows, age_col, rate_any_col, dummy_any)
    rates_sev = _extract_bracketed_series(valid_rows, age_col, rate_severe_col, dummy_sev)
    rates_mot = _extract_bracketed_series(valid_rows, age_col, rate_motor_col, dummy_mot)
    rates_phys2 = _extract_bracketed_series(valid_rows, age_col, rate_phys2_col, dummy_mot)

    # Enforce nesting sanity for subtype totals: severe <= any, motor <= any
    for b in eng.BRACKETS:
        rates_sev[b] = min(float(rates_sev[b]), float(rates_any[b]))
        rates_mot[b] = min(float(rates_mot[b]), float(rates_any[b]))

    rates = eng.AllRates(
        any_dis=eng.Rates(rates_any),
        severe_prof=eng.Rates(rates_sev),
        motor_phys=eng.Rates(rates_mot),
        phys2=eng.Rates(rates_phys2),
    )

    # --- Save inputs used ---
    inputs_df = pd.DataFrame({
        "age_bracket": eng.BRACKETS,
        "any_rate_input": [rates_any[b] for b in eng.BRACKETS],
        "severe_rate_input": [rates_sev[b] for b in eng.BRACKETS],
        "motor_phys_rate_input": [rates_mot[b] for b in eng.BRACKETS],
        "phys2_rate_input": [rates_phys2[b] for b in eng.BRACKETS],
        "inmover_dist": [inmover_probs[b] for b in eng.BRACKETS],
        "general_pop_dist": [gen_probs[b] for b in eng.BRACKETS],
    })
    for j, bucket in enumerate(buckets):
        inputs_df[f"tenure_prob_{bucket}"] = [tenure.probs_by_bracket[b][j] for b in eng.BRACKETS]
    inputs_df.to_csv(args.outdir / "inputs_used.csv", index=False)

    # --- Save adjusted profiles used ---
    prof = eng.make_profiles(rates)
    profiles_df = pd.DataFrame({
        "age_bracket": eng.BRACKETS,
        "adj_any": [prof["adj_any"][b] for b in eng.BRACKETS],
        "cond_severe_given_any": [prof["cond_severe"][b] for b in eng.BRACKETS],
        "cond_motor_phys_given_any": [prof["cond_phys"][b] for b in eng.BRACKETS],
        "adj_phys2": [prof["adj_phys2"][b] for b in eng.BRACKETS],
    })
    profiles_df.to_csv(args.outdir / "profiles_used.csv", index=False)

    # -----------------
    # Run scenarios
    # -----------------

    summaries = []

    # Main
    params_main = eng.SimParams(
        n_props=args.n_props,
        seed=args.seed,
        first_draw_source="inmover",
        horizon_years=args.horizon_years,
    )

    print("\n=== Running scenario: main_inmover_first ===")
    summary_main = eng.run_sim(params_main, rates, tenure, inmovers, return_time_stats=True)
    summaries.append(("main_inmover_first", summary_main))

    # Sensitivity: first household from general population
    params_general_first = eng.SimParams(
        n_props=args.n_props,
        seed=args.seed,
        first_draw_source="general",
        horizon_years=args.horizon_years,
    )
    print("\n=== Running scenario: in-movers drawn from general population ===")
    summary_general_first = eng.run_sim(
        params_general_first, rates, tenure, inmovers,
        general_pop_probs=gen_probs,
        return_time_stats=True,
    )
    summaries.append(("sens_general_first", summary_general_first))

    # Sensitivity: longer tenure if disabled
    params_long_dis = eng.SimParams(
        n_props=args.n_props,
        seed=args.seed,
        first_draw_source="inmover",
        horizon_years=args.horizon_years,
        disabled_tenure_factor=1.2,
    )
    print("\n=== Running scenario: longer tenure for disabled household ===")
    summary_long_dis = eng.run_sim(params_long_dis, rates, tenure, inmovers, return_time_stats=True)
    summaries.append(("sens_disabled_tenure_1p2", summary_long_dis))

    # Example selection scenario: scale all rates by 0.9
    scale = 0.9
    rates_scaled = eng.AllRates(
        any_dis=eng.Rates({b: scale * float(rates_any[b]) for b in eng.BRACKETS}),
        severe_prof=eng.Rates({b: scale * float(rates_sev[b]) for b in eng.BRACKETS}),
        motor_phys=eng.Rates({b: scale * float(rates_mot[b]) for b in eng.BRACKETS}),
        phys2=eng.Rates({b: scale * float(rates_phys2[b]) for b in eng.BRACKETS}),
    )
    params_sel = eng.SimParams(
        n_props=args.n_props,
        seed=args.seed,
        first_draw_source="inmover",
        horizon_years=args.horizon_years,
    )
    print("\n=== Running scenario: people with disability avoid inaccessible homes ===")
    summary_selection = eng.run_sim(params_sel, rates_scaled, tenure, inmovers, return_time_stats=True)
    summaries.append(("selection_scale_0p9", summary_selection))

    # Save scenario summaries (single file)
    sum_df = pd.DataFrame([{"scenario": name, **vals} for name, vals in summaries])
    sum_df.to_csv(args.outdir / "scenario_summaries.csv", index=False)

    # Console output
    def _pp(title: str, d: Dict[str, float]) -> None:
        print(f"\n{title}\n" + "-" * len(title))
        for k in sorted(d.keys()):
            print(f"{k:26s}: {d[k]:.6f}")

    _pp("Main scenario (first household = in-mover)", summary_main)
    _pp("Sensitivity (first household = general population)", summary_general_first)
    _pp("Sensitivity (disabled tenure factor = 1.2)", summary_long_dis)
    _pp("Selection (rates scaled by 0.9)", summary_selection)

    if args.verbose:
        print(f"\nOutputs written to: {args.outdir.resolve()}")
        print(f"Engine file: {Path(eng.__file__).resolve()}")
        print(f"Working dir: {Path.cwd().resolve()}")


if __name__ == "__main__":
    main()