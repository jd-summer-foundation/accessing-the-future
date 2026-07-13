#!/usr/bin/env python3
"""Retrofit-as-needed vs. accessible-new-build NPV comparison.

Combines three reproducible inputs:
1. the model's first-occupancy CDF (results/<run>/first_occupancy_cdf.csv):
   share of new dwellings first occupied by a household with a physical / any
   disability within y years of being built;
2. the processed ABS house-construction cost index
   (data/processed/construction_index.csv, from 6427.0 Table 17), which anchors
   the CIE DRIS 2021 costs to the analysis start year and sets state-specific
   forward inflation;
3. the processed new-dwelling structure-type mix
   (data/processed/dwelling_mix.csv, from ABS 8752.0 Building Activity), which
   weights the CIE DRIS per-type costs into per-state figures;
4. assumptions in configs/cost_analysis.yaml (per-type costs, discount rate,
   build-out, National Housing Accord home counts).

Both arms discount nominal expected costs to the analysis start year:
- New-build arm: every home incurs the accessibility cost in its build year
  (equal cohorts over the build-out period).
- Retrofit arm: a home built in year b incurs the retrofit cost in year b + k
  with probability equal to the CDF increment at year k.

The retrofit arm is a conservative LOWER bound: first occupancies beyond the
model's horizon (20 years per cohort) are excluded, and by then the CDF has
only reached roughly 68-83% of dwellings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_utils import (
    DEFAULT_CONSTRUCTION_INDEX_CSV,
    DEFAULT_REPORTS_DIR,
    DEFAULT_RESULTS_DIR,
    DWELLING_TYPES,
    REPO_ROOT,
    artifact_checksums,
    load_yaml,
    manifest_path,
    sha256_file,
    serialise_for_json,
    utc_now_iso,
    write_json,
)

DEFAULT_COST_CONFIG = REPO_ROOT / "configs/cost_analysis.yaml"
SUMMARY_NAME = "retrofit_vs_newbuild_summary"
CASHFLOW_NAME = "retrofit_vs_newbuild_cashflows"
MANIFEST_NAME = "cost_analysis_manifest.json"

UNCERTAINTY_CASES = ["low", "base", "high"]
CATEGORIES = ["physical", "any"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NPV of accessible new builds vs. retrofitting on first disabled occupancy.")
    parser.add_argument("--config", type=Path, default=DEFAULT_COST_CONFIG, help="Path to cost analysis YAML config.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Model run directory containing first_occupancy_cdf.csv.")
    parser.add_argument("--index-csv", type=Path, default=DEFAULT_CONSTRUCTION_INDEX_CSV, help="Processed construction index CSV.")
    parser.add_argument("--mix-csv", type=Path, default=None, help="Processed dwelling mix CSV (default: dwelling_mix_csv from the config).")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory where tables will be written.")
    return parser.parse_args()


def resolve_state_costs(
    costs_cfg: Dict[str, object],
    states: List[str],
    dwelling_mix_csv: Path | None,
) -> Dict[str, Dict[str, float]]:
    """Per-state base-year costs for each arm.

    `new_build_by_type` / `retrofit_by_type` are weighted by each state's
    new-dwelling structure-type mix (house / townhouse / apartment
    commencement shares), and `new_build_mandate_overhead` (compliance
    verification plus transition costs) is added to the new-build arm.
    """
    if dwelling_mix_csv is None:
        raise ValueError("By-type costs require a dwelling mix CSV (dwelling_mix_csv)")
    mix = pd.read_csv(dwelling_mix_csv)
    new_build_by_type = {t: float(v) for t, v in dict(costs_cfg["new_build_by_type"]).items()}
    retrofit_by_type = {t: float(v) for t, v in dict(costs_cfg["retrofit_by_type"]).items()}
    overhead = float(costs_cfg.get("new_build_mandate_overhead", 0.0))
    for label, by_type in (("new_build_by_type", new_build_by_type), ("retrofit_by_type", retrofit_by_type)):
        if sorted(by_type) != sorted(DWELLING_TYPES):
            raise ValueError(f"{label} must cover exactly {DWELLING_TYPES}, found {sorted(by_type)}")

    resolved: Dict[str, Dict[str, float]] = {}
    for state in states:
        rows = mix.loc[mix["state"] == state]
        shares = dict(zip(rows["dwelling_type"], rows["share"].astype(float)))
        if sorted(shares) != sorted(DWELLING_TYPES):
            raise ValueError(f"Dwelling mix for {state} must cover exactly {DWELLING_TYPES}, found {sorted(shares)}")
        if abs(sum(shares.values()) - 1.0) > 1e-9:
            raise ValueError(f"Dwelling mix shares for {state} sum to {sum(shares.values())}, expected 1")
        resolved[state] = {
            "new_build": sum(shares[t] * new_build_by_type[t] for t in DWELLING_TYPES) + overhead,
            "retrofit": sum(shares[t] * retrofit_by_type[t] for t in DWELLING_TYPES),
        }
    return resolved


def average_inflation(index: pd.Series, window: tuple[int, int]) -> float:
    """Mean year-on-year change of a year-indexed series, for change end-years in the window (inclusive)."""
    first, last = int(window[0]), int(window[1])
    changes = [index[year] / index[year - 1] - 1.0 for year in range(first, last + 1)]
    return float(np.mean(changes))


def cost_path(
    base_cost: float,
    index: pd.Series,
    base_year: int,
    start_year: int,
    inflation: float,
    n_years: int,
) -> np.ndarray:
    """Nominal per-dwelling cost for start_year + 0..n_years-1.

    Anchored to the observed index level at start_year (relative to base_year),
    then inflated forward at the historical average rate.
    """
    anchored = float(base_cost) * float(index[start_year]) / float(index[base_year])
    return anchored * (1.0 + inflation) ** np.arange(n_years, dtype=float)


def load_first_occupancy_increments(
    results_dir: Path,
    scenario: str,
) -> Dict[tuple[str, str], np.ndarray]:
    """Per-year first-occupancy probability increments, keyed by (uncertainty_case, category)."""
    cdf_path = results_dir / "first_occupancy_cdf.csv"
    if not cdf_path.exists():
        raise FileNotFoundError(
            f"Missing {cdf_path}. Run the model with run.return_first_occupancy: true first."
        )
    df = pd.read_csv(cdf_path)
    increments: Dict[tuple[str, str], np.ndarray] = {}
    for case in UNCERTAINTY_CASES:
        for category in CATEGORIES:
            rows = df.loc[
                (df["scenario"] == f"{scenario}_{case}") & (df["category"] == category)
            ].sort_values("year")
            if rows.empty:
                raise ValueError(f"No CDF rows for scenario {scenario}_{case}, category {category}")
            years = rows["year"].to_numpy()
            if not np.array_equal(years, np.arange(len(years))):
                raise ValueError(f"CDF years must be 0..H for {scenario}_{case}/{category}, found {years}")
            cdf = rows["cum_share_first_occupancy"].to_numpy(dtype=float)
            increments[(case, category)] = np.diff(cdf, prepend=0.0)
    return increments


def expected_yearly_costs(
    per_start_costs: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    n_years: int,
) -> np.ndarray:
    """Expected nominal cost per original dwelling for each year offset from start_year.

    probabilities[k] is the chance a home incurs the cost k years after ITS
    clock start; weights[b] is the share of homes whose clock starts at offset
    b. per_start_costs[j] is the nominal unit cost at offset j from start_year.
    """
    out = np.zeros(n_years)
    for b, weight in enumerate(weights):
        out[b : b + len(probabilities)] += weight * probabilities * per_start_costs[b : b + len(probabilities)]
    return out


def run_cost_analysis(
    config: Dict[str, object],
    results_dir: Path,
    index_csv: Path,
    *,
    dwelling_mix_csv: Path | None = None,
) -> Dict[str, pd.DataFrame]:
    costs_cfg = config["costs"]
    analysis_cfg = config["analysis"]
    states_cfg = config["states"]

    base_year = int(costs_cfg["base_year"])
    if dwelling_mix_csv is None and "dwelling_mix_csv" in config:
        dwelling_mix_csv = REPO_ROOT / str(config["dwelling_mix_csv"])
    state_costs = resolve_state_costs(costs_cfg, list(states_cfg), dwelling_mix_csv)
    start_year = int(analysis_cfg["start_year"])
    buildout_years = int(analysis_cfg["buildout_years"])
    discount_rate = float(analysis_cfg["discount_rate"])
    window = tuple(int(y) for y in analysis_cfg["inflation_window"])
    scenario = str(analysis_cfg["scenario"])

    index_df = pd.read_csv(index_csv)
    increments = load_first_occupancy_increments(results_dir, scenario)
    horizon = len(next(iter(increments.values()))) - 1

    # Build cohorts occur in years 0..buildout-1; the last cohort's CDF runs a
    # further `horizon` years beyond its build year.
    n_years = buildout_years + horizon
    discount = (1.0 + discount_rate) ** -np.arange(n_years, dtype=float)
    # Homes are built in equal annual cohorts; each cohort's retrofit clock
    # starts in its build year, symmetrically with the new-build arm.
    build_weights = np.full(buildout_years, 1.0 / buildout_years)
    retrofit_weights = build_weights

    assumption_rows = []
    summary_rows = []
    cashflow_rows = []
    totals: Dict[tuple[str, str], Dict[str, float]] = {}

    for state, state_cfg in states_cfg.items():
        new_homes = int(state_cfg["new_homes"])
        index = pd.Series(
            index_df[f"{state}_index"].to_numpy(dtype=float),
            index=index_df["year"].astype(int).to_numpy(),
        )
        inflation = average_inflation(index, window)
        newbuild_path = cost_path(state_costs[state]["new_build"], index, base_year, start_year, inflation, n_years)
        retrofit_path = cost_path(state_costs[state]["retrofit"], index, base_year, start_year, inflation, n_years)

        assumption_rows.append(
            {
                "state": state,
                "new_homes": new_homes,
                "avg_inflation": inflation,
                f"new_build_cost_{base_year}": state_costs[state]["new_build"],
                f"retrofit_cost_{base_year}": state_costs[state]["retrofit"],
                f"new_build_cost_{start_year}": float(newbuild_path[0]),
                f"retrofit_cost_{start_year}": float(retrofit_path[0]),
            }
        )

        # New-build arm: probability 1 in the build year, spread across cohorts.
        newbuild_yearly = expected_yearly_costs(newbuild_path, np.array([1.0]), build_weights, n_years)
        newbuild_npv = float((newbuild_yearly * discount).sum())

        for case in UNCERTAINTY_CASES:
            for category in CATEGORIES:
                retrofit_yearly = expected_yearly_costs(
                    retrofit_path, increments[(case, category)], retrofit_weights, n_years
                )
                retrofit_npv = float((retrofit_yearly * discount).sum())

                summary_rows.append(
                    {
                        "state": state,
                        "uncertainty_case": case,
                        "category": category,
                        "new_homes": new_homes,
                        "new_build_npv_per_dwelling": newbuild_npv,
                        "retrofit_npv_per_dwelling": retrofit_npv,
                        "new_build_npv_total_billion": newbuild_npv * new_homes / 1e9,
                        "retrofit_npv_total_billion": retrofit_npv * new_homes / 1e9,
                        "difference_billion": (retrofit_npv - newbuild_npv) * new_homes / 1e9,
                    }
                )
                key = (case, category)
                bucket = totals.setdefault(key, {"new_homes": 0.0, "new_build": 0.0, "retrofit": 0.0})
                bucket["new_homes"] += new_homes
                bucket["new_build"] += newbuild_npv * new_homes
                bucket["retrofit"] += retrofit_npv * new_homes

                for offset in range(n_years):
                    cashflow_rows.append(
                        {
                            "state": state,
                            "uncertainty_case": case,
                            "category": category,
                            "calendar_year": start_year + offset,
                            "discount_factor": float(discount[offset]),
                            "new_build_expected_cost_per_dwelling": float(newbuild_yearly[offset]),
                            "retrofit_expected_cost_per_dwelling": float(retrofit_yearly[offset]),
                            "new_build_discounted_per_dwelling": float(newbuild_yearly[offset] * discount[offset]),
                            "retrofit_discounted_per_dwelling": float(retrofit_yearly[offset] * discount[offset]),
                        }
                    )

    for (case, category), bucket in totals.items():
        homes = bucket["new_homes"]
        summary_rows.append(
            {
                "state": "total",
                "uncertainty_case": case,
                "category": category,
                "new_homes": int(homes),
                "new_build_npv_per_dwelling": bucket["new_build"] / homes,
                "retrofit_npv_per_dwelling": bucket["retrofit"] / homes,
                "new_build_npv_total_billion": bucket["new_build"] / 1e9,
                "retrofit_npv_total_billion": bucket["retrofit"] / 1e9,
                "difference_billion": (bucket["retrofit"] - bucket["new_build"]) / 1e9,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["uncertainty_case", "category", "state"],
        key=lambda col: col.map({"low": 0, "base": 1, "high": 2, "physical": 0, "any": 1, "nsw": 0, "wa": 1, "total": 2}).fillna(99)
        if col.name in {"uncertainty_case", "category", "state"}
        else col,
    ).reset_index(drop=True)
    cashflows = pd.DataFrame(cashflow_rows)
    assumptions = pd.DataFrame(assumption_rows)
    return {"summary": summary, "cashflows": cashflows, "assumptions": assumptions, "horizon": horizon}


def _write_markdown_summary(summary: pd.DataFrame, assumptions: pd.DataFrame, horizon: int, path: Path) -> None:
    display = summary.copy()
    for column in ["new_build_npv_per_dwelling", "retrofit_npv_per_dwelling"]:
        display[column] = display[column].map(lambda v: f"{v:,.0f}")
    for column in ["new_build_npv_total_billion", "retrofit_npv_total_billion", "difference_billion"]:
        display[column] = display[column].map(lambda v: f"{v:,.2f}")

    lines = [
        "# Retrofit-as-needed vs. accessible new build",
        "",
        "NPV of expected per-dwelling and total costs, discounted to the analysis start year.",
        "Retrofit clocks are staggered by build cohort.",
        f"The retrofit arm is a conservative lower bound: first occupancies more than {horizon} years",
        "after a home is built fall outside the model horizon and are excluded.",
        "",
        "Derived state assumptions:",
        "",
    ]
    lines.append("| " + " | ".join(assumptions.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(assumptions.columns)) + " |")
    for _, row in assumptions.iterrows():
        formatted = [
            f"{value:,.4f}" if isinstance(value, float) else str(value)
            for value in row.tolist()
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    lines.append("")
    lines.append("| " + " | ".join(display.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    mix_csv = args.mix_csv
    if mix_csv is None and "dwelling_mix_csv" in config:
        mix_csv = REPO_ROOT / str(config["dwelling_mix_csv"])
    outputs = run_cost_analysis(
        config,
        args.results_dir,
        args.index_csv,
        dwelling_mix_csv=mix_csv,
    )

    tables_dir = args.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = tables_dir / f"{SUMMARY_NAME}.csv"
    summary_md = tables_dir / f"{SUMMARY_NAME}.md"
    cashflow_csv = tables_dir / f"{CASHFLOW_NAME}.csv"

    outputs["summary"].to_csv(summary_csv, index=False)
    outputs["cashflows"].to_csv(cashflow_csv, index=False)
    _write_markdown_summary(outputs["summary"], outputs["assumptions"], outputs["horizon"], summary_md)

    generated = [summary_csv, summary_md, cashflow_csv]
    cdf_path = args.results_dir / "first_occupancy_cdf.csv"
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "config": {
            "path": args.config,
            "sha256": sha256_file(args.config),
            "values": serialise_for_json(config),
        },
        "inputs": {
            "first_occupancy_cdf": {
                "path": manifest_path(cdf_path),
                "sha256": sha256_file(cdf_path),
            },
            "construction_index": {
                "path": manifest_path(args.index_csv),
                "sha256": sha256_file(args.index_csv),
            },
            **(
                {
                    "dwelling_mix": {
                        "path": manifest_path(mix_csv),
                        "sha256": sha256_file(mix_csv),
                    }
                }
                if mix_csv is not None
                else {}
            ),
        },
        "derived_assumptions": outputs["assumptions"].to_dict(orient="records"),
        "artifacts": [str(path.relative_to(args.reports_dir)) for path in generated],
        "artifact_checksums": artifact_checksums(generated, relative_to=args.reports_dir),
    }
    write_json(args.reports_dir / MANIFEST_NAME, manifest)

    headline = outputs["summary"]
    headline = headline.loc[(headline["uncertainty_case"] == "base") & (headline["category"] == "physical")]
    print(headline.to_string(index=False))
    print(f"\nTables written to {tables_dir}")


if __name__ == "__main__":
    main()
