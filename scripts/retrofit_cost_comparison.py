#!/usr/bin/env python3
"""Discounted cost comparison: build-in accessibility vs retrofit-at-first-need.

Compares two hypothetical routes to delivering LHDS accessibility to a cohort
of new dwellings:

  A. Build-in: every dwelling incorporates the features at construction, at
     --build-in-cost per dwelling (cost incurred at year 0).
  B. Retrofit-at-first-need: a dwelling is retrofitted at --retrofit-cost in
     the year it is FIRST occupied by a household with the relevant disability
     category, using the first-occupancy cumulative distribution produced by
     scripts/first_occupancy_analysis.py. Dwellings never occupied by such a
     household within the horizon are never retrofitted.

Route B is a delivery-route comparator, NOT a behavioural prediction: it does
not assume households actually retrofit, only asks what it would cost to
deliver equivalent accessibility through retrofitting. Costs after the
simulation horizon are excluded, which is conservative in favour of route B.

Default costs are the Centre for International Economics 2021 Decision RIS
estimates for a Class 1a (detached) dwelling, in 2021 dollars: ~$4,000 to
build in vs ~$19,000 to retrofit. The default dwelling count (505,000) is the
combined NSW (376,000) and WA (129,000) National Housing Accord shares.

Outputs (in --outdir, default results/retrofit_cost/):
  retrofit_cost_comparison.csv
  run_manifest.json
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
    artifact_checksums,
    dependency_versions,
    git_commit,
    serialise_for_json,
    sha256_file,
    utc_now_iso,
    write_json,
)

DEFAULT_CDF = ROOT / "results/first_occupancy/first_occupancy_cdf.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cdf", type=Path, default=DEFAULT_CDF,
                        help="first_occupancy_cdf.csv from scripts/first_occupancy_analysis.py.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/retrofit_cost")
    parser.add_argument("--build-in-cost", type=float, default=4_000.0,
                        help="Cost per dwelling to include features at construction (CIE 2021, Class 1a).")
    parser.add_argument("--retrofit-cost", type=float, default=19_000.0,
                        help="Cost per dwelling to retrofit the same features (CIE 2021, Class 1a).")
    parser.add_argument("--dwellings", type=int, default=505_000,
                        help="Cohort size the per-dwelling costs are scaled to (NSW+WA Accord shares).")
    parser.add_argument("--discount-rates", type=float, nargs="+",
                        default=[0.0, 0.03, 0.05, 0.07],
                        help="Annual real discount rates to evaluate.")
    return parser.parse_args()


def expected_discounted_retrofit_cost(
    cdf: pd.Series,
    retrofit_cost: float,
    rate: float,
) -> float:
    """Expected present cost per dwelling of retrofitting at first need.

    cdf is indexed by integer year and gives the cumulative share of ALL
    dwellings whose first disability occupancy has happened by end of year t;
    the year-on-year increments are the shares first needing the features in
    each year, each costed at retrofit_cost and discounted back to year 0.
    """
    increments = cdf.diff()
    increments.iloc[0] = cdf.iloc[0]
    return float(sum(
        increments.loc[year] * retrofit_cost / (1.0 + rate) ** year
        for year in cdf.index
    ))


def main() -> None:
    args = parse_args()
    cdf_df = pd.read_csv(args.cdf)

    rows = []
    for (uncertainty_case, category), group in cdf_df.groupby(["uncertainty_case", "category"]):
        cdf = group.sort_values("year").set_index("year")["cum_share_first_occupancy"]
        for rate in args.discount_rates:
            per_dwelling = expected_discounted_retrofit_cost(cdf, args.retrofit_cost, rate)
            rows.append({
                "uncertainty_case": uncertainty_case,
                "category": category,
                "discount_rate": rate,
                "horizon_years": int(cdf.index.max()),
                "p_ever_within_horizon": float(cdf.iloc[-1]),
                "build_in_cost_per_dwelling": args.build_in_cost,
                "retrofit_cost_if_needed": args.retrofit_cost,
                "expected_retrofit_cost_per_dwelling": per_dwelling,
                "cost_ratio_retrofit_vs_build_in": per_dwelling / args.build_in_cost,
                "dwellings": args.dwellings,
                "total_build_in_cost": args.build_in_cost * args.dwellings,
                "total_retrofit_cost": per_dwelling * args.dwellings,
            })

    result_df = pd.DataFrame(rows).sort_values(
        ["category", "uncertainty_case", "discount_rate"]
    ).reset_index(drop=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "retrofit_cost_comparison.csv"
    result_df.to_csv(output_path, index=False)

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "run_name": "retrofit_cost_comparison",
        "git_commit": git_commit(ROOT),
        "python_version": sys.version.split()[0],
        "dependency_versions": dependency_versions(),
        "input": {"path": args.cdf, "sha256": sha256_file(args.cdf)},
        "parameters": serialise_for_json({
            "build_in_cost": args.build_in_cost,
            "retrofit_cost": args.retrofit_cost,
            "dwellings": args.dwellings,
            "discount_rates": args.discount_rates,
        }),
        "outputs": {
            "directory": outdir,
            "artifact_checksums": artifact_checksums([output_path], relative_to=outdir),
        },
    }
    write_json(outdir / "run_manifest.json", manifest)

    pd.set_option("display.width", 220)
    shown = result_df[result_df["category"] == "physical"][[
        "uncertainty_case", "discount_rate", "p_ever_within_horizon",
        "expected_retrofit_cost_per_dwelling", "cost_ratio_retrofit_vs_build_in",
        "total_retrofit_cost", "total_build_in_cost",
    ]]
    print("Retrofit-at-first-need vs build-in (physical disability):")
    print(shown.to_string(index=False))
    print(f"\nOutputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
