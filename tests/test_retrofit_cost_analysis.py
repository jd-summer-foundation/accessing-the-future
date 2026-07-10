"""Cross-checks of the cost engine against the original prototype spreadsheet.

The fixture tests/fixtures/spreadsheet_first_occupancy_cdf.csv contains the
first-occupancy CDF exactly as used in the prototype workbook
"Projected costs of retrofit vs initial build 2.0", so the expected values
below are the workbook's own outputs. Two deliberate differences from the
workbook are covered:

- the workbook's projection rows stop 20 years after the analysis start, so
  the year-20 CDF increment is silently dropped; the script uses the full CDF.
  The exact cross-check therefore clamps the fixture's year-20 value.
- the workbook inflated WA costs with the NSW rate from 2028 onward (copy
  error); the script uses each state's own rate, so only NSW figures are
  compared against the workbook.
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.pipeline_utils import DEFAULT_CONSTRUCTION_INDEX_CSV, load_yaml
from scripts.retrofit_cost_analysis import DEFAULT_COST_CONFIG, run_cost_analysis

ROOT = Path(__file__).resolve().parents[1]
SPREADSHEET_CDF = ROOT / "tests/fixtures/spreadsheet_first_occupancy_cdf.csv"


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(DEFAULT_COST_CONFIG)


def _results_dir_with_cdf(tmp_path: Path, cdf: pd.DataFrame) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cdf.to_csv(results_dir / "first_occupancy_cdf.csv", index=False)
    return results_dir


def _clamp_year20(cdf: pd.DataFrame) -> pd.DataFrame:
    """Zero the year-20 increment to mimic the workbook's truncated projection rows."""
    out = cdf.copy()
    year19 = out.loc[out["year"] == 19].set_index(["scenario", "category"])["cum_share_first_occupancy"]
    mask = out["year"] == 20
    out.loc[mask, "cum_share_first_occupancy"] = [
        year19[(scenario, category)]
        for scenario, category in zip(out.loc[mask, "scenario"], out.loc[mask, "category"])
    ]
    return out


def _summary_row(summary: pd.DataFrame, state: str, case: str, category: str) -> pd.Series:
    rows = summary.loc[
        (summary["state"] == state)
        & (summary["uncertainty_case"] == case)
        & (summary["category"] == category)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_nsw_no_stagger_matches_spreadsheet_exactly(tmp_path: Path, config: dict) -> None:
    cdf = _clamp_year20(pd.read_csv(SPREADSHEET_CDF))
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    outputs = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, stagger_override=False)
    summary = outputs["summary"]

    # Workbook Projection!I8/I7 (per-dwelling and total NPV of the new-build arm).
    nsw_base_physical = _summary_row(summary, "nsw", "base", "physical")
    assert nsw_base_physical["new_build_npv_per_dwelling"] == pytest.approx(5423.637416158001, rel=1e-12)
    assert nsw_base_physical["new_build_npv_total_billion"] == pytest.approx(2.039287668475408, rel=1e-12)

    # Workbook Projection!J8..O8 (per-dwelling retrofit NPV per case/category).
    expected_retrofit = {
        ("low", "physical"): 16816.05116012252,
        ("low", "any"): 19444.19020375241,
        ("base", "physical"): 17739.07104595847,
        ("base", "any"): 20144.627684577594,
        ("high", "physical"): 18488.397139705074,
        ("high", "any"): 20768.363825737066,
    }
    for (case, category), expected in expected_retrofit.items():
        row = _summary_row(summary, "nsw", case, category)
        assert row["retrofit_npv_per_dwelling"] == pytest.approx(expected, rel=1e-12)

    # Workbook Summary!E10 (base physical retrofit total, $B).
    assert nsw_base_physical["retrofit_npv_total_billion"] == pytest.approx(6.669890713280385, rel=1e-12)


def test_full_cdf_raises_retrofit_npv_above_truncated_workbook(tmp_path: Path, config: dict) -> None:
    cdf = pd.read_csv(SPREADSHEET_CDF)
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    outputs = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, stagger_override=False)
    row = _summary_row(outputs["summary"], "nsw", "base", "physical")
    # Including the year-20 increment the workbook dropped must increase the NPV.
    assert row["retrofit_npv_per_dwelling"] > 17739.072


def test_stagger_scales_both_arms_by_the_same_cohort_factor(tmp_path: Path, config: dict) -> None:
    cdf = pd.read_csv(SPREADSHEET_CDF)
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    staggered = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, stagger_override=True)
    unstaggered = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, stagger_override=False)

    discount_rate = float(config["analysis"]["discount_rate"])
    buildout = int(config["analysis"]["buildout_years"])

    for state in ("nsw", "wa"):
        inflation = float(
            staggered["assumptions"].loc[staggered["assumptions"]["state"] == state, "avg_inflation"].iloc[0]
        )
        # Delaying a cohort by one year inflates its nominal costs by (1+i) and
        # discounts them by (1+r); averaging over equal cohorts gives the exact
        # ratio between staggered and start-year retrofit NPVs.
        factor = sum(((1.0 + inflation) / (1.0 + discount_rate)) ** b for b in range(buildout)) / buildout
        row_s = _summary_row(staggered["summary"], state, "base", "physical")
        row_u = _summary_row(unstaggered["summary"], state, "base", "physical")
        assert row_s["retrofit_npv_per_dwelling"] == pytest.approx(
            row_u["retrofit_npv_per_dwelling"] * factor, rel=1e-12
        )
        # The new-build arm is always cohort-spread, so it is identical in both modes.
        assert row_s["new_build_npv_per_dwelling"] == pytest.approx(
            row_u["new_build_npv_per_dwelling"], rel=1e-12
        )
        # With inflation below the discount rate, staggering lowers the retrofit NPV.
        assert row_s["retrofit_npv_per_dwelling"] < row_u["retrofit_npv_per_dwelling"]


def test_totals_row_aggregates_states(tmp_path: Path, config: dict) -> None:
    cdf = pd.read_csv(SPREADSHEET_CDF)
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    outputs = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV)
    summary = outputs["summary"]

    nsw = _summary_row(summary, "nsw", "base", "physical")
    wa = _summary_row(summary, "wa", "base", "physical")
    total = _summary_row(summary, "total", "base", "physical")
    assert total["new_homes"] == nsw["new_homes"] + wa["new_homes"]
    for column in ("new_build_npv_total_billion", "retrofit_npv_total_billion", "difference_billion"):
        assert total[column] == pytest.approx(nsw[column] + wa[column], rel=1e-12)

    # Cashflow transparency output covers build-out plus the CDF horizon for every combination.
    cashflows = outputs["cashflows"]
    horizon = outputs["horizon"]
    n_years = int(config["analysis"]["buildout_years"]) + horizon
    assert len(cashflows) == 2 * 3 * 2 * n_years  # states x cases x categories x years
    start_year = int(config["analysis"]["start_year"])
    assert cashflows["calendar_year"].min() == start_year
    assert cashflows["calendar_year"].max() == start_year + n_years - 1
    # Discounted cashflows must sum back to the reported NPVs.
    subset = cashflows.loc[
        (cashflows["state"] == "nsw")
        & (cashflows["uncertainty_case"] == "base")
        & (cashflows["category"] == "physical")
    ]
    assert subset["retrofit_discounted_per_dwelling"].sum() == pytest.approx(
        nsw["retrofit_npv_per_dwelling"], rel=1e-12
    )
    assert subset["new_build_discounted_per_dwelling"].sum() == pytest.approx(
        nsw["new_build_npv_per_dwelling"], rel=1e-12
    )
