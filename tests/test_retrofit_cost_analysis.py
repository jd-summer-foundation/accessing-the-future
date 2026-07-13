"""Engine checks for the retrofit vs. accessible-new-build cost comparison.

These tests exercise the cost engine with synthetic first-occupancy CDFs whose
NPVs can be reasoned about (or computed) directly in the test, plus the
resolution of per-state costs from the CIE DRIS per-type figures and the ABS
dwelling mix.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pipeline_utils import DEFAULT_CONSTRUCTION_INDEX_CSV, DEFAULT_DWELLING_MIX_CSV, load_yaml
from scripts.retrofit_cost_analysis import (
    DEFAULT_COST_CONFIG,
    average_inflation,
    resolve_state_costs,
    run_cost_analysis,
)

UNCERTAINTY_CASES = ["low", "base", "high"]
CATEGORIES = ["physical", "any"]
HORIZON = 20


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(DEFAULT_COST_CONFIG)


def _synthetic_cdf(scenario: str, cdf_values: np.ndarray) -> pd.DataFrame:
    """A first-occupancy CDF file with the same curve for every case/category."""
    rows = []
    for case in UNCERTAINTY_CASES:
        for category in CATEGORIES:
            for year, value in enumerate(cdf_values):
                rows.append(
                    {
                        "scenario": f"{scenario}_{case}",
                        "category": category,
                        "year": year,
                        "cum_share_first_occupancy": value,
                    }
                )
    return pd.DataFrame(rows)


def _results_dir_with_cdf(tmp_path: Path, cdf: pd.DataFrame) -> Path:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    cdf.to_csv(results_dir / "first_occupancy_cdf.csv", index=False)
    return results_dir


def _summary_row(summary: pd.DataFrame, state: str, case: str, category: str) -> pd.Series:
    rows = summary.loc[
        (summary["state"] == state)
        & (summary["uncertainty_case"] == case)
        & (summary["category"] == category)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_state_costs_weight_cie_dris_by_dwelling_mix() -> None:
    """Default config: per-type CIE DRIS costs weighted by the ABS 8752.0 mix."""
    cfg = load_yaml(DEFAULT_COST_CONFIG)
    resolved = resolve_state_costs(cfg["costs"], ["nsw", "wa"], DEFAULT_DWELLING_MIX_CSV)

    mix = pd.read_csv(DEFAULT_DWELLING_MIX_CSV)
    for state in ("nsw", "wa"):
        shares = dict(zip(mix.loc[mix["state"] == state, "dwelling_type"], mix.loc[mix["state"] == state, "share"]))
        expected_new_build = (
            shares["house"] * 3874 + shares["townhouse"] * 4186 + shares["apartment"] * 5748 + 215
        )
        expected_retrofit = (shares["house"] + shares["townhouse"]) * 18821 + shares["apartment"] * 20260
        assert resolved[state]["new_build"] == pytest.approx(expected_new_build, rel=1e-12)
        assert resolved[state]["retrofit"] == pytest.approx(expected_retrofit, rel=1e-12)

    # Guard against silent regressions in the derived 2021-dollar figures.
    assert resolved["nsw"]["new_build"] == pytest.approx(4745.4, abs=0.5)
    assert resolved["nsw"]["retrofit"] == pytest.approx(19273.7, abs=0.5)
    assert resolved["wa"]["new_build"] == pytest.approx(4280.5, abs=0.5)
    assert resolved["wa"]["retrofit"] == pytest.approx(18950.2, abs=0.5)


def test_by_type_costs_require_dwelling_mix() -> None:
    cfg = load_yaml(DEFAULT_COST_CONFIG)
    with pytest.raises(ValueError, match="dwelling mix"):
        resolve_state_costs(cfg["costs"], ["nsw"], None)


def test_immediate_occupancy_makes_arms_equal_when_costs_match(tmp_path: Path, config: dict) -> None:
    """If every home is first occupied by a triggering household in its build
    year (CDF = 1 at year 0) and the retrofit cost equals the new-build cost,
    the two arms must produce identical NPVs: the cohort staggering treats
    both arms symmetrically."""
    cfg = {**config, "costs": dict(config["costs"])}
    cfg["costs"]["new_build_by_type"] = {"house": 1000, "townhouse": 1000, "apartment": 1000}
    cfg["costs"]["retrofit_by_type"] = {"house": 1000, "townhouse": 1000, "apartment": 1000}
    cfg["costs"]["new_build_mandate_overhead"] = 0

    cdf = _synthetic_cdf(str(config["analysis"]["scenario"]), np.ones(HORIZON + 1))
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    outputs = run_cost_analysis(cfg, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, dwelling_mix_csv=DEFAULT_DWELLING_MIX_CSV)

    for state in ("nsw", "wa"):
        row = _summary_row(outputs["summary"], state, "base", "physical")
        assert row["retrofit_npv_per_dwelling"] == pytest.approx(row["new_build_npv_per_dwelling"], rel=1e-12)


def test_npv_matches_closed_form(tmp_path: Path, config: dict) -> None:
    """Cross-check both arms against a direct closed-form computation."""
    analysis = config["analysis"]
    scenario = str(analysis["scenario"])
    start_year = int(analysis["start_year"])
    base_year = int(config["costs"]["base_year"])
    buildout = int(analysis["buildout_years"])
    rate = float(analysis["discount_rate"])

    # CDF reaching 60% linearly over the horizon.
    cdf_values = np.linspace(0.0, 0.6, HORIZON + 1)
    results_dir = _results_dir_with_cdf(tmp_path, _synthetic_cdf(scenario, cdf_values))
    outputs = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, dwelling_mix_csv=DEFAULT_DWELLING_MIX_CSV)

    index_df = pd.read_csv(DEFAULT_CONSTRUCTION_INDEX_CSV).set_index("year")
    state_costs = resolve_state_costs(config["costs"], ["nsw", "wa"], DEFAULT_DWELLING_MIX_CSV)
    increments = np.diff(cdf_values, prepend=0.0)

    for state in ("nsw", "wa"):
        index = index_df[f"{state}_index"]
        inflation = average_inflation(index, tuple(analysis["inflation_window"]))
        anchor = index[start_year] / index[base_year]

        def npv(base_cost: float, probabilities: np.ndarray) -> float:
            total = 0.0
            for b in range(buildout):  # build cohorts, equal weights
                for k, probability in enumerate(probabilities):
                    year = b + k
                    nominal = base_cost * anchor * (1.0 + inflation) ** year
                    total += (1.0 / buildout) * probability * nominal / (1.0 + rate) ** year
            return total

        row = _summary_row(outputs["summary"], state, "base", "physical")
        assert row["new_build_npv_per_dwelling"] == pytest.approx(npv(state_costs[state]["new_build"], np.array([1.0])), rel=1e-12)
        assert row["retrofit_npv_per_dwelling"] == pytest.approx(npv(state_costs[state]["retrofit"], increments), rel=1e-12)


def test_totals_row_aggregates_states(tmp_path: Path, config: dict) -> None:
    cdf = _synthetic_cdf(str(config["analysis"]["scenario"]), np.linspace(0.0, 0.8, HORIZON + 1))
    results_dir = _results_dir_with_cdf(tmp_path, cdf)
    outputs = run_cost_analysis(config, results_dir, DEFAULT_CONSTRUCTION_INDEX_CSV, dwelling_mix_csv=DEFAULT_DWELLING_MIX_CSV)
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
