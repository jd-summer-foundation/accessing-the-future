"""Tests for first-occupancy timing tracking in run_sim."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import au_housing_disability_monte_carlo as eng
from scripts.retrofit_cost_comparison import expected_discounted_retrofit_cost


def _synthetic_inputs() -> tuple[eng.AllRates, eng.TenureDist, eng.InMoverDist]:
    rates = eng.AllRates(
        any_dis=eng.Rates({
            "15-24": 0.10, "25-34": 0.15, "35-44": 0.20, "45-54": 0.30,
            "55-64": 0.40, "65-74": 0.55, "75+": 0.70,
        }),
        motor_phys=eng.Rates({
            "15-24": 0.05, "25-34": 0.08, "35-44": 0.12, "45-54": 0.20,
            "55-64": 0.30, "65-74": 0.45, "75+": 0.60,
        }),
    )
    buckets = ["<1", "1-2", "2-3", "3-4", "5-9", "10-19", "20+"]
    uniform = [1.0 / len(buckets)] * len(buckets)
    tenure = eng.TenureDist(buckets=buckets, probs_by_bracket={b: list(uniform) for b in eng.BRACKETS})
    inmovers = eng.InMoverDist({b: 1.0 / len(eng.BRACKETS) for b in eng.BRACKETS})
    return rates, tenure, inmovers


@pytest.mark.parametrize("mode", eng.AGE_TRANSITION_MODES)
def test_first_occupancy_invariants(mode: str) -> None:
    rates, tenure, inmovers = _synthetic_inputs()
    params = eng.SimParams(n_props=2_000, horizon_years=20, seed=7, age_transition_mode=mode)
    result = eng.run_sim(params, rates, tenure, inmovers,
                         return_time_stats=True, return_first_occupancy=True)
    first = result["first_occupancy"]

    for category, time_key, moved_key, p_key in (
        ("any", "any_time", "any_moved_in", "p_ever_any"),
        ("physical", "phys_time", "phys_moved_in", "p_ever_physical"),
    ):
        times = first[time_key]
        occurred = np.isfinite(times)

        # A first-occupancy time exists exactly for dwellings that ever host.
        assert float(occurred.mean()) == pytest.approx(result[p_key]), category

        # Times fall inside the simulation horizon.
        assert np.all(times[occurred] >= 0.0), category
        assert np.all(times[occurred] < params.horizon_years), category

        # An occupancy recorded at build time can only arise from a household
        # moving in with the condition, never from in-place acquisition.
        at_build = occurred & (times == 0.0)
        assert np.all(first[moved_key][at_build]), category

    # Physical is a subtype of any: any-occupancy can never come later.
    phys_occurred = np.isfinite(first["phys_time"])
    assert np.all(first["any_time"][phys_occurred] <= first["phys_time"][phys_occurred])


def test_first_occupancy_tracking_does_not_perturb_results() -> None:
    rates, tenure, inmovers = _synthetic_inputs()
    params = eng.SimParams(n_props=1_000, horizon_years=20, seed=11)
    plain = eng.run_sim(params, rates, tenure, inmovers, return_time_stats=True)
    tracked = eng.run_sim(params, rates, tenure, inmovers,
                          return_time_stats=True, return_first_occupancy=True)
    for key in ("p_ever_any", "p_ever_physical", "pct_time_any", "pct_time_physical"):
        assert plain[key] == tracked[key]
    assert "first_occupancy" not in plain


def test_expected_discounted_retrofit_cost() -> None:
    import pandas as pd

    # 30% of dwellings need the features at year 0, another 20% by year 1.
    cdf = pd.Series([0.3, 0.5], index=[0, 1])

    undiscounted = expected_discounted_retrofit_cost(cdf, 19_000.0, 0.0)
    assert undiscounted == pytest.approx(0.5 * 19_000.0)

    discounted = expected_discounted_retrofit_cost(cdf, 19_000.0, 0.05)
    assert discounted == pytest.approx(0.3 * 19_000.0 + 0.2 * 19_000.0 / 1.05)
