import au_housing_disability_monte_carlo as eng
import pytest

from scripts.pipeline_utils import TENURE_BUCKETS


def _sim_inputs():
    any_rates = {"15-24": 0.10, "25-34": 0.12, "35-44": 0.15, "45-54": 0.20, "55-64": 0.28, "65-74": 0.38, "75+": 0.55}
    phys_rates = {"15-24": 0.03, "25-34": 0.04, "35-44": 0.06, "45-54": 0.10, "55-64": 0.16, "65-74": 0.25, "75+": 0.40}
    rates = eng.AllRates(any_dis=eng.Rates(any_rates), motor_phys=eng.Rates(phys_rates))
    tenure = eng.TenureDist(
        buckets=TENURE_BUCKETS,
        probs_by_bracket={bracket: [1.0 / len(TENURE_BUCKETS)] * len(TENURE_BUCKETS) for bracket in eng.BRACKETS},
    )
    inmovers = eng.InMoverDist({bracket: 1.0 / len(eng.BRACKETS) for bracket in eng.BRACKETS})
    return rates, tenure, inmovers


def test_first_occupancy_cdf_shape_and_invariants() -> None:
    rates, tenure, inmovers = _sim_inputs()
    params = eng.SimParams(n_props=800, horizon_years=12, seed=7)
    results = eng.run_sim(params, rates, tenure, inmovers, return_first_occupancy=True)

    for category in ("any", "physical"):
        cdf = results[f"first_occupancy_cdf_{category}"]
        assert len(cdf) == params.horizon_years + 1
        assert all(0.0 <= value <= 1.0 for value in cdf)
        # Monotone non-decreasing: first-occupancy is a cumulative event share.
        assert all(later >= earlier for earlier, later in zip(cdf, cdf[1:]))
        # A positive share of first households already has the category at move-in.
        assert cdf[0] > 0.0
        # By construction the CDF at the horizon equals the whole-horizon probability.
        assert cdf[-1] == pytest.approx(results[f"p_ever_{category}"], abs=0.0)

    # Physical is a subtype of any, so its CDF can never exceed the any CDF.
    paired = zip(results["first_occupancy_cdf_physical"], results["first_occupancy_cdf_any"])
    assert all(phys <= any_ for phys, any_ in paired)


def test_first_occupancy_tracking_does_not_disturb_other_outputs() -> None:
    rates, tenure, inmovers = _sim_inputs()
    params = eng.SimParams(n_props=400, horizon_years=10, seed=11)

    without = eng.run_sim(params, rates, tenure, inmovers, return_time_stats=True)
    with_cdf = eng.run_sim(params, rates, tenure, inmovers, return_time_stats=True, return_first_occupancy=True)

    for key, value in without.items():
        assert with_cdf[key] == value
