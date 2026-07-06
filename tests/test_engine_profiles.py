import au_housing_disability_monte_carlo as eng
import pytest


def test_make_profiles_uses_any_rates_as_supplied_and_clips_conditionals() -> None:
    any_rates = {"15-24": 0.10, "25-34": 0.08, "35-44": 0.14, "45-54": 0.20, "55-64": 0.21, "65-74": 0.30, "75+": 0.29}
    rates = eng.AllRates(
        any_dis=eng.Rates(any_rates),
        motor_phys=eng.Rates({"15-24": 0.01, "25-34": 0.02, "35-44": 0.20, "45-54": 0.18, "55-64": 0.19, "65-74": 0.20, "75+": 0.20}),
    )

    profiles = eng.make_profiles(rates)

    # ANY prevalence is used exactly as supplied (no monotone adjustment), so the
    # non-decreasing constraint is no longer imposed (25-34 stays below 15-24).
    assert all(profiles.adj_any[bracket] == any_rates[bracket] for bracket in eng.BRACKETS)
    assert profiles.adj_any["25-34"] < profiles.adj_any["15-24"]

    # Conditional subtype probabilities remain clipped to [0, 1].
    assert all(0.0 <= profiles.cond_phys[bracket] <= 1.0 for bracket in eng.BRACKETS)

    # A declining ANY step yields zero acquisition probability (no recovery).
    assert profiles.acq_any[("15-24", "25-34")] == 0.0


def test_interpolate_bracket_rate_varies_linearly_by_age() -> None:
    by_bracket = {
        "15-24": 0.10,
        "25-34": 0.20,
        "35-44": 0.30,
        "45-54": 0.40,
        "55-64": 0.50,
        "65-74": 0.60,
        "75+": 0.70,
    }

    assert eng._interpolate_bracket_rate(by_bracket, 15) == pytest.approx(0.10)
    assert eng._interpolate_bracket_rate(by_bracket, 19) == pytest.approx(0.10)
    assert eng._interpolate_bracket_rate(by_bracket, 20) == pytest.approx(0.105)
    assert eng._interpolate_bracket_rate(by_bracket, 24) == pytest.approx(0.145)
    assert eng._interpolate_bracket_rate(by_bracket, 25) == pytest.approx(0.155)
    assert eng._interpolate_bracket_rate(by_bracket, 29) == pytest.approx(0.195)
    assert eng._interpolate_bracket_rate(by_bracket, 30) == pytest.approx(0.205)
    assert eng._interpolate_bracket_rate(by_bracket, 90) == pytest.approx(0.70)


def test_annual_age_transition_uses_conditional_yearly_acquisition() -> None:
    rates = eng.AllRates(
        any_dis=eng.Rates({
            "15-24": 0.10,
            "25-34": 0.20,
            "35-44": 0.30,
            "45-54": 0.40,
            "55-64": 0.50,
            "65-74": 0.60,
            "75+": 0.70,
        }),
        motor_phys=eng.Rates({
            "15-24": 0.05,
            "25-34": 0.10,
            "35-44": 0.15,
            "45-54": 0.20,
            "55-64": 0.25,
            "65-74": 0.30,
            "75+": 0.35,
        }),
    )
    profiles = eng.make_profiles(rates)

    class StubRng:
        def __init__(self, draws: list[float]) -> None:
            self._draws = iter(draws)

        def random(self) -> float:
            return next(self._draws)

    any_d, phys_d = eng._apply_annual_age_transition(
        20,
        profiles,
        StubRng([0.01, 0.01]),
        any_d=False,
        phys_d=False,
    )

    assert any_d is True
    assert phys_d is True
