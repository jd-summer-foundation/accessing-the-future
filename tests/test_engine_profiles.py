import au_housing_disability_monte_carlo as eng


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
