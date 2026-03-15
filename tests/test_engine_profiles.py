import au_housing_disability_monte_carlo as eng


def test_make_profiles_enforces_monotone_any_and_clipped_conditionals() -> None:
    rates = eng.AllRates(
        any_dis=eng.Rates({"15-24": 0.10, "25-34": 0.08, "35-44": 0.14, "45-54": 0.20, "55-64": 0.21, "65-74": 0.30, "75+": 0.29}),
        severe_prof=eng.Rates({"15-24": 0.02, "25-34": 0.02, "35-44": 0.03, "45-54": 0.04, "55-64": 0.05, "65-74": 0.06, "75+": 0.06}),
        motor_phys=eng.Rates({"15-24": 0.01, "25-34": 0.02, "35-44": 0.20, "45-54": 0.18, "55-64": 0.19, "65-74": 0.20, "75+": 0.20}),
        phys2=eng.Rates({"15-24": 0.10, "25-34": 0.11, "35-44": 0.09, "45-54": 0.15, "55-64": 0.17, "65-74": 0.16, "75+": 0.18}),
    )

    profiles = eng.make_profiles(rates)
    adjusted_any = [profiles["adj_any"][bracket] for bracket in eng.BRACKETS]
    adjusted_phys2 = [profiles["adj_phys2"][bracket] for bracket in eng.BRACKETS]

    assert adjusted_any == sorted(adjusted_any)
    assert adjusted_phys2 == sorted(adjusted_phys2)
    assert all(0.0 <= profiles["cond_phys"][bracket] <= 1.0 for bracket in eng.BRACKETS)
