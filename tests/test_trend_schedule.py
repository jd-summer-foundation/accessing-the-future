"""Tests for build_trend_schedule (step 1: only the 'none' trend)."""

from __future__ import annotations

import pytest

import au_housing_disability_monte_carlo as eng


BRACKETS = eng.BRACKETS


def _uniform_rates(value: float) -> dict[str, float]:
    return {b: value for b in BRACKETS}


def _make_rates(
    any_val: float = 0.2,
    motor_val: float = 0.08,
) -> eng.AllRates:
    return eng.AllRates(
        any_dis=eng.Rates(_uniform_rates(any_val)),
        motor_phys=eng.Rates(_uniform_rates(motor_val)),
    )


def test_none_reproduces_base_year_rates() -> None:
    rates = _make_rates(any_val=0.20, motor_val=0.08)
    schedule = eng.build_trend_schedule(rates, _uniform_rates(0.20), "none", horizon_years=20)
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.any_dis.by_bracket[b] == pytest.approx(0.20)
            assert snap.rates.motor_phys.by_bracket[b] == pytest.approx(0.08)


def test_schedule_shape() -> None:
    rates = _make_rates()
    for horizon in [5, 10, 20]:
        schedule = eng.build_trend_schedule(rates, _uniform_rates(0.20), "none", horizon_years=horizon)
        assert len(schedule) == horizon + 1
        assert [s.time_year for s in schedule] == pytest.approx(list(range(horizon + 1)))
        assert all(s.profiles is not None for s in schedule)


def test_shared_scaling_path_matches_unit_scale() -> None:
    # motor_phys that would exceed the any ceiling gets capped.
    rates = _make_rates(any_val=0.20, motor_val=0.25)
    schedule = eng.build_trend_schedule(rates, _uniform_rates(0.20), "none", horizon_years=3)
    t0 = schedule[0]
    for b in BRACKETS:
        any_rate = t0.rates.any_dis.by_bracket[b]
        assert t0.rates.motor_phys.by_bracket[b] <= any_rate + 1e-9


@pytest.mark.parametrize("trend", ["linear", "mid", "bogus"])
def test_unimplemented_trend_raises(trend: str) -> None:
    with pytest.raises(ValueError):
        eng.build_trend_schedule(_make_rates(), _uniform_rates(0.20), trend, horizon_years=5)


def test_base_year_any_rates_are_validated() -> None:
    invalid = _uniform_rates(0.20)
    invalid["75+"] = 1.2
    with pytest.raises(ValueError, match="fractions between 0 and 1"):
        eng.build_trend_schedule(_make_rates(), invalid, "none", horizon_years=5)

