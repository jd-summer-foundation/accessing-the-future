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


# ---------------------------------------------------------------------------
# sdac_2003_2022_trend
# ---------------------------------------------------------------------------

def _make_historical_any(rate_2003: float, rate_2022: float) -> dict[int, dict[str, float]]:
    return {
        2003: _uniform_rates(rate_2003),
        2022: _uniform_rates(rate_2022),
    }


def test_sdac_trend_any_applies_linear_increment() -> None:
    """t=0 equals any_2022; t=1 equals any_2022 + (any_2022 - any_2003) / 19."""
    rates = _make_rates(any_val=0.20, motor_val=0.08)
    hist = _make_historical_any(rate_2003=0.10, rate_2022=0.20)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.20), "sdac_2003_2022_trend",
        horizon_years=5, historical_any=hist,
    )
    expected_increment = (0.20 - 0.10) / 19
    for b in BRACKETS:
        assert schedule[0].rates.any_dis.by_bracket[b] == pytest.approx(0.20)
        assert schedule[1].rates.any_dis.by_bracket[b] == pytest.approx(0.20 + expected_increment)
        assert schedule[5].rates.any_dis.by_bracket[b] == pytest.approx(0.20 + 5 * expected_increment)


def test_sdac_trend_motor_increments_proportionally() -> None:
    """motor_phys at t=1 equals motor_2022 + (motor_2022 * relative_annual_trend)."""
    any_2022 = 0.20
    any_2003 = 0.10
    motor_2022 = 0.08
    rates = _make_rates(any_val=any_2022, motor_val=motor_2022)
    hist = _make_historical_any(rate_2003=any_2003, rate_2022=any_2022)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(any_2022), "sdac_2003_2022_trend",
        horizon_years=5, historical_any=hist,
    )
    annual_increment_any = (any_2022 - any_2003) / 19
    relative = annual_increment_any / any_2022
    expected_motor_t1 = motor_2022 + motor_2022 * relative
    for b in BRACKETS:
        assert schedule[0].rates.motor_phys.by_bracket[b] == pytest.approx(motor_2022)
        assert schedule[1].rates.motor_phys.by_bracket[b] == pytest.approx(expected_motor_t1)


def test_sdac_trend_capped_at_unity() -> None:
    """Rates are clipped at 1.0 when the trend is strongly positive over many years."""
    rates = _make_rates(any_val=0.90, motor_val=0.50)
    hist = _make_historical_any(rate_2003=0.01, rate_2022=0.90)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.90), "sdac_2003_2022_trend",
        horizon_years=100, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.any_dis.by_bracket[b] <= 1.0 + 1e-9
            assert snap.rates.motor_phys.by_bracket[b] <= 1.0 + 1e-9


def test_sdac_trend_clipped_at_zero_for_negative_increments() -> None:
    """When any_2003 > any_2022 the annual increment is negative; rates must not go below 0.
    This is a real-world case — several age brackets show declining trends 2003–2022."""
    rates = _make_rates(any_val=0.05, motor_val=0.02)
    # 2003 rate higher → negative increment
    hist = _make_historical_any(rate_2003=0.30, rate_2022=0.05)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.05), "sdac_2003_2022_trend",
        horizon_years=50, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.any_dis.by_bracket[b] >= 0.0 - 1e-9
            assert snap.rates.motor_phys.by_bracket[b] >= 0.0 - 1e-9


def test_sdac_trend_motor_never_exceeds_any_dis() -> None:
    """motor_phys is capped at any_dis at every time step, including when any_dis is declining."""
    rates = _make_rates(any_val=0.20, motor_val=0.15)
    # any_dis declines; motor_phys starts just below any_dis
    hist = _make_historical_any(rate_2003=0.40, rate_2022=0.20)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.20), "sdac_2003_2022_trend",
        horizon_years=20, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.motor_phys.by_bracket[b] <= snap.rates.any_dis.by_bracket[b] + 1e-9


def test_sdac_trend_requires_historical_any() -> None:
    with pytest.raises(ValueError, match="historical_any"):
        eng.build_trend_schedule(
            _make_rates(), _uniform_rates(0.20), "sdac_2003_2022_trend",
            horizon_years=5,
        )


def test_sdac_trend_requires_year_2003() -> None:
    with pytest.raises(ValueError, match="2003"):
        eng.build_trend_schedule(
            _make_rates(), _uniform_rates(0.20), "sdac_2003_2022_trend",
            horizon_years=5,
            historical_any={2022: _uniform_rates(0.20)},
        )


# ---------------------------------------------------------------------------
# sdac_2015_2022_trend
# ---------------------------------------------------------------------------

def _make_historical_any_2015(rate_2015: float, rate_2022: float) -> dict[int, dict[str, float]]:
    return {
        2015: _uniform_rates(rate_2015),
        2022: _uniform_rates(rate_2022),
    }


def test_sdac_2015_trend_any_applies_linear_increment() -> None:
    """t=0 equals any_2022; t=1 equals any_2022 + (any_2022 - any_2015) / 7."""
    rates = _make_rates(any_val=0.20, motor_val=0.08)
    hist = _make_historical_any_2015(rate_2015=0.13, rate_2022=0.20)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.20), "sdac_2015_2022_trend",
        horizon_years=5, historical_any=hist,
    )
    expected_increment = (0.20 - 0.13) / 7
    for b in BRACKETS:
        assert schedule[0].rates.any_dis.by_bracket[b] == pytest.approx(0.20)
        assert schedule[1].rates.any_dis.by_bracket[b] == pytest.approx(0.20 + expected_increment)
        assert schedule[5].rates.any_dis.by_bracket[b] == pytest.approx(0.20 + 5 * expected_increment)


def test_sdac_2015_trend_motor_increments_proportionally() -> None:
    """motor_phys at t=1 equals motor_2022 + (motor_2022 * relative_annual_trend)."""
    any_2022 = 0.20
    any_2015 = 0.13
    motor_2022 = 0.08
    rates = _make_rates(any_val=any_2022, motor_val=motor_2022)
    hist = _make_historical_any_2015(rate_2015=any_2015, rate_2022=any_2022)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(any_2022), "sdac_2015_2022_trend",
        horizon_years=5, historical_any=hist,
    )
    annual_increment_any = (any_2022 - any_2015) / 7
    relative = annual_increment_any / any_2022
    expected_motor_t1 = motor_2022 + motor_2022 * relative
    for b in BRACKETS:
        assert schedule[0].rates.motor_phys.by_bracket[b] == pytest.approx(motor_2022)
        assert schedule[1].rates.motor_phys.by_bracket[b] == pytest.approx(expected_motor_t1)


def test_sdac_2015_trend_capped_at_unity() -> None:
    """Rates are clipped at 1.0 when the trend is strongly positive over many years."""
    rates = _make_rates(any_val=0.90, motor_val=0.50)
    hist = _make_historical_any_2015(rate_2015=0.01, rate_2022=0.90)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.90), "sdac_2015_2022_trend",
        horizon_years=100, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.any_dis.by_bracket[b] <= 1.0 + 1e-9
            assert snap.rates.motor_phys.by_bracket[b] <= 1.0 + 1e-9


def test_sdac_2015_trend_clipped_at_zero_for_negative_increments() -> None:
    """When any_2015 > any_2022 the annual increment is negative; rates must not go below 0."""
    rates = _make_rates(any_val=0.05, motor_val=0.02)
    hist = _make_historical_any_2015(rate_2015=0.30, rate_2022=0.05)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.05), "sdac_2015_2022_trend",
        horizon_years=50, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.any_dis.by_bracket[b] >= 0.0 - 1e-9
            assert snap.rates.motor_phys.by_bracket[b] >= 0.0 - 1e-9


def test_sdac_2015_trend_motor_never_exceeds_any_dis() -> None:
    """motor_phys is capped at any_dis at every time step, including when any_dis is declining."""
    rates = _make_rates(any_val=0.20, motor_val=0.15)
    hist = _make_historical_any_2015(rate_2015=0.40, rate_2022=0.20)
    schedule = eng.build_trend_schedule(
        rates, _uniform_rates(0.20), "sdac_2015_2022_trend",
        horizon_years=20, historical_any=hist,
    )
    for snap in schedule:
        for b in BRACKETS:
            assert snap.rates.motor_phys.by_bracket[b] <= snap.rates.any_dis.by_bracket[b] + 1e-9


def test_sdac_2015_trend_requires_historical_any() -> None:
    with pytest.raises(ValueError, match="historical_any"):
        eng.build_trend_schedule(
            _make_rates(), _uniform_rates(0.20), "sdac_2015_2022_trend",
            horizon_years=5,
        )


def test_sdac_2015_trend_requires_year_2015() -> None:
    with pytest.raises(ValueError, match="2015"):
        eng.build_trend_schedule(
            _make_rates(), _uniform_rates(0.20), "sdac_2015_2022_trend",
            horizon_years=5,
            historical_any={2022: _uniform_rates(0.20)},
        )

