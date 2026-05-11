"""Tests for build_survey_history_schedule and related helpers."""

from __future__ import annotations

import numpy as np
import pytest

import au_housing_disability_monte_carlo as eng


BRACKETS = eng.BRACKETS


def _uniform_rates(value: float) -> dict[str, float]:
    return {b: value for b in BRACKETS}


def _make_rates(
    any_val: float = 0.2,
    severe_val: float = 0.05,
    motor_val: float = 0.08,
    phys2_val: float = 0.15,
) -> eng.AllRates:
    return eng.AllRates(
        any_dis=eng.Rates(_uniform_rates(any_val)),
        severe_prof=eng.Rates(_uniform_rates(severe_val)),
        motor_phys=eng.Rates(_uniform_rates(motor_val)),
        phys2=eng.Rates(_uniform_rates(phys2_val)),
    )


def _survey_rates_any(years: list[int], values_by_year: dict[int, float]) -> dict[int, dict[str, float]]:
    return {year: _uniform_rates(values_by_year[year]) for year in years}


def _get_any(snapshot: eng.TransitionSnapshot) -> dict[str, float]:
    return snapshot.rates.any_dis.by_bracket


# ---- schedule structure ------------------------------------------------


def test_schedule_starts_at_t0() -> None:
    survey = _survey_rates_any([2003, 2009, 2022], {2003: 0.10, 2009: 0.12, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(_make_rates(), survey, horizon_years=20, start_year=2003)
    assert schedule[0].time_year == pytest.approx(0.0)


def test_schedule_creates_annual_snapshots() -> None:
    survey = _survey_rates_any([2003, 2009, 2022], {2003: 0.10, 2009: 0.12, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(_make_rates(), survey, horizon_years=20, start_year=2003)
    # Annual snapshots: t=0,1,...,20 → 21 entries
    assert len(schedule) == 21
    assert [s.time_year for s in schedule] == pytest.approx(list(range(21)))


def test_schedule_length_matches_horizon() -> None:
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    for horizon in [5, 10, 20]:
        schedule = eng.build_survey_history_schedule(_make_rates(), survey, horizon_years=horizon, start_year=2003)
        assert len(schedule) == horizon + 1


def test_interpolated_rate_midpoint_between_surveys() -> None:
    # 2003=0.10, 2009=0.16 → 2006 (midpoint) should give 0.13
    rates_2022 = _make_rates(any_val=0.20)
    survey = {2003: _uniform_rates(0.10), 2009: _uniform_rates(0.16), 2022: _uniform_rates(0.20)}
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    t3 = schedule[3]  # offset=3 → 2006
    for b in BRACKETS:
        assert _get_any(t3)[b] == pytest.approx(0.13, rel=1e-6)


# ---- rate values -------------------------------------------------------


def test_rates_at_t0_scaled_relative_to_2022() -> None:
    rates_2022 = _make_rates(any_val=0.20)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    # 2003 rate is half of 2022 rate → any_dis at t=0 should be 0.10
    for b in BRACKETS:
        assert _get_any(schedule[0])[b] == pytest.approx(0.10)


def test_rates_held_flat_after_last_survey_year() -> None:
    rates_2022 = _make_rates(any_val=0.20)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    # horizon=20, start=2003 → last snapshot is t=20 (year 2023), past the last survey year
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    last = schedule[-1]
    for b in BRACKETS:
        assert _get_any(last)[b] == pytest.approx(0.20)


def test_subtypes_capped_at_any_dis_after_scaling() -> None:
    # At 2003, any_dis is halved; severe_prof starts at 0.18 which would exceed scaled any_dis=0.10
    rates_2022 = _make_rates(any_val=0.20, severe_val=0.18, motor_val=0.15)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    t0 = schedule[0]
    for b in BRACKETS:
        any_rate = t0.rates.any_dis.by_bracket[b]
        sev_rate = t0.rates.severe_prof.by_bracket[b]
        motor_rate = t0.rates.motor_phys.by_bracket[b]
        assert sev_rate <= any_rate + 1e-9
        assert motor_rate <= any_rate + 1e-9


def test_phys2_not_capped_by_any_dis() -> None:
    rates_2022 = _make_rates(any_val=0.20, phys2_val=0.50)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    t0 = schedule[0]
    for b in BRACKETS:
        # phys2 is independent; scaled to 0.25, not capped at any_dis 0.10
        assert t0.rates.phys2.by_bracket[b] == pytest.approx(0.25)


def test_rates_clipped_to_1_when_scale_exceeds_1() -> None:
    rates_2022 = _make_rates(any_val=0.90)
    # 2003 rate 50% higher than 2022 → scaled would be 1.35, must be clipped at 1.0
    survey = _survey_rates_any([2003, 2022], {2003: 0.30, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    for b in BRACKETS:
        assert _get_any(schedule[0])[b] <= 1.0


# ---- prebuilt_schedule integration ------------------------------------


def _make_tenure() -> eng.TenureDist:
    return eng.TenureDist(
        buckets=["20+"],
        probs_by_bracket={bracket: [1.0] for bracket in BRACKETS},
    )


def test_run_sim_accepts_prebuilt_schedule() -> None:
    rates_2022 = _make_rates(any_val=0.20)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    schedule = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)

    summary = eng.run_sim(
        eng.SimParams(n_props=100, seed=42, horizon_years=20),
        rates_2022,
        _make_tenure(),
        eng.InMoverDist({b: 1.0 / len(BRACKETS) for b in BRACKETS}),
        prebuilt_schedule=schedule,
    )
    assert 0.0 <= summary["p_ever_any"] <= 1.0


def test_run_sim_prebuilt_overrides_transition_model() -> None:
    rates_2022 = _make_rates(any_val=0.20)
    survey = _survey_rates_any([2003, 2022], {2003: 0.10, 2022: 0.20})
    prebuilt = eng.build_survey_history_schedule(rates_2022, survey, horizon_years=20, start_year=2003)
    identity_tm = eng.transition_model_from_config({
        "interval_years": 1,
        "matrices": {name: "identity" for name in eng.TRANSITION_SERIES},
    })

    summary_with_prebuilt = eng.run_sim(
        eng.SimParams(n_props=200, seed=7, horizon_years=20),
        rates_2022,
        _make_tenure(),
        eng.InMoverDist({b: 1.0 / len(BRACKETS) for b in BRACKETS}),
        transition_model=identity_tm,
        prebuilt_schedule=prebuilt,
    )
    summary_identity = eng.run_sim(
        eng.SimParams(n_props=200, seed=7, horizon_years=20),
        rates_2022,
        _make_tenure(),
        eng.InMoverDist({b: 1.0 / len(BRACKETS) for b in BRACKETS}),
        transition_model=identity_tm,
    )
    # Starting rates differ (2003 < 2022), so results should differ
    assert summary_with_prebuilt["p_ever_any"] != summary_identity["p_ever_any"]
