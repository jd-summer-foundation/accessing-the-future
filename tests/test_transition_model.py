import numpy as np
import pytest

import au_housing_disability_monte_carlo as eng


def _identity_transition_config() -> dict[str, object]:
    return {
        "interval_years": 1,
        "matrices": {name: "identity" for name in eng.TRANSITION_SERIES},
    }


def _make_rates(
    any_rates: dict[str, float],
    severe_rates: dict[str, float] | None = None,
    motor_rates: dict[str, float] | None = None,
    phys2_rates: dict[str, float] | None = None,
) -> eng.AllRates:
    zeros = {bracket: 0.0 for bracket in eng.BRACKETS}
    return eng.AllRates(
        any_dis=eng.Rates(any_rates),
        severe_prof=eng.Rates(severe_rates or zeros),
        motor_phys=eng.Rates(motor_rates or zeros),
        phys2=eng.Rates(phys2_rates or zeros),
    )


def _make_tenure() -> eng.TenureDist:
    return eng.TenureDist(
        buckets=["20+"],
        probs_by_bracket={bracket: [1.0] for bracket in eng.BRACKETS},
    )


class FakeRNG:
    def __init__(self, *, age: int, uniform_value: float = 25.0) -> None:
        self.age = age
        self.uniform_value = uniform_value

    def random(self) -> float:
        return 0.0

    def integers(self, low: int, high: int | None = None, size: object = None) -> int:
        return self.age

    def uniform(self, low: float = 0.0, high: float | None = None, size: object = None) -> float:
        return self.uniform_value


def test_transition_model_identity_config_expands_to_identity_matrices() -> None:
    transition_model = eng.transition_model_from_config(_identity_transition_config())

    assert transition_model is not None
    assert transition_model.interval_years == 1
    for name in eng.TRANSITION_SERIES:
        np.testing.assert_array_equal(transition_model.matrices[name], np.eye(len(eng.BRACKETS)))


def test_transition_model_rejects_invalid_matrix_shape() -> None:
    config = _identity_transition_config()
    config["matrices"] = dict(config["matrices"])
    config["matrices"]["any_dis"] = [[1.0, 0.0]]

    with pytest.raises(ValueError, match="any_dis transition matrix must have shape"):
        eng.transition_model_from_config(config)


def test_build_transition_schedule_identity_preserves_rates() -> None:
    base_rates = _make_rates(
        {
            "15-24": 0.1,
            "25-34": 0.2,
            "35-44": 0.3,
            "45-54": 0.4,
            "55-64": 0.5,
            "65-74": 0.6,
            "75+": 0.7,
        }
    )
    transition_model = eng.transition_model_from_config(_identity_transition_config())

    schedule = eng.build_transition_schedule(base_rates, horizon_years=3, transition_model=transition_model)

    assert [snapshot.time_year for snapshot in schedule] == [0.0, 1.0, 2.0, 3.0]
    expected = np.asarray([base_rates.any_dis.by_bracket[bracket] for bracket in eng.BRACKETS], dtype=float)
    for snapshot in schedule:
        actual = np.asarray([snapshot.rates.any_dis.by_bracket[bracket] for bracket in eng.BRACKETS], dtype=float)
        np.testing.assert_allclose(actual, expected)


def test_build_transition_schedule_applies_explicit_matrix() -> None:
    base_rates = _make_rates(
        {
            "15-24": 0.2,
            "25-34": 0.4,
            "35-44": 0.6,
            "45-54": 0.8,
            "55-64": 1.0,
            "65-74": 0.5,
            "75+": 0.3,
        }
    )
    half_identity = (0.5 * np.eye(len(eng.BRACKETS))).tolist()
    config = _identity_transition_config()
    config["matrices"] = dict(config["matrices"])
    config["matrices"]["any_dis"] = half_identity
    transition_model = eng.transition_model_from_config(config)

    schedule = eng.build_transition_schedule(base_rates, horizon_years=1, transition_model=transition_model)

    next_values = np.asarray([schedule[1].rates.any_dis.by_bracket[bracket] for bracket in eng.BRACKETS], dtype=float)
    np.testing.assert_allclose(next_values, 0.5 * np.asarray([0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.3]))


def test_run_sim_can_acquire_any_from_time_step_without_ageing(monkeypatch: pytest.MonkeyPatch) -> None:
    base_rates = _make_rates(
        {
            "15-24": 0.0,
            "25-34": 0.0,
            "35-44": 1.0,
            "45-54": 1.0,
            "55-64": 1.0,
            "65-74": 1.0,
            "75+": 1.0,
        }
    )
    matrix = np.eye(len(eng.BRACKETS))
    matrix[0, :] = 0.0
    matrix[0, 2] = 1.0
    config = _identity_transition_config()
    config["matrices"] = dict(config["matrices"])
    config["matrices"]["any_dis"] = matrix.tolist()
    transition_model = eng.transition_model_from_config(config)

    monkeypatch.setattr(eng.np.random, "default_rng", lambda seed: FakeRNG(age=15))

    summary = eng.run_sim(
        eng.SimParams(n_props=1, seed=1, horizon_years=2),
        base_rates,
        _make_tenure(),
        eng.InMoverDist({"15-24": 1.0, "25-34": 0.0, "35-44": 0.0, "45-54": 0.0, "55-64": 0.0, "65-74": 0.0, "75+": 0.0}),
        transition_model=transition_model,
    )

    assert summary["p_ever_any"] == pytest.approx(1.0)


def test_run_sim_processes_time_transition_before_age_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    any_rates = {bracket: 1.0 for bracket in eng.BRACKETS}
    severe_rates = {
        "15-24": 0.0,
        "25-34": 0.0,
        "35-44": 1.0,
        "45-54": 0.0,
        "55-64": 0.0,
        "65-74": 0.0,
        "75+": 0.0,
    }
    base_rates = _make_rates(any_rates, severe_rates=severe_rates)
    matrix = np.eye(len(eng.BRACKETS))
    matrix[0, :] = 0.0
    matrix[0, 2] = 1.0
    matrix[1, :] = 0.0
    config = _identity_transition_config()
    config["matrices"] = dict(config["matrices"])
    config["matrices"]["severe_prof"] = matrix.tolist()
    transition_model = eng.transition_model_from_config(config)

    monkeypatch.setattr(eng.np.random, "default_rng", lambda seed: FakeRNG(age=24))

    summary = eng.run_sim(
        eng.SimParams(n_props=1, seed=1, horizon_years=2),
        base_rates,
        _make_tenure(),
        eng.InMoverDist({"15-24": 1.0, "25-34": 0.0, "35-44": 0.0, "45-54": 0.0, "55-64": 0.0, "65-74": 0.0, "75+": 0.0}),
        transition_model=transition_model,
    )

    assert summary["p_ever_any"] == pytest.approx(1.0)
    assert summary["p_ever_severe"] == pytest.approx(1.0)
