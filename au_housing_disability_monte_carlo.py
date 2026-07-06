#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo: lifetime probability a dwelling ever hosts a household with disability/condition (Australia)

We simulate a dwelling being occupied by a sequence of households over a horizon.
Households:
- enter with an initial age bracket (from an in-mover age distribution by default)
- stay for a tenure duration drawn from an age-specific distribution
- may age into older brackets during their tenure
- can acquire conditions only at age-bracket boundaries
- once acquired, a condition persists for that household (no recovery modelled)

Categories tracked (household-level):
1) any_dis: any disability (base category; prevalence by age bracket, used as supplied)
2) motor_phys: mobility/physical disability proxy, modelled as a SUBTYPE of any_dis (conditional on any_dis)

Outputs (summary dict):
- p_ever_any, p_ever_physical
- optionally: mean % of time a dwelling is occupied by a household with each category
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import time

BRACKETS: List[str] = ["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
BRACKET_IDX = {b: i for i, b in enumerate(BRACKETS)}
AGE_TRANSITION_MODES: Tuple[str, ...] = ("bracket_boundary", "annual_interpolated")
BRACKET_START_AGES: Dict[str, int] = {
    "15-24": 15,
    "25-34": 25,
    "35-44": 35,
    "45-54": 45,
    "55-64": 55,
    "65-74": 65,
    "75+": 75,
}
BRACKET_MIDPOINT_AGES: Dict[str, float] = {
    "15-24": 19.5,
    "25-34": 29.5,
    "35-44": 39.5,
    "45-54": 49.5,
    "55-64": 59.5,
    "65-74": 69.5,
    "75+": 79.5,
}

# Forward-projection trends off the base year.
# "none": hold base-year rates flat for the whole horizon.
# "sdac_2003_2022_trend": project both any_dis and motor_phys forward using the
#   linear 2003–2022 trend from SDACDC01.xlsx Table 1.3 (any disability), with
#   motor_phys using an inferred increment scaled to its 2022 baseline.
# "sdac_2015_2022_trend": same method as above but uses 2015–2022 (7 years)
#   as the historical window instead of 2003–2022 (19 years).
TREND_TYPES: Tuple[str, ...] = ("none", "sdac_2003_2022_trend", "sdac_2015_2022_trend")

DEFAULT_HORIZON_YEARS = 20


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Rates:
    """Age-bracketed prevalence rates for one category (share of all households)."""
    by_bracket: Dict[str, float]


@dataclass(frozen=True)
class AllRates:
    """
    Age-bracketed prevalence rates (share of households) for model categories.

    Interpretation used in the simulation:
    - any_dis is the base disability category.
    - motor_phys is a subtype of any_dis (motor_phys ⊆ any_dis).
      Implementation: we convert total motor_phys rates into conditional P(motor_phys | any_dis)
      using the adjusted any_dis profile.
    """
    any_dis: Rates
    motor_phys: Rates


@dataclass(frozen=True)
class TenureDist:
    """
    Tenure distribution buckets (strings) + probability list per age bracket.
    Example buckets: ["<1","1-2","2-3","3-4","5-9","10-19","20+"]
    """
    buckets: List[str]
    probs_by_bracket: Dict[str, List[float]]


@dataclass(frozen=True)
class InMoverDist:
    """Probability distribution over age brackets for new movers."""
    probs_by_bracket: Dict[str, float]


@dataclass(frozen=True)
class SimParams:
    n_props: int = 50_000
    horizon_years: int = DEFAULT_HORIZON_YEARS
    seed: int = 42

    # If household currently has ANY/PHYSICAL, lengthen tenure by this factor (e.g. 1.2)
    disabled_tenure_factor: float = 1.0
    age_transition_mode: str = "bracket_boundary"


@dataclass(frozen=True)
class Profiles:
    """Prevalence profiles and acquisition probabilities for a rate set.

    - adj_any[b]: ANY prevalence per bracket, used as supplied (not forced monotone)
    - cond_phys[b]: P(motor_phys | any_dis) per bracket
    - acq_any[(b0, b1)]: P(acquire ANY) at the b0 -> b1 age boundary
    - acq_cond_phys[(b0, b1)]: P(acquire subtype | already ANY) at that boundary
    """

    adj_any: Dict[str, float]
    cond_phys: Dict[str, float]
    acq_any: Dict[Tuple[str, str], float]
    acq_cond_phys: Dict[Tuple[str, str], float]


@dataclass(frozen=True)
class TransitionSnapshot:
    """Precomputed rate/profile state for a calendar-time update boundary."""

    time_year: float
    rates: AllRates
    profiles: Profiles


# -------------------------------------------------------------------
# Utilities: validation, profiles, sampling
# -------------------------------------------------------------------

def _validate_bracketed_dict(name: str, d: Dict[str, float]) -> None:
    missing = [b for b in BRACKETS if b not in d]
    if missing:
        raise ValueError(f"{name}: missing brackets: {missing}")


def _validate_probs(name: str, probs: List[float], tol: float = 1e-6) -> List[float]:
    arr = np.asarray(probs, dtype=float)

    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name}: contains NaN/inf values: {probs}")

    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if s <= 0.0:
        raise ValueError(f"{name}: sum of probabilities is zero after clipping: {probs}")

    arr = arr / s
    if not (abs(float(arr.sum()) - 1.0) <= tol):
        # normalization should make it 1.0; this is just belt + braces
        raise ValueError(f"{name}: failed to normalise to 1.0 (sum={arr.sum():.8f})")

    return arr.tolist()


def _acquire_prob(prev_rate: float, next_rate: float) -> float:
    """Persistence-model acquisition probability for a prev->next prevalence step:
    (next - prev) / (1 - prev) when rising, else 0."""
    prev = float(prev_rate)
    nxt = float(next_rate)
    if nxt <= prev or prev >= 1.0:
        return 0.0
    return (nxt - prev) / (1.0 - prev)


def _acquire_probs_from_adjusted(adjusted_rates: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    """
    Given a prevalence profile across age brackets, compute per-transition
    acquisition probabilities under persistence by applying _acquire_prob to each
    b0 -> b1 bracket pair. Where prevalence declines between brackets the
    acquisition probability is zero (conditions are never lost).
    """
    _validate_bracketed_dict("adjusted_rates", adjusted_rates)
    return {
        (BRACKETS[i], BRACKETS[i + 1]): _acquire_prob(
            adjusted_rates[BRACKETS[i]], adjusted_rates[BRACKETS[i + 1]]
        )
        for i in range(len(BRACKETS) - 1)
    }


def _interpolate_bracket_rate(by_bracket: Dict[str, float], age: int) -> float:
    """Linearly interpolate the target prevalence between bracket midpoint ages."""
    _validate_bracketed_dict("by_bracket", by_bracket)
    age_f = float(age)
    if age_f <= BRACKET_MIDPOINT_AGES[BRACKETS[0]]:
        return float(by_bracket[BRACKETS[0]])
    if age_f >= BRACKET_MIDPOINT_AGES["75+"]:
        return float(by_bracket["75+"])

    for idx, bracket in enumerate(BRACKETS[:-1]):
        start_age = BRACKET_MIDPOINT_AGES[bracket]
        next_bracket = BRACKETS[idx + 1]
        next_start_age = BRACKET_MIDPOINT_AGES[next_bracket]
        if start_age <= age_f < next_start_age:
            span = float(next_start_age - start_age)
            frac = float(age_f - start_age) / span
            start_rate = float(by_bracket[bracket])
            end_rate = float(by_bracket[next_bracket])
            return start_rate + frac * (end_rate - start_rate)

    return float(by_bracket["75+"])


def _bracket_for_age(age: int) -> str:
    if age >= BRACKET_START_AGES["75+"]:
        return "75+"
    for idx, bracket in enumerate(BRACKETS[:-1]):
        start_age = BRACKET_START_AGES[bracket]
        next_start_age = BRACKET_START_AGES[BRACKETS[idx + 1]]
        if start_age <= age < next_start_age:
            return bracket
    return BRACKETS[0]


def make_profiles(all_rates: AllRates) -> Profiles:
    """
    Build prevalence profiles and acquisition probabilities for:

      - ANY disability (base): used as supplied, without forcing monotonicity.
        Where prevalence happens to fall between brackets (e.g. 15-24 -> 25-34
        at the household level), the corresponding acquisition probability is
        zero rather than negative, consistent with the no-recovery assumption.
      - motor_phys: conditional on ANY (subtype)

    See the Profiles dataclass for the returned fields.
    """
    _validate_bracketed_dict("any_dis", all_rates.any_dis.by_bracket)
    _validate_bracketed_dict("motor_phys", all_rates.motor_phys.by_bracket)

    # ANY: use the supplied per-bracket prevalence directly (no monotone adjustment).
    adj_any = {b: float(all_rates.any_dis.by_bracket[b]) for b in BRACKETS}

    # Helper to build conditional P(subtype | any) from subtype total + ANY totals
    def build_cond_subtype(total_by_bracket: Dict[str, float]) -> Dict[str, float]:
        cond: Dict[str, float] = {}
        for b in BRACKETS:
            ra = float(adj_any[b])                  # ANY total (as supplied)
            rt = float(total_by_bracket[b])         # subtype total
            if ra <= 0.0:
                # if ANY is 0 but subtype > 0, clamp to 1 (still yields 0 total in simulation unless ANY occurs)
                cond[b] = 0.0 if rt <= 0.0 else 1.0
            else:
                cond[b] = float(np.clip(rt / ra, 0.0, 1.0))
        return cond

    cond_phys = build_cond_subtype(all_rates.motor_phys.by_bracket)

    # Acquisition probs. The conditional subtype reuses the same persistence formula
    # applied to P(subtype | any): when ANY is acquired at a boundary the subtype is
    # seeded from cond[next]; when ANY is already present it can be acquired with
    # probability _acquire_prob(cond[curr], cond[next]).
    acq_any = _acquire_probs_from_adjusted(adj_any)
    acq_cond_phys = _acquire_probs_from_adjusted(cond_phys)

    return Profiles(
        adj_any=adj_any,
        cond_phys=cond_phys,
        acq_any=acq_any,
        acq_cond_phys=acq_cond_phys,
    )


def _scale_all_rates(rates_2022: AllRates, scale: Dict[str, float]) -> AllRates:
    """Scale all sub-category rates proportionally to per-bracket scale factors."""
    def _apply(rates: Rates, cap: Optional[Dict[str, float]] = None) -> Rates:
        scaled = {b: float(np.clip(rates.by_bracket[b] * scale.get(b, 1.0), 0.0, 1.0)) for b in BRACKETS}
        if cap is not None:
            scaled = {b: min(scaled[b], cap[b]) for b in BRACKETS}
        return Rates(scaled)

    any_scaled = _apply(rates_2022.any_dis)
    any_cap = any_scaled.by_bracket
    return AllRates(
        any_dis=any_scaled,
        motor_phys=_apply(rates_2022.motor_phys, any_cap),
    )


# Linear-trend names mapped to (prior survey year, historical window length in years).
_LINEAR_TREND_WINDOWS: Dict[str, Tuple[int, int]] = {
    "sdac_2003_2022_trend": (2003, 19),
    "sdac_2015_2022_trend": (2015, 7),
}


def _build_linear_trend_schedule(
    rates_2022: AllRates,
    any_2022: Dict[str, float],
    historical_any: Optional[Dict[int, Dict[str, float]]],
    trend: str,
    prior_year: int,
    window_years: int,
    horizon_years: int,
    *,
    verbose: bool = False,
) -> List[TransitionSnapshot]:
    """Project rates forward by a linear person-level trend over a historical window.

    Shared implementation for the sdac_*_trend variants, which differ only in the
    prior survey year and window length. Both any_dis and motor_phys start from the
    scenario-scaled household-level rates_2022 and apply per-year increments scaled
    via the person-level relative change between prior_year and 2022.
    """
    if historical_any is None or prior_year not in historical_any:
        raise ValueError(
            f"historical_any must be provided and include year {prior_year} "
            f"for the {trend!r} trend"
        )

    any_prior = historical_any[prior_year]
    _validate_bracketed_dict(f"historical_any[{prior_year}]", any_prior)

    # Annual percentage-point increment for any_dis (fractions, not percentages).
    # Negative when the historical trend is declining for a bracket.
    inc_any: Dict[str, float] = {
        b: (float(any_2022[b]) - float(any_prior[b])) / window_years
        for b in BRACKETS
    }

    any_2022_hh = rates_2022.any_dis.by_bracket
    motor_2022 = rates_2022.motor_phys.by_bracket
    inc_any_hh: Dict[str, float] = {}
    inc_motor: Dict[str, float] = {}
    for b in BRACKETS:
        if float(any_2022[b]) > 0.0:
            relative_annual_trend = float(inc_any[b]) / float(any_2022[b])
            inc_any_hh[b] = float(any_2022_hh[b]) * relative_annual_trend
            inc_motor[b] = float(motor_2022[b]) * relative_annual_trend
        else:
            inc_any_hh[b] = 0.0
            inc_motor[b] = 0.0

    if verbose:
        print(f"Annual increments for {trend!r}:")
        for b in BRACKETS:
            print(
                f"  [{b}]  any: {prior_year}={float(any_prior[b])*100:.3f}%  "
                f"2022(person)={float(any_2022[b])*100:.3f}%  "
                f"2022(hh)={float(any_2022_hh[b])*100:.3f}%  "
                f"inc={inc_any_hh[b]*100:+.4f}%/yr  |  "
                f"motor/phys: 2022={float(motor_2022[b])*100:.3f}%  "
                f"inc={inc_motor[b]*100:+.4f}%/yr"
            )

    snapshots: List[TransitionSnapshot] = []
    for offset in range(int(horizon_years) + 1):
        any_at_t = {
            b: float(np.clip(float(any_2022_hh[b]) + inc_any_hh[b] * offset, 0.0, 1.0))
            for b in BRACKETS
        }
        motor_at_t = {
            b: float(np.clip(
                min(float(motor_2022[b]) + inc_motor[b] * offset, any_at_t[b]),
                0.0, 1.0,
            ))
            for b in BRACKETS
        }
        rates_at_t = AllRates(any_dis=Rates(any_at_t), motor_phys=Rates(motor_at_t))
        snapshots.append(TransitionSnapshot(float(offset), rates_at_t, make_profiles(rates_at_t)))
    return snapshots


def build_trend_schedule(
    rates_2022: AllRates,
    any_2022: Dict[str, float],
    trend: str,
    horizon_years: int,
    *,
    historical_any: Optional[Dict[int, Dict[str, float]]] = None,
    verbose: bool = False,
) -> List[TransitionSnapshot]:
    """
    Build an annual transition schedule projecting rates forward from the base year.

    Args:
        rates_2022: Model rates for the base year (with scenario scaling applied).
        any_2022: Person-level any-disability rate per bracket (fractions 0–1),
            from SDACDC01.xlsx Table 1.3. Used as the denominator for computing
            relative annual changes; NOT used as the simulation starting point
            (which is always the household-level rates_2022).
        trend: One of TREND_TYPES.
        horizon_years: Simulation length in years.
        historical_any: Dict keyed by survey year (e.g. {2003: {...}, 2015: {...}, 2022: {...}})
            containing SDACDC01 any-disability rates (fractions 0–1) per bracket.
            Required for "sdac_2003_2022_trend" (needs year 2003) and
            "sdac_2015_2022_trend" (needs year 2015).
    """
    _validate_bracketed_dict("any_2022", any_2022)
    any_values = np.asarray([float(any_2022[b]) for b in BRACKETS], dtype=float)
    if np.any(~np.isfinite(any_values)):
        raise ValueError("any_2022 contains NaN/inf values")
    if np.any((any_values < 0.0) | (any_values > 1.0)):
        raise ValueError("any_2022 rates must be fractions between 0 and 1")

    if trend not in TREND_TYPES:
        raise ValueError(
            f"Unknown or unimplemented trend {trend!r}; supported: {TREND_TYPES}"
        )

    if trend == "none":
        scales = {b: 1.0 for b in BRACKETS}
        snapshots: List[TransitionSnapshot] = []
        for offset in range(int(horizon_years) + 1):
            rates_at_t = _scale_all_rates(rates_2022, scales)
            snapshots.append(TransitionSnapshot(float(offset), rates_at_t, make_profiles(rates_at_t)))
        return snapshots

    prior_year, window_years = _LINEAR_TREND_WINDOWS[trend]
    return _build_linear_trend_schedule(
        rates_2022,
        any_2022,
        historical_any,
        trend,
        prior_year,
        window_years,
        int(horizon_years),
        verbose=verbose,
    )


def _event_occurs(probability: float, rng: np.random.Generator) -> bool:
    prob = float(probability)
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    return bool(rng.random() < prob)


def _seed_household_state(
    bracket: str,
    profiles: Profiles,
    rng: np.random.Generator,
) -> Tuple[bool, bool]:
    any_d = rng.random() < profiles.adj_any[bracket]
    phys_d = rng.random() < profiles.cond_phys[bracket] if any_d else False
    return any_d, phys_d


def _seed_household_state_at_age(
    age: int,
    profiles: Profiles,
    rng: np.random.Generator,
) -> Tuple[bool, bool]:
    p_any = _interpolate_bracket_rate(profiles.adj_any, age)
    p_phys_cond = _interpolate_bracket_rate(profiles.cond_phys, age)
    any_d = _event_occurs(p_any, rng)
    phys_d = _event_occurs(p_phys_cond, rng) if any_d else False
    return any_d, phys_d


def _apply_time_step_transition(
    bracket: str,
    prev_profiles: Profiles,
    next_profiles: Profiles,
    rng: np.random.Generator,
    any_d: bool,
    phys_d: bool,
) -> Tuple[bool, bool]:
    became_any_now = False

    if not any_d:
        p_any = _acquire_prob(prev_profiles.adj_any[bracket], next_profiles.adj_any[bracket])
        if _event_occurs(p_any, rng):
            any_d = True
            became_any_now = True

    if any_d:
        if became_any_now:
            if not phys_d and _event_occurs(next_profiles.cond_phys[bracket], rng):
                phys_d = True
        else:
            if not phys_d:
                p_phys = _acquire_prob(
                    prev_profiles.cond_phys[bracket],
                    next_profiles.cond_phys[bracket],
                )
                if _event_occurs(p_phys, rng):
                    phys_d = True

    return any_d, phys_d


def _apply_age_boundary_transition(
    bracket: str,
    next_bracket: str,
    profiles: Profiles,
    rng: np.random.Generator,
    any_d: bool,
    phys_d: bool,
) -> Tuple[bool, bool]:
    became_any_now = False

    if not any_d:
        p_any = profiles.acq_any.get((bracket, next_bracket), 0.0)
        if rng.random() < p_any:
            any_d = True
            became_any_now = True

    if any_d:
        if became_any_now:
            if not phys_d and rng.random() < profiles.cond_phys[next_bracket]:
                phys_d = True
        else:
            if not phys_d:
                p_phys = profiles.acq_cond_phys.get((bracket, next_bracket), 0.0)
                if rng.random() < p_phys:
                    phys_d = True

    return any_d, phys_d


def _apply_annual_age_transition(
    age: int,
    profiles: Profiles,
    rng: np.random.Generator,
    any_d: bool,
    phys_d: bool,
) -> Tuple[bool, bool]:
    became_any_now = False
    prev_any = _interpolate_bracket_rate(profiles.adj_any, age)
    next_any = _interpolate_bracket_rate(profiles.adj_any, age + 1)

    if not any_d and _event_occurs(_acquire_prob(prev_any, next_any), rng):
        any_d = True
        became_any_now = True

    if any_d:
        next_cond_phys = _interpolate_bracket_rate(profiles.cond_phys, age + 1)
        if became_any_now:
            if not phys_d and _event_occurs(next_cond_phys, rng):
                phys_d = True
        elif not phys_d:
            prev_cond_phys = _interpolate_bracket_rate(profiles.cond_phys, age)
            if _event_occurs(_acquire_prob(prev_cond_phys, next_cond_phys), rng):
                phys_d = True

    return any_d, phys_d


def _prepare_cdf(name: str, probs: List[float]) -> np.ndarray:
    """Validate and compute CDF once for a probability vector."""
    p = _validate_probs(name, probs)
    return np.cumsum(p)


def _sample_from_cdf(cdf: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an index using a pre-computed CDF (no validation)."""
    return int(np.searchsorted(cdf, rng.random(), side="right"))


def sample_tenure_years(bucket: str, rng: np.random.Generator) -> float:
    """
    Convert a tenure bucket label into a sampled tenure duration in years.
    Bucket format examples:
      "<1", "1-2", "2-3", "3-4", "5-9", "10-19", "20+"
    """
    b = bucket.strip()
    if b.startswith("<"):
        hi = float(b[1:])
        return float(rng.uniform(0.0, hi))
    if b.endswith("+"):
        lo = float(b[:-1])
        # Simple tail: uniform over [lo, lo+10]. Replace if you later adopt a better tail model.
        return float(rng.uniform(lo, lo + 10.0))
    if "-" in b:
        lo_s, hi_s = b.split("-", 1)
        lo, hi = float(lo_s), float(hi_s)
        return float(rng.uniform(lo, hi))
    # Fallback: treat as a point value
    return float(b)

# -------------------------------------------------------------------
# Core simulation
# -------------------------------------------------------------------

def run_sim(
    params: SimParams,
    rates: AllRates,
    tenure: TenureDist,
    inmovers: InMoverDist,
    return_time_stats: bool = False,
    prebuilt_schedule: Optional[List[TransitionSnapshot]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run the Monte Carlo simulation over n_props dwellings.

    Returns a summary dict of probabilities (and optional time stats).
    """

    if params.age_transition_mode not in AGE_TRANSITION_MODES:
        raise ValueError(
            f"Unknown age_transition_mode {params.age_transition_mode!r}; "
            f"supported: {AGE_TRANSITION_MODES}"
        )

    rng = np.random.default_rng(params.seed)

    # Validate inputs
    _validate_bracketed_dict("inmovers", inmovers.probs_by_bracket)

    for b in BRACKETS:
        if b not in tenure.probs_by_bracket:
            raise ValueError(f"tenure.probs_by_bracket missing bracket {b}")
        if len(tenure.probs_by_bracket[b]) != len(tenure.buckets):
            raise ValueError(f"tenure probs length mismatch for {b}: "
                             f"{len(tenure.probs_by_bracket[b])} vs buckets {len(tenure.buckets)}")
    schedule = prebuilt_schedule if prebuilt_schedule is not None else [
        TransitionSnapshot(time_year=0.0, rates=rates, profiles=make_profiles(rates))
    ]

    # Precompute bracket widths (years until next boundary, by current bracket)
    width_map = {
        "15-24": 10.0,
        "25-34": 10.0,
        "35-44": 10.0,
        "45-54": 10.0,
        "55-64": 10.0,
        "65-74": 10.0,
        "75+":   9999.0,  # terminal
    }

    # Inclusive bracket bounds so we can randomise an in-mover's exact age within the bracket.
    # For example, in "35-44" age could be 35..44; years-to-next-boundary would be 10..1 respectively.
    br_bounds = {
        "15-24": (15, 24),
        "25-34": (25, 34),
        "35-44": (35, 44),
        "45-54": (45, 54),
        "55-64": (55, 64),
        "65-74": (65, 74),
        "75+":   (75, None),  # open-ended / terminal
    }

    def years_to_boundary_at_move_in(br: str) -> float:
        """Years until ageing into the next bracket for a new in-mover household."""
        lo, hi = br_bounds[br]
        if hi is None:
            return width_map[br]  # terminal
        age = int(rng.integers(lo, hi + 1))  # uniform over inclusive ages in bracket
        return float(hi - age + 1)           # e.g. 35->10 years; 44->1 year

    def sample_age_at_move_in(br: str) -> int:
        lo, hi = br_bounds[br]
        if hi is None:
            hi = lo + 9
        return int(rng.integers(lo, hi + 1))


    # Pre-compute CDFs for probability vectors (validate once, not on every sample)
    inmover_vec = [float(inmovers.probs_by_bracket[b]) for b in BRACKETS]
    inmover_cdf = _prepare_cdf("inmover", inmover_vec)

    tenure_cdfs = {b: _prepare_cdf(f"tenure_{b}", tenure.probs_by_bracket[b]) for b in BRACKETS}

    def pick_first_bracket() -> str:
        return BRACKETS[_sample_from_cdf(inmover_cdf, rng)]

    ever_any = np.zeros(params.n_props, dtype=bool)
    ever_phys = np.zeros(params.n_props, dtype=bool)

    time_any = np.zeros(params.n_props)
    time_phys = np.zeros(params.n_props)
    time_total = np.zeros(params.n_props)

    t_buckets = tenure.buckets
    
    t0 = time.time()
    PROGRESS_EVERY = 50_000


    for i in range(params.n_props):

        # ---- progress indicator ----
        if verbose and (i + 1) % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            pct = 100.0 * (i + 1) / params.n_props
            print(
                f"Simulated {i+1:,}/{params.n_props:,} dwellings "
                f"({pct:.1f}%) – elapsed {elapsed/60:.1f} min"
            )

        t = 0.0
        transition_idx = 0
        current_snapshot = schedule[transition_idx]
        br = pick_first_bracket()
        age_year = sample_age_at_move_in(br) if params.age_transition_mode == "annual_interpolated" else None
        yrs_to_boundary = 1.0 if params.age_transition_mode == "annual_interpolated" else years_to_boundary_at_move_in(br)

        # Seed first household statuses from prevalence at starting bracket
        if params.age_transition_mode == "annual_interpolated":
            assert age_year is not None
            any_d, phys_d = _seed_household_state_at_age(age_year, current_snapshot.profiles, rng)
        else:
            any_d, phys_d = _seed_household_state(br, current_snapshot.profiles, rng)

        ever_any[i] = ever_any[i] or any_d
        ever_phys[i] = ever_phys[i] or phys_d

        # Dwelling occupancy lifecycle
        while t < float(params.horizon_years):
            # Draw tenure for current bracket
            b_idx = _sample_from_cdf(tenure_cdfs[br], rng)
            dur = sample_tenure_years(t_buckets[b_idx], rng)

            # Extend tenure if disabled household (any/physical)
            if params.disabled_tenure_factor != 1.0 and (any_d or phys_d):
                dur *= float(params.disabled_tenure_factor)

            remaining = dur

            while remaining > 0.0 and t < float(params.horizon_years):
                next_transition_time = (
                    schedule[transition_idx + 1].time_year
                    if transition_idx + 1 < len(schedule)
                    else float("inf")
                )
                seg = min(remaining, yrs_to_boundary, next_transition_time - t, float(params.horizon_years) - t)

                # accumulate time
                time_total[i] += seg
                if any_d:
                    time_any[i] += seg
                if phys_d:
                    time_phys[i] += seg

                t += seg
                remaining -= seg
                yrs_to_boundary -= seg

                hit_transition = next_transition_time < float("inf") and abs(t - next_transition_time) <= 1e-12
                hit_age_boundary = yrs_to_boundary <= 1e-12

                if hit_transition and t < float(params.horizon_years):
                    prev_profiles = current_snapshot.profiles
                    transition_idx += 1
                    current_snapshot = schedule[transition_idx]
                    any_d, phys_d = _apply_time_step_transition(
                        br,
                        prev_profiles,
                        current_snapshot.profiles,
                        rng,
                        any_d,
                        phys_d,
                    )
                    ever_any[i] = ever_any[i] or any_d
                    ever_phys[i] = ever_phys[i] or phys_d

                # If we've hit a bracket boundary, transition to next bracket and apply acquisitions.
                # Calendar-time transitions are processed first when both happen at the same instant.
                if hit_age_boundary and t < float(params.horizon_years):
                    if params.age_transition_mode == "annual_interpolated":
                        assert age_year is not None
                        any_d, phys_d = _apply_annual_age_transition(
                            age_year,
                            current_snapshot.profiles,
                            rng,
                            any_d,
                            phys_d,
                        )
                        age_year += 1
                        br = _bracket_for_age(age_year)
                        yrs_to_boundary = 1.0
                    elif br != "75+":
                        next_br = BRACKETS[BRACKET_IDX[br] + 1]

                        any_d, phys_d = _apply_age_boundary_transition(
                            br,
                            next_br,
                            current_snapshot.profiles,
                            rng,
                            any_d,
                            phys_d,
                        )
                        br = next_br
                        yrs_to_boundary = width_map[br]  # now at the start of the new bracket
                    ever_any[i] = ever_any[i] or any_d
                    ever_phys[i] = ever_phys[i] or phys_d

            # Tenure ended -> household moves out -> new household moves in
            if t < float(params.horizon_years):
                # Subsequent households always use in-mover distribution
                br = BRACKETS[_sample_from_cdf(inmover_cdf, rng)]
                age_year = sample_age_at_move_in(br) if params.age_transition_mode == "annual_interpolated" else None
                yrs_to_boundary = 1.0 if params.age_transition_mode == "annual_interpolated" else years_to_boundary_at_move_in(br)

                if params.age_transition_mode == "annual_interpolated":
                    assert age_year is not None
                    any_d, phys_d = _seed_household_state_at_age(age_year, current_snapshot.profiles, rng)
                else:
                    any_d, phys_d = _seed_household_state(br, current_snapshot.profiles, rng)

                ever_any[i] = ever_any[i] or any_d
                ever_phys[i] = ever_phys[i] or phys_d

    results: Dict[str, float] = {
        "p_ever_any": float(ever_any.mean()),
        "p_ever_physical": float(ever_phys.mean()),
    }

    if return_time_stats:
        # Avoid divide-by-zero (shouldn't occur, but safe)
        denom = np.where(time_total > 0, time_total, 1.0)
        results.update({
            "pct_time_any": float((time_any / denom).mean()),
            "pct_time_physical": float((time_phys / denom).mean()),
        })

    return results
