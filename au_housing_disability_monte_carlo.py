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
1) any_dis: any disability (base category; prevalence by age bracket; enforced non-decreasing with age)
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

    # For the VERY FIRST household in each dwelling:
    #   "inmover" -> draw age from in-mover distribution (default)
    #   "general" -> draw age from general population distribution (supplied separately)
    first_draw_source: str = "inmover"

    # If household currently has ANY/PHYSICAL, lengthen tenure by this factor (e.g. 1.2)
    disabled_tenure_factor: float = 1.0


@dataclass(frozen=True)
class TransitionSnapshot:
    """Precomputed rate/profile state for a calendar-time update boundary."""

    time_year: float
    rates: AllRates
    profiles: Dict[str, object]


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


def _cumulative_max_monotone(vals: List[float]) -> List[float]:
    """Make a list non-decreasing by cumulative max."""
    out = []
    m = -np.inf
    for v in vals:
        m = max(m, float(v))
        out.append(m)
    return out


def _acquire_probs_from_adjusted(adjusted_rates: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    """
    Given an adjusted (non-decreasing) prevalence profile across age brackets,
    compute acquisition probabilities for transitions b0 -> b1 under persistence:
        acquire_prob = (p1 - p0) / (1 - p0)   if p1 > p0
                     = 0                      otherwise
    """
    _validate_bracketed_dict("adjusted_rates", adjusted_rates)

    probs: Dict[Tuple[str, str], float] = {}
    for i in range(len(BRACKETS) - 1):
        b0, b1 = BRACKETS[i], BRACKETS[i + 1]
        p0, p1 = float(adjusted_rates[b0]), float(adjusted_rates[b1])
        if p1 <= p0 or p0 >= 1.0:
            probs[(b0, b1)] = 0.0
        else:
            probs[(b0, b1)] = (p1 - p0) / (1.0 - p0)
    return probs


def make_profiles(all_rates: AllRates) -> Dict[str, object]:
    """
    Build adjusted profiles and acquisition probabilities for:

      - ANY disability (base): monotone with age via cumulative max
      - motor_phys: conditional on ANY (subtype)

    Returned dict includes:
      - adj_any[b]
      - cond_phys[b]    = P(motor_phys | any_dis)
      - acq_any[(b,b1)]
      - acq_cond_phys[(b,b1)]
    """
    _validate_bracketed_dict("any_dis", all_rates.any_dis.by_bracket)
    _validate_bracketed_dict("motor_phys", all_rates.motor_phys.by_bracket)

    # ANY: enforce non-decreasing prevalence
    ordered_any = [float(all_rates.any_dis.by_bracket[b]) for b in BRACKETS]
    adj_any_list = _cumulative_max_monotone(ordered_any)
    adj_any = dict(zip(BRACKETS, adj_any_list))

    # Helper to build conditional P(subtype | any) from subtype total + ANY adjusted totals
    def build_cond_subtype(total_by_bracket: Dict[str, float]) -> Dict[str, float]:
        cond: Dict[str, float] = {}
        for b in BRACKETS:
            ra = float(adj_any[b])                  # adjusted ANY total
            rt = float(total_by_bracket[b])         # subtype total
            if ra <= 0.0:
                # if ANY is 0 but subtype > 0, clamp to 1 (still yields 0 total in simulation unless ANY occurs)
                cond[b] = 0.0 if rt <= 0.0 else 1.0
            else:
                cond[b] = float(np.clip(rt / ra, 0.0, 1.0))
        return cond

    cond_phys = build_cond_subtype(all_rates.motor_phys.by_bracket)

    # Acquisition probs
    acq_any = _acquire_probs_from_adjusted(adj_any)

    # For conditional profiles, build monotone *total* first then compute conditional acquisition
    # by applying the same adjusted ANY totals. Here we just use the conditional profile + ANY transitions:
    # - When ANY is acquired at boundary, subtype can also be "seeded" immediately by cond_*[next_br]
    # - When ANY already present, subtype can be acquired with p = max(0, cond[next]-cond[curr])/(1-cond[curr])
    def acq_from_cond(cond: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        probs: Dict[Tuple[str, str], float] = {}
        for i in range(len(BRACKETS) - 1):
            b0, b1 = BRACKETS[i], BRACKETS[i + 1]
            c0, c1 = float(cond[b0]), float(cond[b1])
            if c1 <= c0 or c0 >= 1.0:
                probs[(b0, b1)] = 0.0
            else:
                probs[(b0, b1)] = (c1 - c0) / (1.0 - c0)
        return probs

    acq_cond_phys = acq_from_cond(cond_phys)

    return {
        "adj_any": adj_any,
        "cond_phys": cond_phys,
        "acq_any": acq_any,
        "acq_cond_phys": acq_cond_phys,
    }


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


def build_trend_schedule(
    rates_2022: AllRates,
    any_2022: Dict[str, float],
    trend: str,
    horizon_years: int,
    *,
    historical_any: Optional[Dict[int, Dict[str, float]]] = None,
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

    snapshots: List[TransitionSnapshot] = []

    if trend == "none":
        scales = {b: 1.0 for b in BRACKETS}
        for offset in range(int(horizon_years) + 1):
            rates_at_t = _scale_all_rates(rates_2022, scales)
            snapshots.append(TransitionSnapshot(float(offset), rates_at_t, make_profiles(rates_at_t)))
        return snapshots

    if trend == "sdac_2015_2022_trend":
        if historical_any is None or 2015 not in historical_any:
            raise ValueError(
                "historical_any must be provided and include year 2015 "
                "for the 'sdac_2015_2022_trend' trend"
            )

        any_2015 = historical_any[2015]
        _validate_bracketed_dict("historical_any[2015]", any_2015)

        # Annual percentage-point increment for any_dis (fractions, not percentages).
        # Negative when the 2015–2022 trend is declining for a bracket.
        inc_any: Dict[str, float] = {
            b: (float(any_2022[b]) - float(any_2015[b])) / 7
            for b in BRACKETS
        }

        # Both any_dis and motor_phys start from the scenario-scaled household-level
        # rates_2022 and apply increments scaled via the person-level relative change.
        any_2022_hh = rates_2022.any_dis.by_bracket
        motor_2022_base = rates_2022.motor_phys.by_bracket
        inc_any_hh: Dict[str, float] = {}
        inc_motor: Dict[str, float] = {}
        for b in BRACKETS:
            if float(any_2022[b]) > 0.0:
                relative_annual_trend = float(inc_any[b]) / float(any_2022[b])
                inc_any_hh[b] = float(any_2022_hh[b]) * relative_annual_trend
                inc_motor[b] = float(motor_2022_base[b]) * relative_annual_trend
            else:
                inc_any_hh[b] = 0.0
                inc_motor[b] = 0.0

        print("Annual increments for 'sdac_2015_2022_trend':")
        for b in BRACKETS:
            print(
                f"  [{b}]  any: 2015={float(any_2015[b])*100:.3f}%  "
                f"2022(person)={float(any_2022[b])*100:.3f}%  "
                f"2022(hh)={float(any_2022_hh[b])*100:.3f}%  "
                f"inc={inc_any_hh[b]*100:+.4f}%/yr  |  "
                f"motor/phys: 2022={float(motor_2022_base[b])*100:.3f}%  "
                f"inc={inc_motor[b]*100:+.4f}%/yr"
            )

        for offset in range(int(horizon_years) + 1):
            any_at_t = {
                b: float(np.clip(float(any_2022_hh[b]) + inc_any_hh[b] * offset, 0.0, 1.0))
                for b in BRACKETS
            }
            motor_at_t = {
                b: float(np.clip(
                    min(float(motor_2022_base[b]) + inc_motor[b] * offset, any_at_t[b]),
                    0.0, 1.0,
                ))
                for b in BRACKETS
            }
            rates_at_t = AllRates(any_dis=Rates(any_at_t), motor_phys=Rates(motor_at_t))
            snapshots.append(TransitionSnapshot(float(offset), rates_at_t, make_profiles(rates_at_t)))

        return snapshots

    # trend == "sdac_2003_2022_trend"
    if historical_any is None or 2003 not in historical_any:
        raise ValueError(
            "historical_any must be provided and include year 2003 "
            "for the 'sdac_2003_2022_trend' trend"
        )

    any_2003 = historical_any[2003]
    _validate_bracketed_dict("historical_any[2003]", any_2003)

    # Annual percentage-point increment for any_dis (fractions, not percentages).
    # Negative when the 2003–2022 trend is declining for a bracket.
    annual_increment_any: Dict[str, float] = {
        b: (float(any_2022[b]) - float(any_2003[b])) / 19
        for b in BRACKETS
    }

    # Both any_dis and motor_phys start from the scenario-scaled household-level
    # rates_2022 and apply increments scaled via the person-level relative change.
    any_2022_hh = rates_2022.any_dis.by_bracket
    motor_2022 = rates_2022.motor_phys.by_bracket
    annual_increment_any_hh: Dict[str, float] = {}
    annual_increment_motor: Dict[str, float] = {}
    for b in BRACKETS:
        if float(any_2022[b]) > 0.0:
            relative_annual_trend = float(annual_increment_any[b]) / float(any_2022[b])
            annual_increment_any_hh[b] = float(any_2022_hh[b]) * relative_annual_trend
            annual_increment_motor[b] = float(motor_2022[b]) * relative_annual_trend
        else:
            annual_increment_any_hh[b] = 0.0
            annual_increment_motor[b] = 0.0

    print("Annual increments for 'sdac_2003_2022_trend':")
    for b in BRACKETS:
        print(
            f"  [{b}]  any: 2003={float(any_2003[b])*100:.3f}%  "
            f"2022(person)={float(any_2022[b])*100:.3f}%  "
            f"2022(hh)={float(any_2022_hh[b])*100:.3f}%  "
            f"inc={annual_increment_any_hh[b]*100:+.4f}%/yr  |  "
            f"motor/phys: 2022={float(motor_2022[b])*100:.3f}%  "
            f"inc={annual_increment_motor[b]*100:+.4f}%/yr"
        )

    for offset in range(int(horizon_years) + 1):
        any_at_t = {
            b: float(np.clip(float(any_2022_hh[b]) + annual_increment_any_hh[b] * offset, 0.0, 1.0))
            for b in BRACKETS
        }
        motor_at_t = {
            b: float(np.clip(
                min(float(motor_2022[b]) + annual_increment_motor[b] * offset, any_at_t[b]),
                0.0, 1.0,
            ))
            for b in BRACKETS
        }
        rates_at_t = AllRates(any_dis=Rates(any_at_t), motor_phys=Rates(motor_at_t))
        snapshots.append(TransitionSnapshot(float(offset), rates_at_t, make_profiles(rates_at_t)))

    return snapshots


def _acquire_prob(prev_rate: float, next_rate: float) -> float:
    prev = float(prev_rate)
    nxt = float(next_rate)
    if nxt <= prev or prev >= 1.0:
        return 0.0
    return (nxt - prev) / (1.0 - prev)


def _event_occurs(probability: float, rng: np.random.Generator) -> bool:
    prob = float(probability)
    if prob <= 0.0:
        return False
    if prob >= 1.0:
        return True
    return bool(rng.random() < prob)


def _seed_household_state(
    bracket: str,
    profiles: Dict[str, object],
    rng: np.random.Generator,
) -> Tuple[bool, bool]:
    adj_any = profiles["adj_any"]
    cond_phys = profiles["cond_phys"]

    any_d = rng.random() < float(adj_any[bracket])
    phys_d = rng.random() < float(cond_phys[bracket]) if any_d else False
    return any_d, phys_d


def _apply_time_step_transition(
    bracket: str,
    prev_profiles: Dict[str, object],
    next_profiles: Dict[str, object],
    rng: np.random.Generator,
    any_d: bool,
    phys_d: bool,
) -> Tuple[bool, bool]:
    became_any_now = False

    if not any_d:
        p_any = _acquire_prob(float(prev_profiles["adj_any"][bracket]), float(next_profiles["adj_any"][bracket]))
        if _event_occurs(p_any, rng):
            any_d = True
            became_any_now = True

    if any_d:
        if became_any_now:
            if not phys_d and _event_occurs(float(next_profiles["cond_phys"][bracket]), rng):
                phys_d = True
        else:
            if not phys_d:
                p_phys = _acquire_prob(
                    float(prev_profiles["cond_phys"][bracket]),
                    float(next_profiles["cond_phys"][bracket]),
                )
                if _event_occurs(p_phys, rng):
                    phys_d = True

    return any_d, phys_d


def _apply_age_boundary_transition(
    bracket: str,
    next_bracket: str,
    profiles: Dict[str, object],
    rng: np.random.Generator,
    any_d: bool,
    phys_d: bool,
) -> Tuple[bool, bool]:
    became_any_now = False

    if not any_d:
        p_any = float(profiles["acq_any"].get((bracket, next_bracket), 0.0))
        if rng.random() < p_any:
            any_d = True
            became_any_now = True

    if any_d:
        if became_any_now:
            if not phys_d and rng.random() < float(profiles["cond_phys"][next_bracket]):
                phys_d = True
        else:
            if not phys_d:
                p_phys = float(profiles["acq_cond_phys"].get((bracket, next_bracket), 0.0))
                if rng.random() < p_phys:
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
    general_pop_probs: Optional[Dict[str, float]] = None,
    return_time_stats: bool = False,
    prebuilt_schedule: Optional[List[TransitionSnapshot]] = None,
) -> Dict[str, float]:
    """
    Run the Monte Carlo simulation over n_props dwellings.

    Returns a summary dict of probabilities (and optional time stats).
    """

    rng = np.random.default_rng(params.seed)

    # Validate inputs
    _validate_bracketed_dict("inmovers", inmovers.probs_by_bracket)
    if params.first_draw_source not in ("inmover", "general"):
        raise ValueError(f"first_draw_source must be 'inmover' or 'general', got: {params.first_draw_source}")

    if params.first_draw_source == "general":
        if general_pop_probs is None:
            raise ValueError("general_pop_probs must be provided when first_draw_source='general'")
        _validate_bracketed_dict("general_pop_probs", general_pop_probs)

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


    # Pre-compute CDFs for probability vectors (validate once, not on every sample)
    inmover_vec = [float(inmovers.probs_by_bracket[b]) for b in BRACKETS]
    inmover_cdf = _prepare_cdf("inmover", inmover_vec)

    if general_pop_probs is not None:
        general_vec = [float(general_pop_probs[b]) for b in BRACKETS]
        general_cdf = _prepare_cdf("general_pop", general_vec)
    else:
        general_cdf = None

    tenure_cdfs = {b: _prepare_cdf(f"tenure_{b}", tenure.probs_by_bracket[b]) for b in BRACKETS}

    def pick_first_bracket() -> str:
        if params.first_draw_source == "general":
            idx = _sample_from_cdf(general_cdf, rng)  # type: ignore[arg-type]
        else:
            idx = _sample_from_cdf(inmover_cdf, rng)
        return BRACKETS[idx]

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
        if (i + 1) % PROGRESS_EVERY == 0:
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
        yrs_to_boundary = years_to_boundary_at_move_in(br)

        # Seed first household statuses from prevalence at starting bracket
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
                if hit_age_boundary and br != "75+" and t < float(params.horizon_years):
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
                yrs_to_boundary = years_to_boundary_at_move_in(br)

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
