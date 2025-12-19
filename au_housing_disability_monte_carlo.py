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
2) severe_prof: severe/profound disability, modelled as a SUBTYPE of any_dis (conditional on any_dis)
3) motor_phys: mobility/physical disability proxy, modelled as a SUBTYPE of any_dis (conditional on any_dis)
4) phys2: long-term physical condition NOT necessarily disability, modelled INDEPENDENTLY of any_dis

Outputs (summary dict):
- p_ever_any, p_ever_severe, p_ever_physical, p_ever_physical2
- optionally: mean % of time a dwelling is occupied by a household with each category
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import time
import re

BRACKETS: List[str] = ["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
BRACKET_IDX = {b: i for i, b in enumerate(BRACKETS)}

DEFAULT_START_YEAR = 2029
DEFAULT_HORIZON_YEARS = 50


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
    - severe_prof is a subtype of any_dis (severe_prof ⊆ any_dis).
    - motor_phys is a subtype of any_dis (motor_phys ⊆ any_dis).
      Implementation: we convert total motor_phys rates into conditional P(motor_phys | any_dis)
      using the adjusted any_dis profile.
    - phys2 is independent of any_dis (separate process; not a subtype).

    Note: "subset" here is a modelling choice: motor_phys and severe_prof only occur when any_dis is true.
    """
    any_dis: Rates
    severe_prof: Rates
    motor_phys: Rates
    phys2: Rates


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
    n_props: int = 1_200_000
    start_year: int = DEFAULT_START_YEAR
    horizon_years: int = DEFAULT_HORIZON_YEARS
    seed: int = 42

    # For the VERY FIRST household in each dwelling:
    #   "inmover" -> draw age from in-mover distribution (default)
    #   "general" -> draw age from general population distribution (supplied separately)
    first_draw_source: str = "inmover"

    # If household currently has ANY/SEVERE/PHYSICAL, lengthen tenure by this factor (e.g. 1.2)
    disabled_tenure_factor: float = 1.0


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


def acquire_probs_from_adjusted(adjusted_rates: Dict[str, float]) -> Dict[Tuple[str, str], float]:
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
      - severe_prof: conditional on ANY (subtype)
      - motor_phys: conditional on ANY (subtype)
      - phys2: independent of ANY (separate monotone profile)

    Returned dict includes:
      - adj_any[b]
      - cond_severe[b]  = P(severe_prof | any_dis)
      - cond_phys[b]    = P(motor_phys | any_dis)
      - adj_phys2[b]
      - acq_any[(b,b1)]
      - acq_cond_severe[(b,b1)]
      - acq_cond_phys[(b,b1)]
      - acq_phys2[(b,b1)]
    """
    _validate_bracketed_dict("any_dis", all_rates.any_dis.by_bracket)
    _validate_bracketed_dict("severe_prof", all_rates.severe_prof.by_bracket)
    _validate_bracketed_dict("motor_phys", all_rates.motor_phys.by_bracket)
    _validate_bracketed_dict("phys2", all_rates.phys2.by_bracket)

    # 1) ANY: enforce non-decreasing prevalence
    ordered_any = [float(all_rates.any_dis.by_bracket[b]) for b in BRACKETS]
    adj_any_list = _cumulative_max_monotone(ordered_any)
    adj_any = dict(zip(BRACKETS, adj_any_list))

    # 2) phys2: enforce non-decreasing prevalence
    ordered_phys2 = [float(all_rates.phys2.by_bracket[b]) for b in BRACKETS]
    adj_phys2_list = _cumulative_max_monotone(ordered_phys2)
    adj_phys2 = dict(zip(BRACKETS, adj_phys2_list))

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

    cond_severe = build_cond_subtype(all_rates.severe_prof.by_bracket)
    cond_phys = build_cond_subtype(all_rates.motor_phys.by_bracket)

    # Acquisition probs
    acq_any = acquire_probs_from_adjusted(adj_any)
    acq_phys2 = acquire_probs_from_adjusted(adj_phys2)

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

    acq_cond_severe = acq_from_cond(cond_severe)
    acq_cond_phys = acq_from_cond(cond_phys)

    return {
        "adj_any": adj_any,
        "cond_severe": cond_severe,
        "cond_phys": cond_phys,
        "adj_phys2": adj_phys2,
        "acq_any": acq_any,
        "acq_cond_severe": acq_cond_severe,
        "acq_cond_phys": acq_cond_phys,
        "acq_phys2": acq_phys2,
    }


def sample_from_discrete(probs: List[float], rng: np.random.Generator) -> int:
    """Sample an index from a probability vector."""
    p = _validate_probs("sample_from_discrete.probs", probs)
    cdf = np.cumsum(p)
    u = rng.random()
    return int(np.searchsorted(cdf, u, side="right"))


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
) -> Dict[str, float]:
    """
    Run the Monte Carlo simulation over n_props dwellings.

    Returns a summary dict of probabilities (and optional time stats).
    """

    DEBUG_AGE_BOUNDARY = False
    DEBUG_PRINT_LIMIT = 20
    debug_prints_done = 0

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

    prof = make_profiles(rates)
    adj_any = prof["adj_any"]
    cond_severe = prof["cond_severe"]
    cond_phys = prof["cond_phys"]
    adj_phys2 = prof["adj_phys2"]

    acq_any = prof["acq_any"]
    acq_cond_severe = prof["acq_cond_severe"]
    acq_cond_phys = prof["acq_cond_phys"]
    acq_phys2 = prof["acq_phys2"]

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


    # Helper: choose first bracket
    inmover_vec = [float(inmovers.probs_by_bracket[b]) for b in BRACKETS]
    if general_pop_probs is not None:
        general_vec = [float(general_pop_probs[b]) for b in BRACKETS]
    else:
        general_vec = None

    def pick_first_bracket() -> str:
        if params.first_draw_source == "general":
            idx = sample_from_discrete(general_vec, rng)  # type: ignore[arg-type]
        else:
            idx = sample_from_discrete(inmover_vec, rng)
        return BRACKETS[idx]

    ever_any = np.zeros(params.n_props, dtype=bool)
    ever_sev = np.zeros(params.n_props, dtype=bool)
    ever_phys = np.zeros(params.n_props, dtype=bool)
    ever_phys2 = np.zeros(params.n_props, dtype=bool)

    time_any = np.zeros(params.n_props)
    time_sev = np.zeros(params.n_props)
    time_phys = np.zeros(params.n_props)
    time_phys2 = np.zeros(params.n_props)
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
        br = pick_first_bracket()
        yrs_to_boundary = years_to_boundary_at_move_in(br)
        if DEBUG_AGE_BOUNDARY and debug_prints_done < DEBUG_PRINT_LIMIT:
            print(
                f"[DEBUG] Move-in bracket={br}, "
                f"yrs_to_boundary={yrs_to_boundary:.1f}"
            )
            debug_prints_done += 1

        # Seed first household statuses from prevalence at starting bracket
        # motor_phys is conditional on any_dis (subtype); phys2 is independent (separate process)

        any_d = rng.random() < float(adj_any[br])

        if any_d:
            sev_d = rng.random() < float(cond_severe[br])
            phys_d = rng.random() < float(cond_phys[br])
        else:
            sev_d = False
            phys_d = False

        phys2_d = rng.random() < float(adj_phys2[br])

        ever_any[i] = ever_any[i] or any_d
        ever_sev[i] = ever_sev[i] or sev_d
        ever_phys[i] = ever_phys[i] or phys_d
        ever_phys2[i] = ever_phys2[i] or phys2_d

        # Dwelling occupancy lifecycle
        while t < float(params.horizon_years):
            # Draw tenure for current bracket
            probs = tenure.probs_by_bracket[br]
            b_idx = sample_from_discrete(probs, rng)
            dur = sample_tenure_years(t_buckets[b_idx], rng)

            # Extend tenure if disabled household (any/severe/physical)
            if params.disabled_tenure_factor != 1.0 and (any_d or sev_d or phys_d):
                dur *= float(params.disabled_tenure_factor)

            remaining = dur

            while remaining > 0.0 and t < float(params.horizon_years):
                seg = min(remaining, yrs_to_boundary, float(params.horizon_years) - t)

                # accumulate time
                time_total[i] += seg
                if any_d:
                    time_any[i] += seg
                if sev_d:
                    time_sev[i] += seg
                if phys_d:
                    time_phys[i] += seg
                if phys2_d:
                    time_phys2[i] += seg

                t += seg
                remaining -= seg
                yrs_to_boundary -= seg

                # If we've hit a bracket boundary, transition to next bracket and apply acquisitions
                if yrs_to_boundary <= 1e-12 and br != "75+" and t < float(params.horizon_years):
                    next_br = BRACKETS[BRACKET_IDX[br] + 1]
                    if DEBUG_AGE_BOUNDARY and debug_prints_done < DEBUG_PRINT_LIMIT:
                        print(f"[DEBUG] Aged into bracket {next_br} at t={t:.1f}")

                    became_any_now = False

                    # ANY acquisition
                    if not any_d:
                        p_a = float(acq_any.get((br, next_br), 0.0))
                        if rng.random() < p_a:
                            any_d = True
                            became_any_now = True
                            ever_any[i] = True

                    # Severe/physical conditional on ANY
                    if any_d:
                        if became_any_now:
                            # Seed subtype immediately using conditional at new bracket
                            if rng.random() < float(cond_severe[next_br]):
                                sev_d = True
                                ever_sev[i] = True
                            if rng.random() < float(cond_phys[next_br]):
                                phys_d = True
                                ever_phys[i] = True
                        else:
                            # Already had ANY; may newly acquire subtype
                            if not sev_d:
                                p_s = float(acq_cond_severe.get((br, next_br), 0.0))
                                if rng.random() < p_s:
                                    sev_d = True
                                    ever_sev[i] = True
                            if not phys_d:
                                p_p = float(acq_cond_phys.get((br, next_br), 0.0))
                                if rng.random() < p_p:
                                    phys_d = True
                                    ever_phys[i] = True

                    # phys2 independent
                    if not phys2_d:
                        p_p2 = float(acq_phys2.get((br, next_br), 0.0))
                        if rng.random() < p_p2:
                            phys2_d = True
                            ever_phys2[i] = True

                    br = next_br
                    yrs_to_boundary = width_map[br]  # now at the start of the new bracket

            # Tenure ended -> household moves out -> new household moves in
            if t < float(params.horizon_years):
                br = pick_first_bracket()
                yrs_to_boundary = years_to_boundary_at_move_in(br)
                if DEBUG_AGE_BOUNDARY and debug_prints_done < DEBUG_PRINT_LIMIT:
                    print(
                        f"[DEBUG] Move-in bracket={br}, "
                        f"yrs_to_boundary={yrs_to_boundary:.1f}"
                    )
                    debug_prints_done += 1

                any_d = rng.random() < float(adj_any[br])
                if any_d:
                    sev_d = rng.random() < float(cond_severe[br])
                    phys_d = rng.random() < float(cond_phys[br])
                else:
                    sev_d = False
                    phys_d = False
                phys2_d = rng.random() < float(adj_phys2[br])

                ever_any[i] = ever_any[i] or any_d
                ever_sev[i] = ever_sev[i] or sev_d
                ever_phys[i] = ever_phys[i] or phys_d
                ever_phys2[i] = ever_phys2[i] or phys2_d

    results: Dict[str, float] = {
        "p_ever_any": float(ever_any.mean()),
        "p_ever_severe": float(ever_sev.mean()),
        "p_ever_physical": float(ever_phys.mean()),
        "p_ever_physical2": float(ever_phys2.mean()),
    }

    if return_time_stats:
        # Avoid divide-by-zero (shouldn't occur, but safe)
        denom = np.where(time_total > 0, time_total, 1.0)
        results.update({
            "pct_time_any": float((time_any / denom).mean()),
            "pct_time_severe": float((time_sev / denom).mean()),
            "pct_time_physical": float((time_phys / denom).mean()),
            "pct_time_physical2": float((time_phys2 / denom).mean()),
        })

    return results


if __name__ == "__main__":
    # Minimal smoke-test
    any_rates =  {"15-24":0.08,"25-34":0.09,"35-44":0.11,"45-54":0.15,"55-64":0.20,"65-74":0.28,"75+":0.45}
    sev_rates =  {"15-24":0.02,"25-34":0.02,"35-44":0.03,"45-54":0.05,"55-64":0.07,"65-74":0.11,"75+":0.20}
    phys_rates = {"15-24":0.05,"25-34":0.06,"35-44":0.08,"45-54":0.12,"55-64":0.16,"65-74":0.22,"75+":0.30}
    phys2_rates = {"15-24":0.10,"25-34":0.12,"35-44":0.14,"45-54":0.18,"55-64":0.22,"65-74":0.30,"75+":0.40}

    buckets = ["<1","1-2","2-3","3-4","5-9","10-19","20+"]
    probs_by_bracket = {b:[0.1,0.1,0.1,0.1,0.25,0.25,0.1] for b in BRACKETS}
    tenure = TenureDist(buckets=buckets, probs_by_bracket=probs_by_bracket)

    inmovers = InMoverDist({b: 1/len(BRACKETS) for b in BRACKETS})
    rates = AllRates(Rates(any_rates), Rates(sev_rates), Rates(phys_rates), Rates(phys2_rates))

    params = SimParams(n_props=2000, seed=1, first_draw_source="inmover", horizon_years=50)
    print("Self-test summary:", run_sim(params, rates, tenure, inmovers, return_time_stats=True))
