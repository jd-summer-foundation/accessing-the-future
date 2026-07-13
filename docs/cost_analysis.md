# Retrofit vs. accessible-new-build cost analysis

`scripts/retrofit_cost_analysis.py` compares the net present value (NPV) of two
policy options for the projected new-home builds in New South Wales and
Western Australia:

1. **Accessible new build** — every new home includes accessibility features at
   build time.
2. **Retrofit as needed** — a home receives tailored home modifications only
   when it is first occupied by a household that includes a person with a
   (physical / any) disability, using the Monte Carlo model's first-occupancy
   CDF.

Run it with `make cost-analysis` (after `make run-baseline`), or as part of
`make reproduce`. All assumptions live in `configs/cost_analysis.yaml`;
outputs go to `reports/tables/retrofit_vs_newbuild_summary.{csv,md}` and
`reports/tables/retrofit_vs_newbuild_cashflows.csv`, with input checksums and
derived assumptions recorded in `reports/cost_analysis_manifest.json`.

## Inputs

| Input | Source | Role |
| --- | --- | --- |
| First-occupancy CDF | `results/baseline/first_occupancy_cdf.csv` (model output; enabled by `run.return_first_occupancy`) | Probability a new dwelling is first occupied by a household with a physical / any disability within *k* years of being built (k = 0..20), for the low/base/high SDAC margin-of-error cases |
| Construction cost index | `data/processed/construction_index.csv` (ABS 6427.0 Table 17, class 3011 House construction, March quarters; see `data/provenance.md`) | Anchors 2021 costs to observed price levels and sets forward inflation |
| Dwelling mix | `data/processed/dwelling_mix.csv` (ABS 8752.0 Building Activity: shares of houses / townhouses / apartments among new-dwelling commencements, most recent 12 quarters; see `data/provenance.md`) | Weights the per-type costs into per-state figures |
| Per-dwelling costs by type | Centre for International Economics Decision RIS (2021 dollars): new-build Silver $3,874 house / $4,186 townhouse / $5,748 apartment plus $215 mandate overhead; retrofit $18,821 Class 1a / $20,260 Class 2 (see [What the cost figures represent](#what-the-cost-figures-represent)) | The cost of each arm |
| New homes | National Housing Accord state breakdown: 376,000 NSW, 129,000 WA | Scale per-dwelling NPVs to totals |

## What the cost figures represent

All per-dwelling costs come from the CIE Decision RIS on minimum accessibility
standards in the National Construction Code (2021 dollars) and are specified
*per dwelling type* in `configs/cost_analysis.yaml`, then weighted into
per-state figures using each state's new-dwelling structure-type mix
(`data/processed/dwelling_mix.csv`, from ABS 8752.0: commencement shares of
houses / townhouses / apartments over the most recent 3 years).

### New-build arm (Silver-level accessibility at build time)

From CIE DRIS table 6.1, Option 1 (Silver), non-exempt dwellings: **$3,874 per
separate house, $4,186 per townhouse, $5,748 per apartment**. Each figure is
the DCWC quantity-surveyor construction cost (adjusted from Canberra to the
national average using Rawlinson's regional indices) plus the net opportunity
cost of space. The space cost is trivial for Class 1a dwellings ($25/m² net)
but material for apartments ($2,904 per dwelling), where extra functional
space must come out of living space. CIE's own comparison (DRIS table 6.3)
places the DCWC construction estimates at the *low* end of consultation
submissions ($0.6k–$15k for Silver).

Because the scenario is a mandate, a **$215 per-dwelling mandate overhead** is
added: $200 compliance verification (CIE's central case — an additional
building-surveyor inspection; range $43.75–$440) and ~$15 in one-off
transition costs ($28.47M of government and industry retraining spread over
the ~1.90M national dwelling completions of a 10-year regulatory period).
Volume-builder redesign costs are treated as zero, following the DRIS's
qualitative assessment that a transition period largely absorbs them.

With the current dwelling mix this yields roughly **$4,745 (NSW)** and
**$4,281 (WA)** per new dwelling in 2021 dollars (see
`reports/cost_analysis_manifest.json` for the exact derived values).

### Retrofit arm (tailored home modifications)

From CIE DRIS table 3.32 / appendix D ("Approach to estimating the avoidable
cost of home modifications"): **$18,821 per Class 1a dwelling
(houses/townhouses) and $20,260 per Class 2 dwelling (apartments)**.

**These are the average cost of tailored home modifications, not of a full
Silver-standard retrofit.** CIE combines quantity-surveyor (DCWC) unit costs
for retrofitting each Silver-level design element with SDAC 2018 data on which
modifications households with disability actually make. The result is the
weighted-average cost of the modification bundle that actually proceeds in a
modified dwelling, and is corroborated by an independent YPINH/Monash (2015)
estimate of $19,400 for retrofitting basic accessibility features.
Retrofitting *all* Silver elements into one dwelling would cost far more; no
claim of that kind is made here. With the current dwelling mix the weighted
figures are roughly **$19,274 (NSW)** and **$18,950 (WA)**.

The retrofit figures are conservative in two ways:

- They use Silver-level (Option 1) unit costs throughout. CIE maps wheelchair
  users to Gold-level modification costs ($49,706 Class 1a) and reports a
  blended avoidable cost of $22,899 per dwelling. Because the model's
  "physical disability" trigger includes wheelchair users, using the
  Silver-only figure understates the retrofit arm. It is also internally
  consistent: the new-build arm funds Silver features, so the avoided cost is
  costed at Silver level.
- CIE excluded DCWC's "retrofit not practicable" scenarios, so the average is
  over dwellings where modification is feasible; households in homes that
  cannot be modified bear real (e.g. relocation) costs that are not priced
  here.

Retrofit costs also exclude compliance verification and transition overheads
(these are mandate costs applied to the new-build arm only), which further
biases the comparison against the accessible-new-build option.

## Method

For each state, uncertainty case (low/base/high) and category (physical/any):

0. **Per-state costs**: the per-type 2021 costs are weighted by the state's
   new-dwelling structure-type mix (house/townhouse/apartment commencement
   shares from ABS 8752.0, most recent 12 quarters), and the mandate overhead
   is added to the new-build arm. Houses and townhouses take the Class 1a
   retrofit cost; apartments the Class 2 cost. A flat legacy mode
   (`new_build_per_dwelling` / `retrofit_per_dwelling`) reproduces the
   prototype spreadsheet's single $4,000/$19,000 figures.
1. **Forward inflation** is the average year-on-year change of the state's
   March-quarter house construction index over 2000–2019. The window
   deliberately stops before the COVID-era spike (2021–2024 saw rises of up to
   20% per year that would be unreasonable to extrapolate). This gives 3.73%
   p.a. for NSW and 4.05% p.a. for WA.
2. **Cost levels**: the 2021 costs are inflated to the analysis start year
   (2026) using the observed index ratio (e.g. NSW: 189.3 / 136.8), then
   projected forward at the state's average inflation rate.
3. **Build-out**: homes are built in equal annual cohorts over 5 years
   starting in 2026.
4. **New-build arm**: each cohort incurs the per-dwelling accessibility cost
   in its build year.
5. **Retrofit arm**: a home built in year *b* incurs the retrofit cost in year
   *b + k* with probability equal to the CDF increment at year *k* (year 0 =
   the home's first household already includes a person with the disability).
   Cohort staggering (`analysis.stagger_by_cohort`) treats the retrofit clock
   symmetrically with the build-out; the `--no-stagger` flag reverts to the
   prototype spreadsheet's simplification of starting every home's clock in
   2026, and exists for cross-checking against it.
6. **Discounting**: all expected nominal costs are discounted to 2026 at 4.8%
   (approximate 10-year Australian government bond yield). Both the costs and
   the discount rate are nominal, so the comparison is internally consistent.
7. Per-dwelling NPVs are multiplied by each state's new-home count; a `total`
   row aggregates the states.

## Interpretation caveats

- **The retrofit NPV is a conservative lower bound.** The model horizon is 20
  years per cohort, and by year 20 the first-occupancy CDF has reached only
  roughly 70–84% of dwellings (depending on case and category). Every first
  occupancy beyond a cohort's 20-year window — eventually most of the
  remainder — would add retrofit cost but is excluded. Since retrofitting is
  the more expensive arm, this biases the comparison *against* the
  accessible-new-build option.
- The retrofit arm assumes every first occupancy by a household with the
  relevant disability triggers one average-cost set of tailored home
  modifications, and no home is modified twice. In practice some such
  households would need no structural modification (the trigger includes
  moderate physical disability), which cuts the other way from the
  conservatisms noted above.
- Costs per dwelling type are single point estimates from the CIE DRIS; the
  low/base/high range only reflects SDAC prevalence margins of error, not cost
  uncertainty.
- The dwelling mix is a recent-average baseline (last 12 quarters of
  commencements), not a forecast. Within that window the attached share
  (townhouses + apartments) is drifting upward in both states, so the mix
  slightly understates the future apartment share — which understates both
  arms' costs slightly, more so the (apartment-heavier) new-build arm.
- The headline comparison uses the **base / physical** case: retrofit need is
  driven by physical disability, with the `any`-disability columns as an upper
  bracket.

## Differences from the prototype spreadsheet

This analysis reproduces the prototype workbook ("Projected costs of retrofit
vs initial build 2.0") and then deliberately departs from it in four ways.
With `--no-stagger` and the legacy flat costs ($4,000/$19,000), NSW figures
match the workbook exactly except for (2)
(see `tests/test_retrofit_cost_analysis.py`):

1. **Cohort staggering** (default on): the workbook spread new-build costs
   over the 5-year build-out but started every home's retrofit clock in 2026,
   as if all homes existed on day one. Staggering delays each cohort's
   retrofit exposure to its build year.
2. **Full CDF horizon**: the workbook's projection rows stopped 20 years after
   2026, silently dropping the year-20 CDF increment; the script uses the full
   0–20 CDF.
3. **WA inflation fix**: the workbook inflated WA costs at the NSW rate from
   2028 onward (a copy error); the script uses each state's own rate, so WA
   figures are somewhat higher than the workbook's.
4. **Mix-weighted per-state costs**: the workbook used flat $4,000/$19,000
   for both states; the default config weights the CIE DRIS per-type costs by
   each state's dwelling mix and adds the mandate overhead to the new-build
   arm (see [What the cost figures represent](#what-the-cost-figures-represent)).
