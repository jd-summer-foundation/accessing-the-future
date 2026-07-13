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
| Per-dwelling costs | Centre for International Economics Decision RIS (2021 dollars): $4,000 new-build accessibility, $19,000 retrofit (see [What the cost figures represent](#what-the-cost-figures-represent)) | The cost of each arm |
| New homes | National Housing Accord state breakdown: 376,000 NSW, 129,000 WA | Scale per-dwelling NPVs to totals |

## What the cost figures represent

The $19,000 retrofit figure is derived from the CIE Decision RIS on minimum
accessibility standards in the National Construction Code (2021), specifically
table 3.32 and appendix D ("Approach to estimating the avoidable cost of home
modifications").

**It is the average cost of tailored home modifications, not of a full
Silver-standard retrofit.** CIE combines quantity-surveyor (DCWC) unit costs
for retrofitting each Silver-level design element with SDAC 2018 data on which
modifications households with disability actually make. The result is the
weighted-average cost of the modification bundle that actually proceeds in a
modified dwelling — $18,821 for Class 1a dwellings (houses/townhouses) and
$20,260 for Class 2 (apartments), both in 2021 dollars. The rounded $19,000
sits between the two, matching a mixed housing stock, and is corroborated by
an independent YPINH/Monash (2015) estimate of $19,400 for retrofitting basic
accessibility features. Retrofitting *all* Silver elements into one dwelling
would cost far more; no claim of that kind is made here.

The figure is conservative in two ways:

- It uses Silver-level (Option 1) unit costs throughout. CIE maps wheelchair
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

## Method

For each state, uncertainty case (low/base/high) and category (physical/any):

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
- Costs per dwelling are single point estimates from the CIE DRIS; the
  low/base/high range only reflects SDAC prevalence margins of error, not cost
  uncertainty.
- The headline comparison uses the **base / physical** case: retrofit need is
  driven by physical disability, with the `any`-disability columns as an upper
  bracket.

## Differences from the prototype spreadsheet

This analysis reproduces the prototype workbook ("Projected costs of retrofit
vs initial build 2.0") and then deliberately departs from it in three ways.
With `--no-stagger`, NSW figures match the workbook exactly except for (2)
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
