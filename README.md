# Accessing the Future: Reproducible Research Package

This repository contains a Monte Carlo simulation model estimating the likelihood that Australian dwellings will, over a 20-year default horizon, be occupied by households including a person with disability or physical long-term health needs.

The repository now supports a raw-to-results workflow:

1. Read the raw SDAC22 and housing mobility workbooks in `data/raw/`
2. Build canonical processed model inputs in `data/processed/`
3. Run the simulation scenarios into `results/`
4. Generate paper-facing tables and figures into `reports/`

## Quick Start

Create a Python 3.13 environment, then install the pinned dependencies:

```bash
python3 -m pip install -r requirements.lock.txt
```

Verify the canonical raw inputs before building anything:

```bash
make verify-data
```

Then run the full workflow from the repository root:

```bash
make reproduce
```

That command will:

1. Verify the raw source workbooks against `data/checksums.sha256`
2. Build `data/processed/model_inputs.csv`
3. Validate the processed inputs against the raw source derivation
4. Run the baseline scenarios into `results/baseline/`
5. Generate manuscript artifacts in `reports/`

## Canonical Commands

```bash
make verify-data
make build-data
make validate-data
make construction-index
make dwelling-mix
make run-baseline
make report
make manuscript
make cost-analysis
make trend-tables
make reproduce
make release-check
```

For a fast smoke run:

```bash
make smoke
```

`make smoke` writes run outputs to `results/smoke/` and report artifacts to `reports/smoke/`.

## Repository Layout

```text
.
├── au_housing_disability_monte_carlo.py
├── run_from_excel.py
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── results/
├── reports/
└── tests/
```

## Inputs -> Process -> Outputs

### Inputs
- Raw source workbook: [data/raw/sdac22_household_disability.xlsx](data/raw/sdac22_household_disability.xlsx)
- Raw housing mobility workbook: [data/raw/2. Housing mobility.xlsx](data/raw/2.%20Housing%20mobility.xlsx)
- SDAC time-series data cube: [data/raw/SDACDC01.xlsx](data/raw/SDACDC01.xlsx) — historical disability proportions by age for 2003–2022 (Table 1.3) and population weights (Table 3.1), used to compute annual increments for the trend scenarios
- Producer Price Indexes workbook: [data/raw/6427017.xlsx](data/raw/6427017.xlsx) — ABS 6427.0 Table 17 house-construction indexes for NSW and WA, used by the retrofit cost analysis
- Building Activity data cube: [data/raw/87520_activity.xlsb](data/raw/87520_activity.xlsb) — ABS 8752.0 dwelling commencements by structure type, used by the retrofit cost analysis to weight per-type costs into per-state figures
- Version-controlled derivation rules: [configs/derivation.yaml](configs/derivation.yaml)
- Version-controlled scenario definitions: [configs/baseline.yaml](configs/baseline.yaml)
- Version-controlled cost assumptions: [configs/cost_analysis.yaml](configs/cost_analysis.yaml)

### Process
- [scripts/build_model_inputs.py](scripts/build_model_inputs.py) extracts disability prevalence, margin-of-error bounds, and household totals from SDAC22, tenure distributions from Housing Mobility Table 2.2, and derives the in-mover distribution from raw `<1 year` tenure counts by age.
- [run_from_excel.py](run_from_excel.py) reads `data/processed/model_inputs.csv`, normalizes distributions, projects disability rates forward under the configured trend (see [Time-Varying Disability Rates](#time-varying-disability-rates)), expands each configured scenario into low/base/high confidence-bound cases when MoE inputs are available, and writes run manifests.
- [scripts/generate_reports.py](scripts/generate_reports.py) converts scenario summaries into one table and two figures.
- [scripts/manuscript_figures.py](scripts/manuscript_figures.py) produces the paper-facing Table 1 and Figures 1-2 (base estimate with low/high error bars per scenario) into `reports/manuscript/`.
- [scripts/build_construction_index.py](scripts/build_construction_index.py) extracts the NSW/WA house-construction cost index from the PPI workbook.
- [scripts/build_dwelling_mix.py](scripts/build_dwelling_mix.py) derives the NSW/WA new-dwelling structure-type mix (house/townhouse/apartment) from the Building Activity cube.
- [scripts/retrofit_cost_analysis.py](scripts/retrofit_cost_analysis.py) compares the NPV of building all new homes accessible against retrofitting on first occupancy by a household with disability, using the model's first-occupancy CDF (see [docs/cost_analysis.md](docs/cost_analysis.md)).
- [scripts/trend_tables.py](scripts/trend_tables.py) emits the paper's Appendix B trend-parameter table (historical person-level rates, household-level 2022 base rates, and the relative annual change for each trend window).

### Outputs
- `data/processed/model_inputs.csv`
- `data/processed/construction_index.csv`
- `data/processed/dwelling_mix.csv`
- `results/<run_name>/scenario_summaries.csv`
- `results/<run_name>/first_occupancy_cdf.csv` — cumulative share of dwellings first occupied by a household with a physical/any disability, by year since build
- `results/<run_name>/inputs_used.csv`
- `results/<run_name>/profiles_used.csv`
- `results/<run_name>/run_manifest.json`
- `reports/tables/table_01_scenario_summary.csv`
- `reports/tables/table_01_scenario_summary.md`
- `reports/figures/figure_01_ever_probabilities.png`
- `reports/figures/figure_02_time_share.png`
- `reports/manuscript/table_scenario_summary.{csv,md}` (paper Table 1)
- `reports/manuscript/figure_01_ever_probabilities.png`, `reports/manuscript/figure_02_time_share.png` (paper Figures 1-2)
- `reports/tables/retrofit_vs_newbuild_summary.{csv,md}` and `reports/tables/retrofit_vs_newbuild_cashflows.csv` — retrofit vs. accessible-new-build NPV comparison ([docs/cost_analysis.md](docs/cost_analysis.md))
- `reports/tables/appendix_b_trend_parameters.{csv,md}` — paper Appendix B trend parameters (`make trend-tables`)

## Reproducibility Notes

- Fixed defaults for the baseline run live in [configs/baseline.yaml](configs/baseline.yaml): `seed=123`, `n_props=50000`, `horizon_years=20`, `start_year=2022` (the SDAC base year the trend projection is anchored to).
- Households age one year at a time. Disability acquisition is tested every year of age against prevalence linearly interpolated between bracket midpoint ages, and a new household's initial status is seeded from the interpolated rate at its exact age.
- `make verify-data` checks the canonical raw workbooks listed in [data/checksums.sha256](data/checksums.sha256) before rebuilds.
- The processed CSV is deterministic and is validated against the current raw-source derivation on every `make validate-data`.
- The run manifest records commit hash, dependency versions, input checksum, config checksum, and runtime parameters.

## Time-Varying Disability Rates

The model projects how age-bracket disability rates evolve over the simulation horizon with explicit trend settings selected by `transition_model.type` in the run YAML.

### Trend Projection

SDAC 2022 provides the baseline disability rate per age bracket. Because the model runs ~20 years into the future, we project those rates forward under a named trend. The simulation is anchored at 2022 (`run.start_year: 2022`), so the horizon genuinely projects forward (2022 → ~2042) rather than replaying history.

The default `transition_model` block in [configs/baseline.yaml](configs/baseline.yaml) is:

```yaml
transition_model:
  type: trend
  trend: none
  base_year: 2022    # must equal run.start_year
```

Individual scenarios can override the trend via a `trend:` key. The baseline run includes three trend scenarios:

| Trend | Description |
|---|---|
| `none` | Hold each bracket's 2022 rate flat for the whole horizon. |
| `sdac_2003_2022_trend` | Project `any_dis` forward using the linear 2003–2022 annual increment derived from SDACDC01.xlsx Table 1.3 (person-level, population-weighted). `motor_phys` is inferred by scaling the same relative increment to its 2022 household-level baseline. |
| `sdac_2015_2022_trend` | Same method as `sdac_2003_2022_trend` but uses the 2015–2022 window (7 years) instead of the full 19-year series. |

For `sdac_2003_2022_trend` and `sdac_2015_2022_trend` the historical annual increment is:

```
annual_increment = (rate_end_year − rate_start_year) / n_years
```

These increments are person-level (from the SDAC time-series cube) and are used only to compute the slope; the simulation's base rates remain household-level from SDAC22. Each bracket's rate is capped at 1.0 and floored at 0.0 during projection. Each scenario still expands into low/base/high margin-of-error cases, and the resolved `trend` is recorded in `scenario_summaries.csv` and the run manifest. `base_year` must equal `run.start_year` or the run fails fast.

## Data Provenance

The SDAC22 workbook contains the disability prevalence and total-household counts used by the model. The housing mobility workbook provides the tenure profile by age and the raw ingredients for deriving the in-mover distribution. The derivation details are documented in [data/provenance.md](data/provenance.md).

## Testing

Run the automated test suite with:

```bash
python3 -m pytest
```

## Citation and Archival Metadata

- Citation metadata: [CITATION.cff](CITATION.cff)
- Zenodo metadata: [.zenodo.json](.zenodo.json)
- Release checklist: [docs/release.md](docs/release.md)

Run the release validation step before tagging an archival snapshot:

```bash
make release-check
```
