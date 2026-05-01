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
make run-baseline
make report
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
- Legacy processed workbook retained for migration checks: [data/processed/legacy_model_inputs.xlsx](data/processed/legacy_model_inputs.xlsx)
- Version-controlled derivation rules: [configs/derivation.yaml](configs/derivation.yaml)
- Version-controlled scenario definitions: [configs/baseline.yaml](configs/baseline.yaml)

### Process
- [scripts/build_model_inputs.py](scripts/build_model_inputs.py) extracts disability prevalence and household totals from SDAC22, tenure distributions from Housing Mobility Table 2.2, and derives the in-mover distribution from raw `<1 year` tenure counts by age.
- [run_from_excel.py](run_from_excel.py) reads `data/processed/model_inputs.csv`, normalizes distributions, runs the four scenarios, and writes run manifests.
- [scripts/generate_reports.py](scripts/generate_reports.py) converts scenario summaries into one table and two figures.

### Outputs
- `data/processed/model_inputs.csv`
- `results/<run_name>/scenario_summaries.csv`
- `results/<run_name>/inputs_used.csv`
- `results/<run_name>/profiles_used.csv`
- `results/<run_name>/run_manifest.json`
- `reports/tables/table_01_scenario_summary.csv`
- `reports/tables/table_01_scenario_summary.md`
- `reports/figures/figure_01_ever_probabilities.png`
- `reports/figures/figure_02_time_share.png`

## Reproducibility Notes

- Fixed defaults for the baseline run live in [configs/baseline.yaml](configs/baseline.yaml): `seed=123`, `n_props=44346`, `horizon_years=20`.
- `make verify-data` checks the canonical raw workbooks listed in [data/checksums.sha256](data/checksums.sha256) before rebuilds.
- The processed CSV is deterministic and is validated against the current raw-source derivation on every `make validate-data`.
- The run manifest records commit hash, dependency versions, input checksum, config checksum, and runtime parameters.

## Transition Matrices

The model now supports optional calendar-time transition matrices so disability rates can change over time even within the same age bracket.

This is configured in the run YAML, for example [configs/smoke.yaml](configs/smoke.yaml) and [configs/baseline.yaml](configs/baseline.yaml). Those shipped configs use:

```yaml
transition_model:
  interval_years: 1
  matrices:
    any_dis: identity
    severe_prof: identity
    motor_phys: identity
    phys2: identity
```

`identity` means "do not change the rate vector over calendar time." It is a no-op proof of concept.

### Why The Matrix Is 7x7

The rate vector has one entry per model age bracket:

```text
["15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
```

That is 7 brackets, so each series is represented as a 7-element vector and each transition operator is a 7x7 matrix.

For each modeled series, the engine computes:

```text
next_rates = M @ current_rates
```

with the bracket order above.

`M[row][col]` tells the model how much of the current rate in source bracket `col` contributes to the next-step rate in destination bracket `row`.

### Important Interpretation

These matrices are for **calendar-time change in prevalence within age brackets**, not for ageing people from one bracket to the next.

The simulation already has separate logic for households ageing from `15-24` to `25-34`, `25-34` to `35-44`, and so on. Because of that:

- Do use the matrix to represent changes like "the disability rate among 55-64 year olds is drifting upward over calendar time."
- Do not use the matrix to encode age-to-age movement probabilities. That would double-count ageing.

For most practical first versions, a **diagonal matrix** is the safest place to start. That means each age bracket's future rate depends only on its own current rate.

### What Data Should Go Into The Matrix

There are four separate matrices, one per modeled series:

- `any_dis`
- `severe_prof`
- `motor_phys`
- `phys2`

In the current MVP, the matrix values are best thought of as a **rate-update operator** for each series.

If you estimate that, for example, the `45-54` prevalence for `any_dis` should rise from `0.20` this year to `0.21` next year, then a simple diagonal approximation would use:

```text
M[45-54, 45-54] = 0.21 / 0.20 = 1.05
```

and keep the other off-diagonal cells at `0`.

In other words, if your empirical work produces bracket-specific prevalence rates at time `t` and `t + 1`, a simple way to build the first matrix is:

```text
diagonal_i = target_rate_i_next_period / current_rate_i
```

This is only a starting point. If you later estimate a richer operator from repeated cross-sections, you can replace the diagonal with a full 7x7 matrix.

### How To Supply A Real Matrix

Replace `identity` with an explicit 7x7 numeric array in the YAML. Example:

```yaml
transition_model:
  interval_years: 1
  matrices:
    any_dis:
      - [1.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
      - [0.00, 1.01, 0.00, 0.00, 0.00, 0.00, 0.00]
      - [0.00, 0.00, 1.01, 0.00, 0.00, 0.00, 0.00]
      - [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
      - [0.00, 0.00, 0.00, 0.00, 0.99, 0.00, 0.00]
      - [0.00, 0.00, 0.00, 0.00, 0.00, 0.99, 0.00]
      - [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
    severe_prof: identity
    motor_phys: identity
    phys2: identity
```

That example says:

- `15-24` `any_dis` rates grow by 2% per transition step
- `25-34` and `35-44` grow by 1% per step
- `45-54` stays flat
- `55-64` and `65-74` decline by 1% per step
- the other three modeled series remain unchanged

### Where To Put The Matrix

You can define `transition_model` at two levels:

- Top level in the YAML: applies to every scenario in the run
- Inside an individual scenario: overrides the top-level setting for that scenario only

Example scenario override:

```yaml
scenarios:
  - name: main_inmover_first
    first_draw_source: inmover
    disabled_tenure_factor: 1.0
    rate_scale: 1.0

  - name: custom_transition_scenario
    first_draw_source: inmover
    disabled_tenure_factor: 1.0
    rate_scale: 1.0
    transition_model:
      interval_years: 5
      matrices:
        any_dis: identity
        severe_prof: identity
        motor_phys: identity
        phys2:
          - [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          - [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
          - [0.00, 0.00, 1.02, 0.00, 0.00, 0.00, 0.00]
          - [0.00, 0.00, 0.00, 1.02, 0.00, 0.00, 0.00]
          - [0.00, 0.00, 0.00, 0.00, 1.01, 0.00, 0.00]
          - [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00]
          - [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
```

### Practical Workflow

1. Start from `identity` everywhere to verify the run still reproduces the current baseline.
2. Decide which series you want to vary over time. Often `any_dis` is the cleanest first target.
3. Estimate the calendar-time change you want within each age bracket.
4. Translate that into a 7x7 matrix for that series. Start diagonal unless you have a strong reason not to.
5. Leave the other series as `identity` until you have data for them.
6. Put the matrix into a copy of [configs/baseline.yaml](configs/baseline.yaml) or [configs/smoke.yaml](configs/smoke.yaml) and run:

```bash
python3 run_from_excel.py --config path/to/your_config.yaml
```

### Guardrails

- The matrix must be numeric and exactly 7 rows by 7 columns.
- The engine clips resulting rates to `[0, 1]` after each transition step.
- The model still enforces the existing monotonicity and subtype rules after each update.
- If you only want one series to change, set the other three matrices to `identity`.
- If you already have true age-to-age transition probabilities from a richer state model, those are not yet plugged in directly here. The current MVP expects a calendar-time operator on the 7-element prevalence vector for each series.

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
