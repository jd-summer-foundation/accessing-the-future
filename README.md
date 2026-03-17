# Accessing the Future: Reproducible Research Package

This repository contains a Monte Carlo simulation model estimating the likelihood that Australian dwellings will, over a 50-year horizon, be occupied by households including a person with disability or physical long-term health needs.

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

- Fixed defaults for the baseline run live in [configs/baseline.yaml](configs/baseline.yaml): `seed=123`, `n_props=44346`, `horizon_years=50`.
- `make verify-data` checks the canonical raw workbooks listed in [data/checksums.sha256](data/checksums.sha256) before rebuilds.
- The processed CSV is deterministic and is validated against the current raw-source derivation on every `make validate-data`.
- The run manifest records commit hash, dependency versions, input checksum, config checksum, and runtime parameters.

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
