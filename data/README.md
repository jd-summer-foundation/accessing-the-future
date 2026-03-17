# Data Layout

## `data/raw/`
- Contains external source data used to derive model-ready inputs.
- Current canonical raw input: [sdac22_household_disability.xlsx](raw/sdac22_household_disability.xlsx)
- Housing mobility source: [2. Housing mobility.xlsx](raw/2.%20Housing%20mobility.xlsx)

## `data/processed/`
- Contains deterministic artifacts produced by the build step.
- Canonical processed input: [model_inputs.csv](processed/model_inputs.csv)
- Legacy oracle workbook for parity checks: [legacy_model_inputs.xlsx](processed/legacy_model_inputs.xlsx)

## Workflow

```bash
make verify-data
make build-data
make validate-data
```

`make verify-data` checks the canonical raw workbook checksums recorded in [checksums.sha256](checksums.sha256). `make build-data` regenerates `model_inputs.csv` from the raw workbooks and the derivation config. `make validate-data` checks schema, bracket ordering, probability sanity, and agreement with the raw-derived build output.
