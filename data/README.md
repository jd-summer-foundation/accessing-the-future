# Data Layout

## `data/raw/`
- Contains external source data used to derive model-ready inputs.
- Current canonical raw input: [sdac22_household_disability.xlsx](raw/sdac22_household_disability.xlsx)
- Housing mobility source: [2. Housing mobility.xlsx](raw/2.%20Housing%20mobility.xlsx)

## `data/processed/`
- Contains deterministic artifacts produced by the build step.
- Canonical processed input: [model_inputs.csv](processed/model_inputs.csv)

## Workflow

```bash
make verify-data
make build-data
make validate-data
```

`make verify-data` checks the canonical raw workbook checksums recorded in [checksums.sha256](checksums.sha256). `make build-data` regenerates `model_inputs.csv` from the raw workbooks and the derivation config. `make validate-data` checks schema, bracket ordering, probability sanity, and agreement with the raw-derived build output.

## Licensing and attribution

Raw data in `data/raw/` originates from the Australian Bureau of Statistics (ABS) and is licensed separately from this repository's own code license (see top-level `LICENSE`, CC0 1.0, which covers the code and derived work authored in this repository, not the ABS source material itself).

- `sdac22_household_disability.xlsx` was supplied by the ABS under a custom-extract quotation (ABS quotation LS008242) licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). That licence permits copying, redistribution, and derivative or commercial use of this file, conditional on attribution, and its terms take precedence over the ABS's default copyright conditions for customised products.
- `2. Housing mobility.xlsx` and `SDACDC01.xlsx` are standard public ABS data cube downloads.

Attribution: "Source: Australian Bureau of Statistics" for unmodified ABS files, or "Based on Australian Bureau of Statistics data" for `data/processed/model_inputs.csv` and any other output derived or transformed from ABS material, per the ABS's Creative Commons Attribution 4.0 International licence conditions.
