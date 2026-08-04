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

All five raw workbooks are licensed by the ABS under Creative Commons Attribution 4.0 International (CC BY 4.0), which permits copying, redistribution, and derivative or commercial use conditional on attribution.

| File | Source | Licensing note |
| --- | --- | --- |
| `sdac22_household_disability.xlsx` | ABS SDAC22 custom extract, ABS quotation LS008242 | Supplied under a custom-extract quotation licensed CC BY 4.0; those terms take precedence over the ABS's default copyright conditions for customised products |
| `2. Housing mobility.xlsx` | ABS Housing Mobility and Conditions, public data cube | CC BY 4.0 under the ABS's default website licence |
| `SDACDC01.xlsx` | ABS SDAC time-series, public data cube | CC BY 4.0 under the ABS's default website licence |
| `6427017.xlsx` | ABS 6427.0 Producer Price Indexes Table 17, public data cube | CC BY 4.0 under the ABS's default website licence |
| `87520_activity.xlsb` | ABS 8752.0 Building Activity, public data cube | CC BY 4.0 under the ABS's default website licence |

Release-page download links for the four public data cubes are recorded in [provenance.md](provenance.md).

### Required attribution

- Unmodified ABS files — everything in `data/raw/` — carry "Source: Australian Bureau of Statistics".
- Material derived or transformed from ABS data carries "Based on Australian Bureau of Statistics data". This covers `data/processed/model_inputs.csv`, `data/processed/construction_index.csv`, `data/processed/dwelling_mix.csv`, and the model outputs written to `results/` and `reports/`.

### Archival deposits

The CC BY 4.0 attribution condition travels with the ABS material. Any archival deposit that bundles `data/raw/` — a Zenodo, openICPSR, or Dataverse record, for example — should therefore be recorded under CC BY 4.0 rather than CC0, even though the code in this repository is CC0 1.0. See the [release checklist](../docs/release.md).
