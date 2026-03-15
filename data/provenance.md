# Data Provenance

## Raw sources

- [data/raw/sdac22_household_disability.xlsx](raw/sdac22_household_disability.xlsx), an ABS SDAC22 extract containing household-level disability and physical long-term health condition tables by age of the household responsible adult.
- [data/raw/2. Housing mobility.xlsx](raw/2.%20Housing%20mobility.xlsx), an ABS Housing Mobility and Conditions extract containing age-specific length-of-time-in-dwelling profiles. This workbook is downloadable from the ABS release page [Housing Mobility and Conditions, latest release](https://www.abs.gov.au/statistics/people/housing/housing-mobility-and-conditions/latest-release#data-downloads), under the data cube name `2. Housing mobility.xlsx`.

## Derived directly from SDAC22

The following processed columns are extracted directly from the raw workbook:

- `DIP_any` from `Table 1_3 Proportions`, column 1
- `DIP_severe` from `Table 1_3 Proportions`, column 2
- `DIP_physical` from `Table 1_3 Proportions`, column 4
- `DIP_physical2` from `Table 1_3 Proportions`, column 5
- `Age distribution` from `Table 1_1 Estimates`, total households column, rounded half-up to match the legacy workbook

## Derived from Housing Mobility Table 2.2

- The seven tenure columns are extracted directly from `Table 2.2 ALL HOUSEHOLDS, Length of time reference person has lived in current dwelling, by selected household characteristics`, using the `Age of household reference person` block.
- The in-mover distribution is derived from the same table as:

`(<1 year share / 100) * estimated households`, normalized across the seven model age brackets.

`Table 2.5` is intentionally not used because it collapses `65 and over` into one bin and therefore cannot reproduce the model's separate `65-74` and `75+` categories.

## Validation oracle

The legacy processed workbook [data/processed/legacy_model_inputs.xlsx](processed/legacy_model_inputs.xlsx) is retained only as a migration reference. The canonical starting point for reproduction is the pair of raw workbooks plus the derivation config, not the legacy workbook.
