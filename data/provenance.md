# Data Provenance

## Raw sources

- [data/raw/sdac22_household_disability.xlsx](raw/sdac22_household_disability.xlsx), an ABS SDAC22 extract containing household-level disability and physical long-term health condition tables by age of the household responsible adult.
- [data/raw/2. Housing mobility.xlsx](raw/2.%20Housing%20mobility.xlsx), an ABS Housing Mobility and Conditions extract containing age-specific length-of-time-in-dwelling profiles. This workbook is downloadable from the ABS release page [Housing Mobility and Conditions, latest release](https://www.abs.gov.au/statistics/people/housing/housing-mobility-and-conditions/latest-release#data-downloads), under the data cube name `2. Housing mobility.xlsx`.
- [data/raw/SDACDC01.xlsx](raw/SDACDC01.xlsx), an ABS SDAC time-series data cube containing "any disability" proportions by age for survey years 2003, 2009, 2012, 2015, 2018, and 2022 (Table 1.3, "all persons"), and population estimates by fine-grained age group (Table 3.1 Estimates, column L, rows 43–54).

## Derived directly from SDAC22

The following processed columns are extracted directly from the raw workbook:

- `DIP_any` from `Table 1_3 Proportions`, column 1
- `DIP_physical` from `Table 1_3 Proportions`, column 4
- `DIP_any_moe` and `DIP_physical_moe` from the matching columns in `Table 1_4 MoEs`
- `Age distribution` from `Table 1_1 Estimates`, total households column, rounded half-up.

## Derived from Housing Mobility Table 2.2

- The seven tenure columns are extracted directly from `Table 2.2 ALL HOUSEHOLDS, Length of time reference person has lived in current dwelling, by selected household characteristics`, using the `Age of household reference person` block.
- The in-mover distribution is derived from the same table as:

`(<1 year share / 100) * estimated households`, normalized across the seven model age brackets.

`Table 2.5` is intentionally not used because it collapses `65 and over` into one bin and therefore cannot reproduce the model's separate `65-74` and `75+` categories.

## Derived from SDACDC01.xlsx

The `DIP_any_hist_YYYY` columns in the processed model inputs (for years 2003, 2009, 2012, 2015, 2018, 2022) are read from `Table 1.3 Proportions` ("all persons") in `SDACDC01.xlsx`. The table uses fine-grained age groups (15–24, 25–34, 35–44, 45–54, 55–59, 60–64, 65–69, 70–74, 75–79, 80–84, 85–89, 90 and over) which are combined into the model's 7 age brackets using population-weighted averaging:

```
combined_rate = sum(rate_i * population_i) / sum(population_i)
```

Population counts for weighting come from `Table 3.1 Estimates`, column L (total persons), rows 43–54 (the same 12 fine-grained age groups, in order). Weights are the same for all survey years.

These rates are **person-level** disability prevalence and are used only to compute the `sdac_2003_2022_trend` annual increment — they are **not** the simulation's base rates (which are household-level, from SDAC22).

## Validation

The canonical starting point for reproduction is the pair of raw workbooks plus the derivation config. `make validate-data` rebuilds the processed input from those sources and compares it with [data/processed/model_inputs.csv](processed/model_inputs.csv).
