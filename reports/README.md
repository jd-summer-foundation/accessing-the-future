# Report Artifacts

The reporting step currently generates the following manuscript-facing artifacts:

- `tables/table_01_scenario_summary.csv`
- `tables/table_01_scenario_summary.md`
- `figures/figure_01_ever_probabilities.png`
- `figures/figure_02_time_share.png`

## Artifact map

- `table_01_scenario_summary` summarises the four baseline scenarios and their key outcome metrics.
- `figure_01_ever_probabilities` plots the probability that a dwelling ever hosts each household category.
- `figure_02_time_share` plots the average share of occupancy time for each household category.

These artifacts are generated from `results/<run_name>/scenario_summaries.csv` via [scripts/generate_reports.py](../scripts/generate_reports.py).

## Manuscript artifacts

The table and figures used directly in the paper (Table 1, Figures 1-2) are generated separately by [scripts/manuscript_figures.py](../scripts/manuscript_figures.py) into `manuscript/`:

- `manuscript/table_scenario_summary.csv`, `manuscript/table_scenario_summary.md` — the base-case estimate with its low-high margin-of-error range, one row per scenario.
- `manuscript/figure_01_ever_probabilities.png` — probability a new dwelling is ever occupied by a physical-disability or any-disability household, by scenario, with low-high error bars.
- `manuscript/figure_02_time_share.png` — average share of the 20-year horizon occupied by each household type, by scenario, with low-high error bars.

Unlike `generate_reports.py`, which draws one bar per scenario/uncertainty row, this groups the low/base/high cases into a base estimate with an error bar per scenario. Run with `make manuscript` (also run as part of `make reproduce`). Like the other report outputs, these are reproducible artifacts and are not committed to the repository.
