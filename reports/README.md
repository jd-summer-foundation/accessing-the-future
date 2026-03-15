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
