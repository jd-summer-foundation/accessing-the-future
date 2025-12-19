# Australian Housing Disability Monte Carlo Model

This repository contains a Monte Carlo simulation model used to estimate the likelihood that Australian dwellings will, over their lifetime, be occupied by a household that includes a person with disability.

The model supports research on accessible and inclusive housing design in the context of population ageing, household turnover, and long-lived housing stock.

---

## Repository structure

```
.
├── au_housing_disability_monte_carlo.py   # Core simulation engine
├── run_from_excel.py                     # Runner script (loads inputs and runs scenarios)
├── inputs/                               # Input data (Excel files; not included in this repo by default)
└── outputs/                              # Model outputs (generated when run)
```

---

## Overview

The model simulates a large number of dwellings over a fixed time horizon. For each dwelling, households move in and out over time according to observed tenure patterns. As households age while remaining in a dwelling, there is a probability that a person within the household acquires a disability, based on age-specific prevalence and transition rates.

Key features:

- Explicit simulation of household turnover within dwellings
- Use of in-mover age distributions (rather than the general population)
- Ageing of households while in place, with disability acquisition evaluated at age transitions
- Persistence of disability status once acquired
- Monte Carlo aggregation across many dwellings

---

## How the model works

1. A fixed number of dwellings are simulated over a specified horizon (default: 50 years).
2. At the start of the simulation, a household moves into each dwelling.
3. Households are assigned to an age bracket based on the age distribution of recent in-movers.
4. Within each age bracket, households are randomly assigned an exact age.
5. Each household remains in the dwelling for a stochastic tenure duration drawn from observed tenure distributions.
6. If a household ages into an older age bracket while remaining in the dwelling, there is a chance that a person in the household acquires a disability.
7. Once a household contains a person with disability, that status persists for the remainder of their occupancy.
8. When a household moves out, a new household moves in until the simulation horizon is reached.
9. Results are aggregated across all dwellings.

---

## Running the model

Run from the repository root:

```
python3 run_from_excel.py
```

### Quick sanity-check run

```
python3 run_from_excel.py --n-props 1000
```

---

## Command-line parameters

- `--n-props`: Number of dwellings to simulate (default: 376000)
- `--horizon-years`: Simulation horizon in years (default: 50)
- `--seed`: Random seed for reproducibility
- `--input-excel`: Path to Excel file containing model inputs

To see all options:

```
python3 run_from_excel.py --help
```

---

## Outputs

Outputs are written to the `outputs/` directory and include scenario summaries and copies of key inputs used in the run.

---

## Intended use and limitations

This model is intended for housing policy analysis and long-run planning. It is not intended for individual-level prediction or short-term forecasting.

---

## Licence and citation

Add licence and citation details here.
