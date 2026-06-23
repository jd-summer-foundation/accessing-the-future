#!/usr/bin/env python3
"""Generate the manuscript table and figures from scenario summaries.

These are the artifacts used as Table 1 and Figures 1-2 in the paper. They
differ from scripts/generate_reports.py: rather than drawing one bar per
scenario/uncertainty row, this groups the low/base/high cases into a single
base-case estimate with a low-high error bar per scenario, and writes a compact
"base [low-high]" summary table.

Outputs (under <reports-dir>/manuscript/):
- table_scenario_summary.md, table_scenario_summary.csv
- figure_01_ever_probabilities.png
- figure_02_time_share.png
- manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_utils import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_RESULTS_DIR,
    artifact_checksums,
    manifest_path,
    sha256_file,
    utc_now_iso,
    write_json,
)

# Base scenario name -> display label, in the manuscript's display order.
SCENARIOS = [
    ("main_inmover_first", "Central\n(no trend)"),
    ("sdac_2003_2022_trend", "Trend\n2003–2022"),
    ("sdac_2015_2022_trend", "Trend\n2015–2022"),
    ("sens_disabled_tenure_1p2", "Longer tenure\n(×1.2)"),
    ("selection_scale_0p9", "Self-sorting\n(×0.9)"),
]

DARK, LIGHT = "#0B3954", "#7FA8C9"

TABLE_NAME = "table_scenario_summary"
FIGURE_1_NAME = "figure_01_ever_probabilities"
FIGURE_2_NAME = "figure_02_time_share"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript table and figures from a model run.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing scenario_summaries.csv.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory under which a manuscript/ folder is written.")
    return parser.parse_args()


def _load_cases(df: pd.DataFrame) -> dict:
    """Return {base_scenario: {case: row}} for each manuscript scenario."""
    cases: dict = {}
    for base, _ in SCENARIOS:
        by_case = {}
        for case in ("low", "base", "high"):
            match = df.loc[df["scenario"] == f"{base}_{case}"]
            if match.empty:
                raise ValueError(
                    f"Missing scenario row '{base}_{case}' in scenario_summaries.csv. "
                    f"Run the full baseline (make run-baseline) before generating manuscript artifacts."
                )
            by_case[case] = match.iloc[0]
        cases[base] = by_case
    return cases


def _series(cases: dict, metric: str) -> tuple[list[float], list[list[float]]]:
    """Return (base values, [lower errors, upper errors]) across scenarios."""
    base = [float(cases[b]["base"][metric]) for b, _ in SCENARIOS]
    lower = [float(cases[b]["base"][metric] - cases[b]["low"][metric]) for b, _ in SCENARIOS]
    upper = [float(cases[b]["high"][metric] - cases[b]["base"][metric]) for b, _ in SCENARIOS]
    return base, [lower, upper]


def _grouped_bar(cases: dict, metric_phys: str, metric_any: str, title: str, ylabel: str, ymax: float, output_path: Path) -> None:
    labels = [label for _, label in SCENARIOS]
    positions = range(len(labels))
    width = 0.38
    phys_base, phys_err = _series(cases, metric_phys)
    any_base, any_err = _series(cases, metric_any)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    xp = [p - width / 2 for p in positions]
    xa = [p + width / 2 for p in positions]
    ax.bar(xp, phys_base, width, yerr=phys_err, capsize=4, color=DARK, label="Physical disability", error_kw=dict(ecolor="#444", lw=1.1))
    ax.bar(xa, any_base, width, yerr=any_err, capsize=4, color=LIGHT, label="Any disability", error_kw=dict(ecolor="#444", lw=1.1))
    for x, value, err in zip(xp, phys_base, phys_err[1]):
        ax.text(x, value + err + ymax * 0.018, f"{value * 100:.0f}%", ha="center", va="bottom", fontsize=9, color=DARK, fontweight="bold")
    for x, value, err in zip(xa, any_base, any_err[1]):
        ax.text(x, value + err + ymax * 0.018, f"{value * 100:.0f}%", ha="center", va="bottom", fontsize=9, color="#33617f")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0.0, ymax)
    ax.set_yticks([tick / 10 for tick in range(0, int(ymax * 10) + 1)])
    ax.set_yticklabels([f"{int(tick * 100)}%" for tick in ax.get_yticks()])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_table(cases: dict, manuscript_dir: Path) -> list[Path]:
    columns = [
        ("p_ever_physical", "Ever — physical disability"),
        ("p_ever_any", "Ever — any disability"),
        ("pct_time_physical", "Time share — physical"),
        ("pct_time_any", "Time share — any"),
    ]

    def cell(base: str, metric: str) -> str:
        low = float(cases[base]["low"][metric]) * 100
        mid = float(cases[base]["base"][metric]) * 100
        high = float(cases[base]["high"][metric]) * 100
        return f"{mid:.1f}% [{low:.1f}–{high:.1f}]"

    header = ["Scenario"] + [title for _, title in columns]
    rows = [[label.replace("\n", " ")] + [cell(base, metric) for metric, _ in columns] for base, label in SCENARIOS]

    table_md = manuscript_dir / f"{TABLE_NAME}.md"
    table_csv = manuscript_dir / f"{TABLE_NAME}.csv"
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    md_lines += ["| " + " | ".join(row) + " |" for row in rows]
    table_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows, columns=header).to_csv(table_csv, index=False)
    return [table_csv, table_md]


def generate(results_dir: Path, reports_dir: Path) -> dict:
    scenario_path = results_dir / "scenario_summaries.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario summary file: {scenario_path}")

    df = pd.read_csv(scenario_path)
    cases = _load_cases(df)

    manuscript_dir = reports_dir / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    generated_paths.extend(_write_table(cases, manuscript_dir))

    figure_1 = manuscript_dir / f"{FIGURE_1_NAME}.png"
    figure_2 = manuscript_dir / f"{FIGURE_2_NAME}.png"
    _grouped_bar(
        cases,
        "p_ever_physical",
        "p_ever_any",
        "Probability a new dwelling is ever occupied by such a household (20-year horizon)",
        "Probability",
        0.95,
        figure_1,
    )
    _grouped_bar(
        cases,
        "pct_time_physical",
        "pct_time_any",
        "Average share of the 20-year horizon occupied by such a household",
        "Share of time",
        0.55,
        figure_2,
    )
    generated_paths.extend([figure_1, figure_2])

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "results_dir": results_dir,
        "reports_dir": reports_dir,
        "source_summary": {
            "path": manifest_path(scenario_path, relative_to=results_dir),
            "sha256": sha256_file(scenario_path),
        },
        "artifacts": [str(path.relative_to(reports_dir)) for path in generated_paths],
        "artifact_checksums": artifact_checksums(generated_paths, relative_to=reports_dir),
    }
    write_json(manuscript_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate(args.results_dir, args.reports_dir)
    print(f"Generated {len(manifest['artifacts'])} manuscript artifacts in {Path(args.reports_dir) / 'manuscript'}")


if __name__ == "__main__":
    main()
