#!/usr/bin/env python3
"""Generate manuscript-ready tables and figures from scenario summaries."""

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

from scripts.pipeline_utils import DEFAULT_REPORTS_DIR, DEFAULT_RESULTS_DIR, utc_now_iso, write_json
from scripts.pipeline_utils import artifact_checksums, manifest_path, sha256_file

TABLE_NAME = "table_01_scenario_summary"
FIGURE_1_NAME = "figure_01_ever_probabilities"
FIGURE_2_NAME = "figure_02_time_share"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tables and figures from a model run.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing scenario_summaries.csv.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory where tables and figures will be written.")
    return parser.parse_args()


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _scenario_label(name: str) -> str:
    return name.replace("_", " ")


def _write_summary_table(df: pd.DataFrame, tables_dir: Path) -> list[Path]:
    table_csv = tables_dir / f"{TABLE_NAME}.csv"
    table_md = tables_dir / f"{TABLE_NAME}.md"

    display = df.copy()
    for column in [c for c in display.columns if c != "scenario"]:
        display[column] = display[column].map(_format_percent)
    display.to_csv(table_csv, index=False)

    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    table_md.write_text("\n".join([header, divider, *rows]) + "\n", encoding="utf-8")
    return [table_csv, table_md]


def _grouped_bar(df: pd.DataFrame, metrics: list[str], title: str, ylabel: str, output_path: Path) -> None:
    labels = [_scenario_label(name) for name in df["scenario"]]
    width = 0.18
    positions = range(len(labels))
    palette = ["#0B3954", "#BFD7EA", "#FF6663", "#E0FF4F"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        offset = [pos + (idx - (len(metrics) - 1) / 2) * width for pos in positions]
        ax.bar(offset, df[metric], width=width, label=metric.replace("_", " "), color=palette[idx % len(palette)])

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_reports(results_dir: Path, reports_dir: Path) -> dict[str, object]:
    scenario_path = results_dir / "scenario_summaries.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario summary file: {scenario_path}")

    df = pd.read_csv(scenario_path)
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    generated_paths.extend(_write_summary_table(df, tables_dir))

    figure_1 = figures_dir / f"{FIGURE_1_NAME}.png"
    figure_2 = figures_dir / f"{FIGURE_2_NAME}.png"
    _grouped_bar(
        df,
        ["p_ever_any", "p_ever_severe", "p_ever_physical", "p_ever_physical2"],
        "Probability a dwelling ever hosts a relevant household type",
        "Probability over 50-year horizon",
        figure_1,
    )
    _grouped_bar(
        df,
        ["pct_time_any", "pct_time_severe", "pct_time_physical", "pct_time_physical2"],
        "Average share of dwelling occupancy time by household type",
        "Share of occupied time",
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
    write_json(reports_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate_reports(args.results_dir, args.reports_dir)
    print(f"Generated {len(manifest['artifacts'])} report artifacts in {args.reports_dir}")


if __name__ == "__main__":
    main()
