import json
from pathlib import Path
import subprocess

import pytest

from scripts.check_release import validate_release

ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_minimal_release_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write_text(
        repo / "README.md",
        "# Accessing the Future\n\nRun `make reproduce` and `make release-check` before release.\n",
    )
    _write_text(
        repo / "docs/release.md",
        "# Release Checklist\n\nRun `make release-check`.\n",
    )
    _write_text(
        repo / "CITATION.cff",
        "\n".join(
            [
                'cff-version: 1.2.0',
                'title: "Accessing the Future: Reproducible Monte Carlo Housing Disability Model"',
                'version: "0.1.0"',
                "authors:",
                "  - family-names: Dignam",
                "    given-names: Joel",
            ]
        )
        + "\n",
    )
    _write_json(
        repo / ".zenodo.json",
        {
            "title": "Accessing the Future: Reproducible Monte Carlo Housing Disability Model",
            "creators": [{"name": "Dignam, Joel"}],
        },
    )
    _write_text(
        repo / "pyproject.toml",
        "\n".join(
            [
                "[project]",
                'version = "0.1.0"',
                'readme = "README.md"',
            ]
        )
        + "\n",
    )
    _write_text(repo / "data/checksums.sha256", "abc\n")
    for relative_path in [
        "results/baseline/scenario_summaries.csv",
        "results/baseline/inputs_used.csv",
        "results/baseline/profiles_used.csv",
        "reports/tables/table_01_scenario_summary.csv",
        "reports/tables/table_01_scenario_summary.md",
        "reports/figures/figure_01_ever_probabilities.png",
        "reports/figures/figure_02_time_share.png",
    ]:
        _write_text(repo / relative_path, "artifact\n")
    _write_json(
        repo / "results/baseline/run_manifest.json",
        {
            "run_name": "baseline",
            "config": {"path": "configs/baseline.yaml"},
            "outputs": {
                "artifact_checksums": {
                    "inputs_used.csv": "a",
                    "profiles_used.csv": "b",
                    "scenario_summaries.csv": "c",
                }
            },
        },
    )
    _write_json(
        repo / "reports/manifest.json",
        {
            "artifacts": [
                "figures/figure_01_ever_probabilities.png",
                "figures/figure_02_time_share.png",
                "tables/table_01_scenario_summary.csv",
                "tables/table_01_scenario_summary.md",
            ]
        },
    )
    return repo


def test_validate_release_accepts_current_repo() -> None:
    subprocess.run(["make", "reproduce"], cwd=ROOT, check=True)
    validate_release(ROOT)


def test_validate_release_rejects_metadata_mismatch(tmp_path: Path) -> None:
    repo = _make_minimal_release_repo(tmp_path)
    _write_json(
        repo / ".zenodo.json",
        {
            "title": "Different title",
            "creators": [{"name": "Dignam, Joel"}],
        },
    )

    with pytest.raises(ValueError, match="title must match"):
        validate_release(repo)
