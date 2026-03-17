#!/usr/bin/env python3
"""Validate release metadata and canonical archived artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_BASELINE_ARTIFACTS = [
    "results/baseline/scenario_summaries.csv",
    "results/baseline/inputs_used.csv",
    "results/baseline/profiles_used.csv",
    "results/baseline/run_manifest.json",
    "reports/tables/table_01_scenario_summary.csv",
    "reports/tables/table_01_scenario_summary.md",
    "reports/figures/figure_01_ever_probabilities.png",
    "reports/figures/figure_02_time_share.png",
    "reports/manifest.json",
    "data/checksums.sha256",
    "CITATION.cff",
    ".zenodo.json",
]

EXPECTED_REPORT_ARTIFACTS = [
    "figures/figure_01_ever_probabilities.png",
    "figures/figure_02_time_share.png",
    "tables/table_01_scenario_summary.csv",
    "tables/table_01_scenario_summary.md",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected TOML table in {path}")
    return data


def _ensure(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_release(repo_root: Path) -> None:
    citation_path = repo_root / "CITATION.cff"
    zenodo_path = repo_root / ".zenodo.json"
    pyproject_path = repo_root / "pyproject.toml"
    readme_path = repo_root / "README.md"
    release_doc_path = repo_root / "docs/release.md"
    baseline_manifest_path = repo_root / "results/baseline/run_manifest.json"
    reports_manifest_path = repo_root / "reports/manifest.json"

    citation = _load_yaml(citation_path)
    zenodo = _load_json(zenodo_path)
    pyproject = _load_toml(pyproject_path)
    baseline_manifest = _load_json(baseline_manifest_path)
    reports_manifest = _load_json(reports_manifest_path)
    readme_text = readme_path.read_text(encoding="utf-8")
    release_doc_text = release_doc_path.read_text(encoding="utf-8")

    project = pyproject.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("Missing [project] table in pyproject.toml")

    citation_title = citation.get("title")
    zenodo_title = zenodo.get("title")
    citation_version = citation.get("version")
    pyproject_version = project.get("version")
    citation_authors = citation.get("authors", [])
    zenodo_creators = zenodo.get("creators", [])
    baseline_outputs = baseline_manifest.get("outputs", {})

    failures: list[str] = []

    _ensure(citation_title == zenodo_title, "CITATION.cff title must match .zenodo.json title", failures)
    _ensure(citation_version == pyproject_version, "CITATION.cff version must match pyproject.toml project.version", failures)
    _ensure(project.get("readme") == "README.md", "pyproject.toml must declare README.md as the project readme", failures)
    _ensure(bool(citation_authors), "CITATION.cff must include at least one author", failures)
    _ensure(bool(zenodo_creators), ".zenodo.json must include at least one creator", failures)
    if citation_authors and zenodo_creators:
        first_author = citation_authors[0]
        first_creator = zenodo_creators[0]
        expected_name = f"{first_author.get('family-names')}, {first_author.get('given-names')}"
        _ensure(first_creator.get("name") == expected_name, "First Zenodo creator must match first CITATION author", failures)

    _ensure("# Accessing the Future" in readme_text, "README.md must include the project title heading", failures)
    _ensure("make reproduce" in readme_text, "README.md must document make reproduce", failures)
    _ensure("make release-check" in readme_text, "README.md must document make release-check", failures)
    _ensure("make release-check" in release_doc_text, "docs/release.md must include the release-check command", failures)

    for relative_path in EXPECTED_BASELINE_ARTIFACTS:
        _ensure((repo_root / relative_path).exists(), f"Required release artifact is missing: {relative_path}", failures)

    _ensure(baseline_manifest.get("run_name") == "baseline", "Baseline run manifest must record run_name=baseline", failures)
    _ensure(baseline_manifest.get("config", {}).get("path") == "configs/baseline.yaml", "Baseline run manifest must point to configs/baseline.yaml", failures)
    _ensure(sorted(reports_manifest.get("artifacts", [])) == EXPECTED_REPORT_ARTIFACTS, "reports/manifest.json must list the canonical report artifacts", failures)

    if isinstance(baseline_outputs, dict):
        checksums = baseline_outputs.get("artifact_checksums", {})
        expected_output_keys = ["inputs_used.csv", "profiles_used.csv", "scenario_summaries.csv"]
        _ensure(sorted(checksums.keys()) == expected_output_keys, "Baseline run manifest must checksum the three canonical run artifacts", failures)
    else:
        failures.append("Baseline run manifest outputs section must be a mapping")

    if failures:
        raise ValueError("Release check failed:\n- " + "\n- ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate release metadata and canonical archived artifacts.")
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="Repository root to validate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_release(args.repo_root.resolve())
    print(f"Release metadata and artifacts validated in {args.repo_root.resolve()}")


if __name__ == "__main__":
    main()
