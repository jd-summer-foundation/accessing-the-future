import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.build_model_inputs import build_model_inputs

from scripts.pipeline_utils import DEFAULT_DERIVATION_CONFIG

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SMOKE = ROOT / "tests/fixtures/expected_smoke_summaries.csv"
EXPECTED_BASELINE = ROOT / "tests/fixtures/expected_baseline_summaries.csv"


def _build_temp_input(tmp_path: Path) -> Path:
    df = build_model_inputs(DEFAULT_DERIVATION_CONFIG)
    input_path = tmp_path / "model_inputs.csv"
    df.to_csv(input_path, index=False)
    return input_path


def _run_config(tmp_path: Path, config_path: Path, output_name: str) -> Path:
    input_path = _build_temp_input(tmp_path)
    output_dir = tmp_path / output_name
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_from_excel.py"),
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--outdir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )
    return output_dir


def test_smoke_regression_matches_expected_fixture(tmp_path: Path) -> None:
    output_dir = _run_config(tmp_path, ROOT / "configs/smoke.yaml", "run")
    actual = pd.read_csv(output_dir / "scenario_summaries.csv")
    expected = pd.read_csv(EXPECTED_SMOKE)
    pd.testing.assert_frame_equal(actual, expected)


def test_smoke_run_is_deterministic(tmp_path: Path) -> None:
    first = _run_config(tmp_path, ROOT / "configs/smoke.yaml", "run_one")
    second = _run_config(tmp_path, ROOT / "configs/smoke.yaml", "run_two")
    assert (first / "scenario_summaries.csv").read_text() == (second / "scenario_summaries.csv").read_text()
    assert (first / "run_manifest.json").exists()
    assert (second / "run_manifest.json").exists()


def test_baseline_regression_matches_expected_fixture(tmp_path: Path) -> None:
    output_dir = _run_config(tmp_path, ROOT / "configs/baseline.yaml", "baseline")
    actual = pd.read_csv(output_dir / "scenario_summaries.csv")
    expected = pd.read_csv(EXPECTED_BASELINE)
    pd.testing.assert_frame_equal(actual, expected)

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["config"]["path"] == "configs/baseline.yaml"
    assert manifest["outputs"]["scenario_summaries"]["path"] == "scenario_summaries.csv"
    assert sorted(manifest["outputs"]["artifact_checksums"].keys()) == [
        "inputs_used.csv",
        "profiles_used.csv",
        "scenario_summaries.csv",
    ]
