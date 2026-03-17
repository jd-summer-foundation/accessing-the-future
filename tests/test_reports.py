import shutil
from pathlib import Path

from scripts.generate_reports import generate_reports

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SMOKE = ROOT / "tests/fixtures/expected_smoke_summaries.csv"


def test_generate_reports_creates_expected_artifacts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(EXPECTED_SMOKE, results_dir / "scenario_summaries.csv")

    manifest = generate_reports(results_dir, reports_dir)

    assert sorted(manifest["artifacts"]) == [
        "figures/figure_01_ever_probabilities.png",
        "figures/figure_02_time_share.png",
        "tables/table_01_scenario_summary.csv",
        "tables/table_01_scenario_summary.md",
    ]
    assert sorted(manifest["artifact_checksums"].keys()) == sorted(manifest["artifacts"])
    assert manifest["source_summary"]["path"] == "scenario_summaries.csv"
    assert (reports_dir / "manifest.json").exists()
