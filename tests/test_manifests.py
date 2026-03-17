from pathlib import Path

from scripts.pipeline_utils import REPO_ROOT, portable_path


def test_portable_path_relativises_repo_paths() -> None:
    assert portable_path(REPO_ROOT / "configs/baseline.yaml") == "configs/baseline.yaml"


def test_portable_path_preserves_external_paths(tmp_path: Path) -> None:
    external = tmp_path / "outside.csv"
    assert portable_path(external) == str(external)
