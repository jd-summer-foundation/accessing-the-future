import hashlib
from pathlib import Path

import pytest

from scripts.verify_data import verify_checksums


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_verify_checksums_accepts_matching_raw_entries(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_a = raw_dir / "sdac data.xlsx"
    file_b = raw_dir / "2. Housing mobility.xlsx"
    other_dir = tmp_path / "data/processed"
    other_dir.mkdir(parents=True, exist_ok=True)
    other_file = other_dir / "model_inputs.csv"

    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")
    other_file.write_text("ignore-me", encoding="utf-8")

    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(
        "\n".join(
            [
                f"{_sha256(file_a)}  data/raw/sdac data.xlsx",
                f"{_sha256(file_b)}  data/raw/2. Housing mobility.xlsx",
                f"{_sha256(other_file)}  data/processed/model_inputs.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    verified = verify_checksums(checksums, base_dir=tmp_path, prefix="data/raw")

    assert verified == [
        Path("data/raw/sdac data.xlsx"),
        Path("data/raw/2. Housing mobility.xlsx"),
    ]


def test_verify_checksums_rejects_mismatched_raw_entry(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    workbook = raw_dir / "sdac.xlsx"
    workbook.write_text("alpha", encoding="utf-8")

    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(
        f"{'0' * 64}  data/raw/sdac.xlsx\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_checksums(checksums, base_dir=tmp_path, prefix="data/raw")
