#!/usr/bin/env python3
"""Verify checksums for canonical pipeline inputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_utils import REPO_ROOT, sha256_file

DEFAULT_CHECKSUMS = REPO_ROOT / "data/checksums.sha256"
DEFAULT_PREFIX = "data/raw"
CHECKSUM_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.*)$")


def parse_checksums(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = CHECKSUM_LINE.match(line)
            if match is None:
                raise ValueError(f"Invalid checksum line {line_number} in {path}: {raw_line.rstrip()}")
            entries.append((match.group(1).lower(), Path(match.group(2))))
    return entries


def _matches_prefix(relative_path: Path, prefix: str | None) -> bool:
    if prefix is None:
        return True
    clean_prefix = prefix.strip().strip("/")
    relative = relative_path.as_posix()
    return relative == clean_prefix or relative.startswith(f"{clean_prefix}/")


def verify_checksums(
    checksums_path: Path = DEFAULT_CHECKSUMS,
    *,
    base_dir: Path = REPO_ROOT,
    prefix: str | None = DEFAULT_PREFIX,
) -> list[Path]:
    matched_entries = [
        (expected, relative_path)
        for expected, relative_path in parse_checksums(checksums_path)
        if _matches_prefix(relative_path, prefix)
    ]
    if not matched_entries:
        scope = f" for prefix {prefix!r}" if prefix else ""
        raise ValueError(f"No checksum entries found in {checksums_path}{scope}")

    verified_paths: list[Path] = []
    failures: list[str] = []
    for expected_digest, relative_path in matched_entries:
        target_path = base_dir / relative_path
        if not target_path.exists():
            failures.append(f"Missing file: {relative_path}")
            continue
        actual_digest = sha256_file(target_path)
        if actual_digest != expected_digest:
            failures.append(
                f"Checksum mismatch for {relative_path}: expected {expected_digest}, got {actual_digest}"
            )
            continue
        verified_paths.append(relative_path)

    if failures:
        raise ValueError("Checksum verification failed:\n- " + "\n- ".join(failures))

    return verified_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checksums for canonical pipeline inputs.")
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS, help="Path to checksum manifest.")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Restrict verification to checksum entries under this relative path prefix. Use an empty string to verify all entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix or None
    verified_paths = verify_checksums(args.checksums, prefix=prefix)
    scope = prefix or "."
    print(f"Verified {len(verified_paths)} file(s) under {scope} using {args.checksums}")


if __name__ == "__main__":
    main()
