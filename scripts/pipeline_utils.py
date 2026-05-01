#!/usr/bin/env python3
"""Shared helpers for the raw-to-results research workflow."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

import au_housing_disability_monte_carlo as eng

REPO_ROOT = Path(__file__).resolve().parents[1]

AGE_COL = "Age of Reference Person (Years)"
RATE_ANY_COL = "DIP_any"
RATE_MOTOR_COL = "DIP_physical"
RATE_SEVERE_COL = "DIP_severe"
RATE_PHYS2_COL = "DIP_physical2"
RATE_MOE_SUFFIX = "_moe"
GENPOP_COL = "Age distribution"
INMOVER_COL = "Of those who have length of tenure <1 year, what proportion are in each age bucket?"

TENURE_BUCKETS = ["<1", "1-2", "2-3", "3-4", "5-9", "10-19", "20+"]
TENURE_COLUMNS = [
    "Length of tenure: less than 1 year  (% of all households, not count)",
    "Length of tenure: 1-2 years",
    "Length of tenure: 2-3 years",
    "Length of tenure: 3-4 years",
    "Length of tenure: 5-9 years",
    "Length of tenure: 10-19 years",
    "Length of tenure: 20+ years",
]

MODEL_INPUT_COLUMNS = [
    AGE_COL,
    RATE_ANY_COL,
    RATE_MOTOR_COL,
    RATE_SEVERE_COL,
    RATE_PHYS2_COL,
    GENPOP_COL,
    *TENURE_COLUMNS,
    INMOVER_COL,
]

RATE_COLUMNS = [RATE_ANY_COL, RATE_MOTOR_COL, RATE_SEVERE_COL, RATE_PHYS2_COL]
RATE_MOE_COLUMNS = [f"{column}{RATE_MOE_SUFFIX}" for column in RATE_COLUMNS]

DEFAULT_RAW_WORKBOOK = REPO_ROOT / "data/raw/sdac22_household_disability.xlsx"
DEFAULT_HOUSING_MOBILITY_WORKBOOK = REPO_ROOT / "data/raw/2. Housing mobility.xlsx"
DEFAULT_MODEL_INPUT_CSV = REPO_ROOT / "data/processed/model_inputs.csv"
DEFAULT_MODEL_INPUT_XLSX = REPO_ROOT / "data/processed/model_inputs.xlsx"
DEFAULT_LEGACY_ORACLE = REPO_ROOT / "data/processed/legacy_model_inputs.xlsx"
DEFAULT_DERIVATION_CONFIG = REPO_ROOT / "configs/derivation.yaml"
DEFAULT_BASELINE_CONFIG = REPO_ROOT / "configs/baseline.yaml"
DEFAULT_SMOKE_CONFIG = REPO_ROOT / "configs/smoke.yaml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results/baseline"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

DIRECT_DEPENDENCIES = [
    "numpy",
    "pandas",
    "openpyxl",
    "PyYAML",
    "matplotlib",
    "pytest",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, found {type(data).__name__}")
    return data


def read_tabular_data(path: Path, sheet: int | str = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    raise ValueError(f"Unsupported input format for {path}")


def canonicalise_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    optional_columns = [column for column in RATE_MOE_COLUMNS if column in df.columns]
    out = df.loc[:, [*MODEL_INPUT_COLUMNS, *optional_columns]].copy()
    out[AGE_COL] = out[AGE_COL].astype(str)
    return out


def _validate_probability_series(values: pd.Series, name: str, *, expected_total: float, tolerance: float) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} contains non-numeric values")
    if (numeric < 0).any():
        raise ValueError(f"{name} contains negative values")
    total = float(numeric.sum())
    if abs(total - expected_total) > tolerance:
        raise ValueError(f"{name} sums to {total:.6f}; expected approximately {expected_total:.6f}")


def validate_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = canonicalise_model_inputs(df)

    brackets = out[AGE_COL].tolist()
    if brackets != eng.BRACKETS:
        raise ValueError(f"Age brackets must be {eng.BRACKETS}, found {brackets}")

    percent_like = [RATE_ANY_COL, RATE_MOTOR_COL, RATE_SEVERE_COL, RATE_PHYS2_COL, *TENURE_COLUMNS]
    percent_like.extend(column for column in RATE_MOE_COLUMNS if column in out.columns)
    for column in percent_like:
        numeric = pd.to_numeric(out[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{column} contains non-numeric values")
        if (numeric < 0).any():
            raise ValueError(f"{column} contains negative values")
        if (numeric > 100).any():
            raise ValueError(f"{column} contains values above 100")

    gen_dist = pd.to_numeric(out[GENPOP_COL], errors="coerce")
    if gen_dist.isna().any() or (gen_dist <= 0).any():
        raise ValueError(f"{GENPOP_COL} must contain positive numeric values")

    for _, row in out.iterrows():
        _validate_probability_series(
            row[TENURE_COLUMNS],
            f"tenure distribution for {row[AGE_COL]}",
            expected_total=100.0,
            tolerance=1.5,
        )

    _validate_probability_series(
        out[INMOVER_COL],
        INMOVER_COL,
        expected_total=1.0,
        tolerance=0.01,
    )
    return out


def load_model_inputs(path: Path, sheet: int | str = 0, *, validate: bool = True) -> pd.DataFrame:
    df = read_tabular_data(path, sheet=sheet)
    return validate_model_inputs(df) if validate else canonicalise_model_inputs(df)


def resolve_input_path(requested: Path | None = None) -> Path:
    if requested is not None:
        return requested
    candidates = [
        DEFAULT_MODEL_INPUT_CSV,
        DEFAULT_MODEL_INPUT_XLSX,
        DEFAULT_LEGACY_ORACLE,
        REPO_ROOT / "inputs/Data for modelling.xlsx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No model input file found in canonical or legacy locations.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, *, base_dir: Path = REPO_ROOT) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def manifest_path(path: Path, *, relative_to: Path | None = None) -> str:
    if relative_to is not None:
        try:
            return path.relative_to(relative_to).as_posix()
        except ValueError:
            pass
    return portable_path(path)


def artifact_checksums(paths: list[Path], *, relative_to: Path | None = None) -> Dict[str, str]:
    return {
        manifest_path(path, relative_to=relative_to): sha256_file(path)
        for path in paths
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def dependency_versions(packages: list[str] | None = None) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package in packages or DIRECT_DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def serialise_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {str(key): serialise_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialise_for_json(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialise_for_json(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def compare_model_frames(left: pd.DataFrame, right: pd.DataFrame, *, tolerance: float = 1e-9) -> list[str]:
    left_canon = canonicalise_model_inputs(left).reset_index(drop=True)
    right_canon = canonicalise_model_inputs(right).reset_index(drop=True)
    if left_canon.shape != right_canon.shape:
        return [f"Shape mismatch: {left_canon.shape} != {right_canon.shape}"]

    diffs: list[str] = []
    for column in MODEL_INPUT_COLUMNS:
        if column == AGE_COL:
            if not left_canon[column].equals(right_canon[column]):
                diffs.append(f"Column {column} differs")
            continue

        left_numeric = pd.to_numeric(left_canon[column], errors="coerce")
        right_numeric = pd.to_numeric(right_canon[column], errors="coerce")
        delta = (left_numeric - right_numeric).abs()
        mismatch_idx = delta[delta > tolerance].index.tolist()
        if mismatch_idx:
            idx = mismatch_idx[0]
            diffs.append(
                f"Column {column} differs at row {idx}: "
                f"{left_numeric.iloc[idx]} != {right_numeric.iloc[idx]}"
            )
    return diffs


def print_stderr(message: str) -> None:
    print(message, file=sys.stderr)
