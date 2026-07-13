import pandas as pd
import pytest

from scripts.build_dwelling_mix import build_dwelling_mix
from scripts.pipeline_utils import (
    DEFAULT_BUILDING_ACTIVITY_WORKBOOK,
    DEFAULT_DWELLING_MIX_CSV,
    DWELLING_MIX_COLUMNS,
)


@pytest.fixture(scope="module")
def mix_df() -> pd.DataFrame:
    return build_dwelling_mix(DEFAULT_BUILDING_ACTIVITY_WORKBOOK)


def test_mix_matches_known_abs_values(mix_df: pd.DataFrame) -> None:
    assert list(mix_df.columns) == DWELLING_MIX_COLUMNS
    by_key = mix_df.set_index(["state", "dwelling_type"])

    # Dwelling units commenced, ABS 8752.0 building-activity cube, New /
    # Total Sectors, summed over the 12 quarters 2023-Q2..2026-Q1.
    assert by_key.loc[("nsw", "house"), "commenced"] == 66332
    assert by_key.loc[("nsw", "townhouse"), "commenced"] == 30209
    assert by_key.loc[("nsw", "apartment"), "commenced"] == 44306
    assert by_key.loc[("wa", "house"), "commenced"] == 49578
    assert by_key.loc[("wa", "townhouse"), "commenced"] == 4429
    assert by_key.loc[("wa", "apartment"), "commenced"] == 5325

    assert by_key.loc[("nsw", "house"), "share"] == pytest.approx(0.471, abs=5e-4)
    assert by_key.loc[("nsw", "apartment"), "share"] == pytest.approx(0.315, abs=5e-4)
    assert by_key.loc[("wa", "house"), "share"] == pytest.approx(0.836, abs=5e-4)

    assert (mix_df["window_start_quarter"] == "2023-06").all()
    assert (mix_df["window_end_quarter"] == "2026-03").all()


def test_shares_sum_to_one_per_state(mix_df: pd.DataFrame) -> None:
    sums = mix_df.groupby("state")["share"].sum()
    assert sums.to_numpy() == pytest.approx([1.0, 1.0], abs=1e-12)
    # Shares must be internally consistent with the commencement counts.
    for state, group in mix_df.groupby("state"):
        expected = group["commenced"] / group["commenced"].sum()
        assert group["share"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-6)


def test_processed_csv_matches_rebuild(mix_df: pd.DataFrame) -> None:
    committed = pd.read_csv(DEFAULT_DWELLING_MIX_CSV)
    pd.testing.assert_frame_equal(committed, mix_df, check_dtype=False)
