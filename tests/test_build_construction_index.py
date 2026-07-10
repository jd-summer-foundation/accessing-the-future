import pandas as pd
import pytest

from scripts.build_construction_index import build_construction_index
from scripts.pipeline_utils import CONSTRUCTION_INDEX_COLUMNS, DEFAULT_CONSTRUCTION_INDEX_CSV, DEFAULT_PPI_WORKBOOK
from scripts.retrofit_cost_analysis import average_inflation


@pytest.fixture(scope="module")
def index_df() -> pd.DataFrame:
    return build_construction_index(DEFAULT_PPI_WORKBOOK)


def test_index_matches_known_abs_values(index_df: pd.DataFrame) -> None:
    assert list(index_df.columns) == CONSTRUCTION_INDEX_COLUMNS
    by_year = index_df.set_index("year")
    # Spot values from ABS 6427.0 Table 17, class 3011 House construction,
    # March quarters (series A2333673T / A2333757A).
    assert by_year.loc[2021, "nsw_index"] == pytest.approx(136.8)
    assert by_year.loc[2021, "wa_index"] == pytest.approx(125.4)
    assert by_year.loc[2026, "nsw_index"] == pytest.approx(189.3)
    assert by_year.loc[2026, "wa_index"] == pytest.approx(214.1)
    assert by_year.index.min() == 1999
    assert by_year.index.max() == 2026


def test_average_inflation_over_precovid_window(index_df: pd.DataFrame) -> None:
    by_year = index_df.set_index("year")
    nsw = average_inflation(by_year["nsw_index"], (2000, 2019))
    wa = average_inflation(by_year["wa_index"], (2000, 2019))
    assert nsw == pytest.approx(0.03734128109347932)
    assert wa == pytest.approx(0.04050211136245664)


def test_processed_csv_matches_rebuild(index_df: pd.DataFrame) -> None:
    committed = pd.read_csv(DEFAULT_CONSTRUCTION_INDEX_CSV)
    pd.testing.assert_frame_equal(committed, index_df, check_dtype=False)
