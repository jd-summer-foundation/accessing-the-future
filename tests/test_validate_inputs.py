import pytest

from scripts.pipeline_utils import AGE_COL, DEFAULT_MODEL_INPUT_CSV, eng, load_model_inputs, validate_model_inputs


def test_validate_model_inputs_accepts_canonical_csv() -> None:
    df = load_model_inputs(DEFAULT_MODEL_INPUT_CSV)
    assert df[AGE_COL].tolist() == eng.BRACKETS


def test_validate_model_inputs_rejects_invalid_age_bracket() -> None:
    df = load_model_inputs(DEFAULT_MODEL_INPUT_CSV)
    df.loc[0, AGE_COL] = "invalid"
    with pytest.raises(ValueError, match="Age brackets"):
        validate_model_inputs(df)
