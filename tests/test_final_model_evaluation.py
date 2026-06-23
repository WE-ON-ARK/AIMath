import numpy as np
import pandas as pd

from src.final_model_evaluation import (
    _expected_calibration_error,
    _markdown_table,
)


def test_expected_calibration_error_is_zero_for_perfect_bins():
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.0, 0.0, 1.0, 1.0])
    assert _expected_calibration_error(target, probability, bins=2) == 0.0


def test_markdown_table_does_not_require_optional_dependency():
    table = _markdown_table(pd.DataFrame([{"모델": "XGBoost", "F1": 0.53125}]))
    assert "| 모델 | F1 |" in table
    assert "0.531" in table
