import pandas as pd

from src.preprocessor import Preprocessor


def test_normalize_features_adds_zero_for_constant_column():
    frame = pd.DataFrame({"a": [10, 20, 30], "constant": [4, 4, 4]})
    result = Preprocessor.normalize_features(frame, ["a", "constant"])
    assert result["a_norm"].tolist() == [0.0, 0.5, 1.0]
    assert result["constant_norm"].tolist() == [0.0, 0.0, 0.0]

