import numpy as np
import pandas as pd

from src.data_driven_cmcs import (
    FEATURE_SPECS,
    _bivariate_moran,
    _knn_neighbors,
    _weights_from_combined_evidence,
    prepare_weight_learning_table,
)


def test_weight_formula_normalizes_dimensions_and_subweights():
    evidence = {
        spec.feature: float(index + 1)
        for index, spec in enumerate(FEATURE_SPECS)
    }
    total = sum(evidence.values())
    evidence = {key: value / total for key, value in evidence.items()}
    weights, _ = _weights_from_combined_evidence(evidence)
    assert np.isclose(sum(weights.dimensions.values()), 1.0)
    for sub_weights in weights.sub_weights.values():
        assert np.isclose(sum(sub_weights.values()), 1.0)
    assert sum(weights.safety_bonus.values()) <= 0.15


def test_weight_learning_table_uses_unique_segments_and_no_accident_predictor():
    rows = []
    for edge_id in ("E1", "E2"):
        row = {
            "edge_id": edge_id,
            "segment_id": "S1",
            "district": "서구",
            "center_x": 1_000_000.0,
            "center_y": 1_800_000.0,
            "length_m": 100.0,
            "accident_count": 2.0,
        }
        for spec in FEATURE_SPECS:
            if spec.feature not in {
                "light_deficit",
                "crosswalk_deficit",
                "signal_deficit",
            }:
                row[spec.feature] = 0.2
        row.update(
            {
                "light_density_norm": 0.8,
                "has_crosswalk": 1,
                "has_signal": 1,
            }
        )
        rows.append(row)
    table = prepare_weight_learning_table(pd.DataFrame(rows))
    assert table.iloc[0]["segment_count"] == 1
    assert "accident_count_norm" not in table.columns


def test_bivariate_moran_detects_matching_spatial_pattern():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    neighbors = _knn_neighbors(coordinates, neighbors=1)
    values = np.array([0.0, 0.1, 0.9, 1.0])
    assert _bivariate_moran(values, values, neighbors) > 0
