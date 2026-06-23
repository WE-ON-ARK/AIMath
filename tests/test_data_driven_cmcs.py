import numpy as np
import pandas as pd

from src.data_driven_cmcs import (
    FEATURE_SPECS,
    _bivariate_moran,
    _combine_evidence,
    _knn_neighbors,
    _normalize_scores,
    _spearman_evidence,
    _weights_from_combined_evidence,
    prepare_weight_learning_table,
)


def _make_edge_features(n_segments=10, n_edges_per_segment=2):
    rows = []
    for seg_idx in range(n_segments):
        for edge_idx in range(n_edges_per_segment):
            row = {
                "edge_id": f"E{seg_idx}_{edge_idx}",
                "segment_id": f"S{seg_idx}",
                "district": ["서구", "동구", "중구", "유성구", "대덕구"][seg_idx % 5],
                "center_x": 1_000_000.0 + seg_idx * 200,
                "center_y": 1_800_000.0 + seg_idx * 200,
                "length_m": 100.0 + seg_idx * 10,
                "accident_count": float(seg_idx % 3),
            }
            for spec in FEATURE_SPECS:
                if spec.feature not in {
                    "light_deficit", "crosswalk_deficit", "signal_deficit",
                }:
                    row[spec.feature] = 0.1 + 0.05 * seg_idx
            row.update({
                "light_density_norm": 0.8 - 0.03 * seg_idx,
                "has_crosswalk": int(seg_idx % 2 == 0),
                "has_signal": int(seg_idx % 3 == 0),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def test_weight_formula_normalizes_dimensions_and_subweights():
    evidence = {
        spec.feature: float(index + 1)
        for index, spec in enumerate(FEATURE_SPECS)
    }
    total = sum(evidence.values())
    evidence = {key: value / total for key, value in evidence.items()}
    weights, details = _weights_from_combined_evidence(evidence)
    assert np.isclose(sum(weights.dimensions.values()), 1.0)
    for sub_weights in weights.sub_weights.values():
        assert np.isclose(sum(sub_weights.values()), 1.0)
    assert sum(weights.safety_bonus.values()) <= 0.15
    assert "derivation_formula" in details


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


def test_normalize_scores_l1():
    scores = {"a": 0.3, "b": -0.1, "c": 0.7}
    normalized = _normalize_scores(scores)
    assert np.isclose(sum(normalized.values()), 1.0)
    assert normalized["b"] == 0.0
    assert normalized["c"] > normalized["a"]


def test_combine_evidence_averages_channels():
    features = [spec.feature for spec in FEATURE_SPECS]
    half = len(features) // 2
    m1_scores = {f: (1.0 if i < half else 0.0) for i, f in enumerate(features)}
    m2_scores = {f: (0.0 if i < half else 1.0) for i, f in enumerate(features)}
    combined, normalized = _combine_evidence({"m1": m1_scores, "m2": m2_scores})
    assert np.isclose(sum(combined.values()), 1.0)
    assert len(normalized) == 2


def test_spearman_evidence_positive_correlation():
    table = pd.DataFrame({
        "accident_count": [0, 1, 2, 3, 4, 5, 6, 7],
        "traffic_volume_norm": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "accident_label": [0, 0, 1, 1, 1, 1, 1, 1],
    })
    for spec in FEATURE_SPECS:
        if spec.feature != "traffic_volume_norm" and spec.feature not in table:
            table[spec.feature] = 0.5
    scores, details = _spearman_evidence(table)
    assert scores["traffic_volume_norm"] > 0
    assert details["traffic_volume_norm"]["rho"] > 0


def test_preprocessing_integrated_in_weight_learning():
    """prepare_weight_learning_table이 전처리를 거친다."""
    edge_features = _make_edge_features()
    edge_features.loc[0, "traffic_volume_norm"] = 1.5
    table = prepare_weight_learning_table(edge_features)
    preprocessing = table.attrs.get("preprocessing_report", {})
    assert preprocessing.get("input_rows", 0) > 0
    assert len(table) > 0


def test_end_to_end_weight_source():
    """합성된 가중치의 source가 data_driven_statistical이다."""
    evidence = {
        spec.feature: max(0.01, float(hash(spec.feature) % 100) / 100)
        for spec in FEATURE_SPECS
    }
    total = sum(evidence.values())
    evidence = {k: v / total for k, v in evidence.items()}
    weights, _ = _weights_from_combined_evidence(evidence)
    assert weights.source == "data_driven_statistical"
