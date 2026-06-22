import numpy as np
import pandas as pd

from src.full_pipeline import (
    EDGE_MODEL_FEATURES,
    _aggregate_regional_features,
    _build_edge_model_candidates,
    _edge_metrics,
    _highway_value,
    _optimal_f1_threshold,
    _parse_number,
    _select_best_tree_model,
    canonical_segment_id,
)


def test_canonical_segment_id_is_direction_independent():
    assert canonical_segment_id(10, 20, 123) == canonical_segment_id(20, 10, 123)


def test_parse_osm_tag_values():
    assert _parse_number("50 mph", 30) == 50
    assert _parse_number(["2", "3"], 1) == 2
    assert _highway_value(["residential", "service"]) == "residential"


def test_edge_model_candidates_include_requested_models():
    candidates = _build_edge_model_candidates(
        positive_count=10, negative_count=90, random_state=42
    )
    assert {"LogisticRegression", "RandomForest", "XGBoost"} == set(candidates)
    assert candidates["XGBoost"].named_steps["model"].scale_pos_weight == 9


def test_optimal_threshold_is_derived_from_oof_probabilities():
    target = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.05, 0.10, 0.40, 0.45, 0.90])
    threshold = _optimal_f1_threshold(target, probability)
    assert 0.4 < threshold <= 0.45


def test_tree_model_selection_treats_near_equal_ap_as_tie():
    metrics = {
        "RandomForest": {
            "average_precision": 0.1555,
            "roc_auc": 0.68,
            "brier_score": 0.17,
            "optimized_threshold": {"f1": 0.25},
        },
        "XGBoost": {
            "average_precision": 0.1561,
            "roc_auc": 0.66,
            "brier_score": 0.172,
            "optimized_threshold": {"f1": 0.24},
        },
    }
    assert _select_best_tree_model(metrics) == "RandomForest"


def test_edge_metrics_reports_optimized_f1_target():
    target = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.10, 0.60, 0.55, 0.70, 0.90])
    metrics = _edge_metrics(target, probability)
    assert metrics["optimized_threshold"]["f1"] >= 0.75
    assert metrics["optimized_threshold"]["threshold"] != 0.5


def test_regional_cells_do_not_cross_district_boundaries():
    rows = []
    for district, segment_id, accident_count in [
        ("서구", "S-1", 1),
        ("유성구", "S-2", 0),
    ]:
        row = {
            "district": district,
            "segment_id": segment_id,
            "center_x": 1000.0,
            "center_y": 1000.0,
            "accident_count": accident_count,
        }
        row.update({feature: 0.0 for feature in EDGE_MODEL_FEATURES})
        rows.append(row)

    _, regions = _aggregate_regional_features(
        pd.DataFrame(rows),
        cell_size_m=1500,
    )

    assert len(regions) == 2
    assert set(regions["district"]) == {"서구", "유성구"}
