import pandas as pd
import pytest

from src.cmcs_calculator import CMCSCalculator, CMCSWeights


def test_safer_infrastructure_reduces_cmcs():
    features = pd.DataFrame(
        [
            {
                "accident_count_norm": 0.8,
                "traffic_volume_norm": 0.8,
                "avg_speed_norm": 0.8,
                "narrow_sidewalk_norm": 0.8,
                "illegal_parking_norm": 0.8,
                "light_density_norm": 0.2,
                "has_crosswalk": 0,
                "has_signal": 0,
                "lane_count_norm": 0.8,
                "has_speed_bump": 0,
                "has_cctv": 0,
                "is_school_zone": 0,
            },
            {
                "accident_count_norm": 0.1,
                "traffic_volume_norm": 0.1,
                "avg_speed_norm": 0.1,
                "narrow_sidewalk_norm": 0.1,
                "illegal_parking_norm": 0.1,
                "light_density_norm": 0.9,
                "has_crosswalk": 1,
                "has_signal": 1,
                "lane_count_norm": 0.1,
                "has_speed_bump": 1,
                "has_cctv": 1,
                "is_school_zone": 1,
            },
        ]
    )
    scores = CMCSCalculator().calculate_cmcs_ahp(features)
    assert 0 <= scores.iloc[1] < scores.iloc[0] <= 1


def test_invalid_dimension_weights_are_rejected():
    weights = CMCSWeights(
        dimensions={
            "risk": 0.5,
            "discomfort": 0.5,
            "congestion": 0.5,
            "obstruction": 0.0,
            "crossing": 0.0,
        }
    )
    with pytest.raises(ValueError):
        CMCSCalculator(weights)


def test_legacy_ahp_name_is_only_compatibility_alias():
    features = pd.DataFrame([{"traffic_volume_norm": 0.5}])
    calculator = CMCSCalculator()
    assert calculator.calculate_cmcs(features).equals(
        calculator.calculate_cmcs_ahp(features)
    )
    assert calculator.weights.source == "manual_heuristic_legacy"
