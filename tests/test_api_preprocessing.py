import numpy as np
import pandas as pd

from src.api_preprocessing import preprocess_api_frame, preprocess_for_cmcs_analysis


def test_api_preprocessing_corrects_swapped_coordinates_and_deduplicates():
    frame = pd.DataFrame(
        {
            "sn": ["A", "A", "B"],
            "latitude": [127.38, 36.35, 37.56],
            "longitude": [36.35, 127.38, 126.97],
            "name": ["  학교  앞 ", "학교 앞", "서울"],
        }
    )
    clean, report = preprocess_api_frame(frame, "school_zone")
    assert len(clean) == 1
    assert clean.iloc[0]["_latitude"] == 36.35
    assert clean.iloc[0]["_longitude"] == 127.38
    assert clean.iloc[0]["name"] == "학교 앞"
    assert report["coordinate_swaps_corrected"] == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["out_of_daejeon_rows"] == 1


def test_api_preprocessing_preserves_raw_columns_and_adds_provenance():
    frame = pd.DataFrame(
        {
            "LATITUDE": [36.35],
            "LONGITUDE": [127.38],
            "custom": ["raw"],
        }
    )
    clean, _ = preprocess_api_frame(frame, "speed_bump")
    assert clean.iloc[0]["custom"] == "raw"
    assert clean.iloc[0]["_source_dataset"] == "speed_bump"
    assert clean.iloc[0]["_preprocessing_version"] == "2.0"
    assert "_processed_at_utc" in clean


def test_preprocess_for_cmcs_clips_norm_columns():
    frame = pd.DataFrame({
        "segment_id": ["S1"],
        "traffic_volume_norm": [1.5],
        "has_crosswalk": [1],
        "accident_count": [2],
        "length_m": [100.0],
    })
    result, report = preprocess_for_cmcs_analysis(frame)
    assert result.iloc[0]["traffic_volume_norm"] <= 1.0
    assert "traffic_volume_norm" in report.get("range_clipped_columns", [])


def test_preprocess_for_cmcs_median_imputes_missing():
    frame = pd.DataFrame({
        "segment_id": ["S1", "S2"],
        "traffic_volume_norm": [0.5, np.nan],
        "accident_count": [1, 2],
        "length_m": [100.0, 200.0],
    })
    result, report = preprocess_for_cmcs_analysis(frame)
    assert result["traffic_volume_norm"].isna().sum() == 0
    assert "traffic_volume_norm" in report.get("median_imputation", {})


def test_preprocess_for_cmcs_flags_outliers():
    values = list(range(100)) + [10000]
    frame = pd.DataFrame({
        "segment_id": [f"S{i}" for i in range(101)],
        "crosswalk_count": [float(v) for v in values],
        "accident_count": [0] * 101,
        "length_m": [100.0] * 101,
    })
    result, report = preprocess_for_cmcs_analysis(frame)
    assert "crosswalk_count" in report.get("iqr_outlier_flags", {})
    assert "_outlier_crosswalk_count" in result.columns
