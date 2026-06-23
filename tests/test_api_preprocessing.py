import pandas as pd

from src.api_preprocessing import preprocess_api_frame


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
