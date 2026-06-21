import pandas as pd
import pytest

from src.data_quality import (
    DAEJEON_LAT_MAX,
    DAEJEON_LAT_MIN,
    DAEJEON_LON_MAX,
    DAEJEON_LON_MIN,
    CoordinateReport,
    _in_daejeon_bbox,
    check_coordinate_column,
)


def test_bbox_daejeon_center():
    assert _in_daejeon_bbox(36.35, 127.38) is True


def test_bbox_rejects_seoul():
    assert _in_daejeon_bbox(37.56, 126.97) is False


def test_bbox_boundary_values():
    assert _in_daejeon_bbox(DAEJEON_LAT_MIN, DAEJEON_LON_MIN) is True
    assert _in_daejeon_bbox(DAEJEON_LAT_MAX, DAEJEON_LON_MAX) is True
    assert _in_daejeon_bbox(DAEJEON_LAT_MIN - 0.001, DAEJEON_LON_MIN) is False


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


def test_check_no_issues():
    df = _make_df(lat=[36.35, 36.40], lon=[127.38, 127.40])
    r = check_coordinate_column(df, "lat", "lon", "test")
    assert r.missing == 0
    assert r.out_of_bbox == 0
    assert r.pass_rate == 1.0


def test_check_missing_values():
    df = _make_df(lat=[36.35, None, 36.40], lon=[127.38, 127.40, None])
    r = check_coordinate_column(df, "lat", "lon", "test")
    assert r.missing == 2
    assert r.total == 3


def test_check_out_of_bbox():
    df = _make_df(lat=[36.35, 37.56], lon=[127.38, 126.97])
    r = check_coordinate_column(df, "lat", "lon", "test")
    assert r.out_of_bbox == 1
    assert len(r.suspect_indices) == 1


def test_check_duplicates():
    df = _make_df(lat=[36.35, 36.35, 36.40], lon=[127.38, 127.38, 127.40])
    r = check_coordinate_column(df, "lat", "lon", "test")
    assert r.duplicates == 1


def test_pass_rate_calculation():
    df = _make_df(lat=[36.35, None], lon=[127.38, 127.40])
    r = check_coordinate_column(df, "lat", "lon", "test")
    assert r.pass_rate == 0.5


def test_coordinate_report_summary_contains_name():
    r = CoordinateReport(name="신호등", total=100, missing=2, duplicates=0, out_of_bbox=1)
    assert "신호등" in r.summary()
    assert "missing=2" in r.summary()
