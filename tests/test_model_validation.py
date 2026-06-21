"""모델 검증 모듈 단위 테스트."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cmcs_calculator import CMCSWeights
from src.model_validation import (
    _perturb_dimension_weights,
    ablation_test,
    calibration_analysis,
    cmcs_sensitivity_analysis,
    district_holdout_summary,
    optimize_threshold,
    temporal_split_validation,
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _dummy_features(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    districts = ["대덕구", "동구", "중구", "서구", "유성구"]
    df = pd.DataFrame(
        {
            "crosswalk_count_100m": rng.integers(0, 5, n).astype(float),
            "crosswalk_count_300m": rng.integers(0, 10, n).astype(float),
            "crosswalk_count_500m": rng.integers(0, 15, n).astype(float),
            "signal_count_100m": rng.integers(0, 4, n).astype(float),
            "signal_count_300m": rng.integers(0, 8, n).astype(float),
            "signal_count_500m": rng.integers(0, 12, n).astype(float),
            "nearest_crosswalk_m": rng.uniform(10, 500, n),
            "nearest_signal_m": rng.uniform(10, 600, n),
            "pedestrian_crosswalk_signal_ratio_300m": rng.uniform(0, 1, n),
            "crosswalk_audio_ratio_300m": rng.uniform(0, 1, n),
            "tactile_block_ratio_300m": rng.uniform(0, 1, n),
            "raised_crosswalk_ratio_300m": rng.uniform(0, 1, n),
            "focused_light_ratio_300m": rng.uniform(0, 1, n),
            "avg_lane_count_300m": rng.uniform(1, 4, n),
            "actuated_signal_ratio_300m": rng.uniform(0, 1, n),
            "countdown_signal_ratio_300m": rng.uniform(0, 1, n),
            "signal_audio_ratio_300m": rng.uniform(0, 1, n),
            "academy_count_district": rng.integers(50, 500, n).astype(float),
            "illegal_parking_count_district": rng.integers(1000, 100000, n).astype(float),
            "district": [districts[i % len(districts)] for i in range(n)],
            "accident_hotspot_within_radius": (rng.uniform(0, 1, n) > 0.65).astype(int),
            "latitude": rng.uniform(36.25, 36.50, n),
            "longitude": rng.uniform(127.30, 127.55, n),
        }
    )
    return df


def _dummy_cmcs_features(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "accident_count_norm": rng.uniform(0, 1, n),
            "traffic_volume_norm": rng.uniform(0, 1, n),
            "avg_speed_norm": rng.uniform(0, 1, n),
            "narrow_sidewalk_norm": rng.uniform(0, 1, n),
            "slope_norm": rng.uniform(0, 1, n),
            "is_alley": rng.integers(0, 2, n).astype(float),
            "pedestrian_flow_norm": rng.uniform(0, 1, n),
            "academy_density_norm": rng.uniform(0, 1, n),
            "bus_stop_nearby_norm": rng.uniform(0, 1, n),
            "illegal_parking_norm": rng.uniform(0, 1, n),
            "light_density_norm": rng.uniform(0, 1, n),
            "has_crosswalk": rng.integers(0, 2, n).astype(float),
            "has_signal": rng.integers(0, 2, n).astype(float),
            "lane_count_norm": rng.uniform(0, 1, n),
            "has_speed_bump": rng.integers(0, 2, n).astype(float),
            "has_cctv": rng.integers(0, 2, n).astype(float),
            "is_school_zone": rng.integers(0, 2, n).astype(float),
        }
    )


def _dummy_hotspots(n: int = 10, seed: int = 2, school_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """학교 위치와 일부 핫스팟이 가까이 있도록 생성해 양성 샘플을 보장한다."""
    rng = np.random.default_rng(seed)
    lats = rng.uniform(36.25, 36.50, n)
    lons = rng.uniform(127.30, 127.55, n)
    if school_df is not None and len(school_df) > 0:
        # 처음 2개 핫스팟을 학교 좌표 근처(~50m)에 배치
        for i in range(min(2, n, len(school_df))):
            lats[i] = float(school_df["latitude"].iloc[i]) + 0.0002
            lons[i] = float(school_df["longitude"].iloc[i]) + 0.0002
    return pd.DataFrame(
        {
            "la_crd": lats,
            "lo_crd": lons,
            "search_year": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024][:n],
            "occrrnc_cnt": rng.integers(1, 5, n),
            "caslt_cnt": rng.integers(1, 8, n),
            "dth_dnv_cnt": rng.integers(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# CMCS 민감도 분석 테스트
# ---------------------------------------------------------------------------

def test_perturb_weights_sums_to_one():
    base = {"risk": 0.35, "discomfort": 0.15, "congestion": 0.15, "obstruction": 0.15, "crossing": 0.20}
    perturbed = _perturb_dimension_weights(base, "risk", 0.07)
    assert abs(sum(perturbed.values()) - 1.0) < 1e-9


def test_perturb_weights_changes_target():
    base = {"risk": 0.35, "discomfort": 0.15, "congestion": 0.15, "obstruction": 0.15, "crossing": 0.20}
    perturbed = _perturb_dimension_weights(base, "congestion", +0.03)
    assert perturbed["congestion"] > base["congestion"]


def test_cmcs_sensitivity_keys(tmp_path):
    features = _dummy_cmcs_features()
    result = cmcs_sensitivity_analysis(features, output_path=tmp_path / "sens.json")
    assert "dimensions" in result
    for dim in CMCSWeights().dimensions:
        assert dim in result["dimensions"]
        assert "plus" in result["dimensions"][dim]
        assert "minus" in result["dimensions"][dim]


def test_cmcs_sensitivity_spearman_high(tmp_path):
    """±20% 변동에서도 스피어만 상관이 0.9 이상이어야 한다."""
    features = _dummy_cmcs_features(n=200)
    result = cmcs_sensitivity_analysis(features, delta_fraction=0.20, output_path=tmp_path / "s.json")
    for dim, vals in result["dimensions"].items():
        for direction in ("plus", "minus"):
            assert vals[direction]["spearman_r"] >= 0.9, (
                f"{dim}/{direction} 스피어만 상관이 너무 낮음: {vals[direction]['spearman_r']}"
            )


# ---------------------------------------------------------------------------
# Ablation 테스트
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "crosswalk_count_100m", "crosswalk_count_300m", "crosswalk_count_500m",
    "signal_count_100m", "signal_count_300m", "signal_count_500m",
    "nearest_crosswalk_m", "nearest_signal_m",
    "pedestrian_crosswalk_signal_ratio_300m", "crosswalk_audio_ratio_300m",
    "tactile_block_ratio_300m", "raised_crosswalk_ratio_300m",
    "focused_light_ratio_300m", "avg_lane_count_300m",
    "actuated_signal_ratio_300m", "countdown_signal_ratio_300m",
    "signal_audio_ratio_300m", "academy_count_district",
    "illegal_parking_count_district",
]


def test_ablation_returns_dataframe(tmp_path):
    df = _dummy_features(n=60)
    result = ablation_test(df, FEATURE_COLS, output_path=tmp_path / "abl.json")
    assert isinstance(result, pd.DataFrame)
    assert "removed_feature" in result.columns
    assert len(result) == len(FEATURE_COLS)


def test_ablation_all_features_covered(tmp_path):
    df = _dummy_features(n=60)
    result = ablation_test(df, FEATURE_COLS, output_path=tmp_path / "abl.json")
    assert set(result["removed_feature"]) == set(FEATURE_COLS)


# ---------------------------------------------------------------------------
# Calibration 테스트
# ---------------------------------------------------------------------------

def test_calibration_returns_brier(tmp_path):
    rng = np.random.default_rng(7)
    y = pd.Series(rng.integers(0, 2, 80))
    proba = rng.uniform(0, 1, 80)
    result = calibration_analysis(
        y, proba,
        output_path=tmp_path / "cal.json",
        chart_path=tmp_path / "cal.png",
    )
    assert "brier_score" in result
    assert 0.0 <= result["brier_score"] <= 1.0


def test_calibration_perfect_model(tmp_path):
    y = pd.Series([0, 0, 1, 1, 0, 1])
    proba = np.array([0.05, 0.05, 0.95, 0.95, 0.05, 0.95])
    result = calibration_analysis(y, proba, output_path=tmp_path / "cp.json", chart_path=tmp_path / "cp.png")
    assert result["brier_score"] < result["brier_dummy"]


# ---------------------------------------------------------------------------
# 임계값 최적화 테스트
# ---------------------------------------------------------------------------

def test_optimize_threshold_output_keys(tmp_path):
    rng = np.random.default_rng(9)
    y = pd.Series(rng.integers(0, 2, 100))
    proba = rng.uniform(0, 1, 100)
    result = optimize_threshold(y, proba, output_path=tmp_path / "thr.json")
    assert "optimal_threshold" in result
    assert "threshold" in result["optimal_threshold"]
    assert 0.0 < result["optimal_threshold"]["threshold"] < 1.0


def test_optimize_threshold_f1_nonnegative(tmp_path):
    rng = np.random.default_rng(10)
    y = pd.Series(rng.integers(0, 2, 100))
    proba = rng.uniform(0, 1, 100)
    result = optimize_threshold(y, proba, output_path=tmp_path / "thr2.json")
    assert result["optimal_threshold"]["f1"] >= 0.0


# ---------------------------------------------------------------------------
# 시간 분리 검증 테스트
# ---------------------------------------------------------------------------

def test_temporal_split_returns_dict(tmp_path):
    df = _dummy_features(n=60)
    hs = _dummy_hotspots(n=10, school_df=df)
    result = temporal_split_validation(
        df, hs, FEATURE_COLS,
        train_years=[2015, 2016, 2017, 2018, 2019],
        test_years=[2020, 2021, 2022, 2023, 2024],
        label_radius_m=50000.0,  # 더미 데이터에서 최소 양성 보장
        output_path=tmp_path / "temp.json",
    )
    assert isinstance(result, dict)
    assert "train_years" in result
    assert "test_years" in result


def test_temporal_split_year_partition(tmp_path):
    df = _dummy_features(n=60)
    hs = _dummy_hotspots(n=10, school_df=df)
    result = temporal_split_validation(
        df, hs, FEATURE_COLS,
        train_years=[2015, 2016, 2017, 2018, 2019],
        test_years=[2020, 2021, 2022, 2023, 2024],
        label_radius_m=50000.0,
        output_path=tmp_path / "tp2.json",
    )
    assert set(result["train_years"]).isdisjoint(set(result["test_years"]))


# ---------------------------------------------------------------------------
# 구 단위 홀드아웃 요약 테스트
# ---------------------------------------------------------------------------

def test_district_holdout_covers_all_districts(tmp_path):
    df = _dummy_features(n=60)
    result = district_holdout_summary(df, FEATURE_COLS, output_path=tmp_path / "dh.json")
    assert isinstance(result, pd.DataFrame)
    assert "held_out_district" in result.columns
    expected = {"대덕구", "동구", "중구", "서구", "유성구"}
    assert set(result["held_out_district"]) == expected


def test_district_holdout_json_written(tmp_path):
    df = _dummy_features(n=60)
    out = tmp_path / "dh.json"
    district_holdout_summary(df, FEATURE_COLS, output_path=out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "per_district" in data
