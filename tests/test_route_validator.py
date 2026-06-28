"""경로 추천 검증 모듈 단위 테스트."""
import json
import tempfile
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from src.route_optimizer import RouteOptimizer
from src.route_validator import (
    AGE_PROFILES,
    AgeProfile,
    TimeWeights,
    age_profile,
    apply_time_weights_to_cmcs,
    batch_od_evaluation,
    check_detour_ratio,
    time_weights,
)


# ---------------------------------------------------------------------------
# 테스트 그래프 헬퍼
# ---------------------------------------------------------------------------

def _build_graph() -> tuple[nx.MultiDiGraph, RouteOptimizer]:
    G = nx.MultiDiGraph()
    for i, node in enumerate(["A", "B", "C", "D"]):
        G.add_node(node, x=float(i), y=0.0)
    scores = pd.DataFrame([
        {"edge_id": "ab", "cmcs": 0.9},
        {"edge_id": "bd", "cmcs": 0.9},
        {"edge_id": "ac", "cmcs": 0.1},
        {"edge_id": "cd", "cmcs": 0.1},
    ])
    G.add_edge("A", "B", edge_id="ab", length=100.0)
    G.add_edge("B", "D", edge_id="bd", length=100.0)
    G.add_edge("A", "C", edge_id="ac", length=140.0)
    G.add_edge("C", "D", edge_id="cd", length=140.0)
    opt = RouteOptimizer(graph=G, cmcs_data=scores)
    return G, opt


# ---------------------------------------------------------------------------
# 연령대 프로파일 테스트
# ---------------------------------------------------------------------------

def test_all_age_groups_defined():
    for key in ("low", "mid", "high", "middle"):
        p = age_profile(key)
        assert isinstance(p, AgeProfile)
        assert 0.0 <= p.lam <= 1.0
        assert p.max_detour_ratio >= 1.0


def test_invalid_age_group_raises():
    with pytest.raises(ValueError, match="age_group"):
        age_profile("unknown")


def test_younger_is_more_conservative():
    low = age_profile("low")
    mid = age_profile("mid")
    high = age_profile("high")
    assert low.lam < mid.lam < high.lam
    assert low.max_detour_ratio >= mid.max_detour_ratio >= high.max_detour_ratio


# ---------------------------------------------------------------------------
# 시간대 가중치 테스트
# ---------------------------------------------------------------------------

def test_nighttime_light_multiplier_greater():
    night = time_weights(23)
    day = time_weights(12)
    assert night.light_multiplier > day.light_multiplier


def test_rush_hour_congestion_multiplier_greater():
    rush = time_weights(8)
    night = time_weights(2)
    assert rush.congestion_multiplier > night.congestion_multiplier


def test_time_weight_label_coverage():
    for hour in range(24):
        tw = time_weights(hour)
        assert tw.label in ("등교 러시아워", "낮 시간", "하교 러시아워", "야간")


def test_apply_time_weights_increases_night_cmcs():
    import pandas as pd, numpy as np
    cmcs = pd.Series([0.3, 0.5, 0.7])
    light = pd.Series([0.0, 0.5, 1.0])  # 조명 없음 → 있음
    tw_night = time_weights(23)
    tw_day = time_weights(12)
    night_adj = apply_time_weights_to_cmcs(cmcs, tw_night, light)
    day_adj = apply_time_weights_to_cmcs(cmcs, tw_day, light)
    # 야간은 조명이 없는 구간에서 CMCS가 더 높아야 함
    assert night_adj.iloc[0] > day_adj.iloc[0]


def test_apply_time_weights_clips_to_unit():
    import pandas as pd
    cmcs = pd.Series([0.95])
    tw = time_weights(23)
    result = apply_time_weights_to_cmcs(cmcs, tw)
    assert result.iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# 우회율 테스트
# ---------------------------------------------------------------------------

def test_detour_ratio_exact():
    ratio, exceeded = check_detour_ratio(100.0, 200.0, max_ratio=2.0)
    assert ratio == 2.0
    assert not exceeded  # 정확히 2.0은 초과 아님


def test_detour_ratio_exceeded():
    ratio, exceeded = check_detour_ratio(100.0, 201.0, max_ratio=2.0)
    assert ratio > 2.0
    assert exceeded


def test_detour_ratio_zero_shortest():
    ratio, exceeded = check_detour_ratio(0.0, 100.0)
    assert ratio == 0.0
    assert not exceeded


# ---------------------------------------------------------------------------
# OD 일괄 평가 테스트
# ---------------------------------------------------------------------------

def test_batch_od_evaluation_basic(tmp_path):
    _, opt = _build_graph()
    od = [("A", "D", "학교A", "학원D")]
    df = batch_od_evaluation(opt, od, age_group="mid", output_path=tmp_path / "od.json")
    assert len(df) == 1
    assert "shortest_m" in df.columns
    assert "safest_m" in df.columns
    assert "detour_ratio" in df.columns
    assert "risk_reduction_pct" in df.columns


def test_batch_od_detour_ratio_correct(tmp_path):
    _, opt = _build_graph()
    od = [("A", "D", "학교A", "학원D")]
    df = batch_od_evaluation(opt, od, age_group="mid", output_path=tmp_path / "od.json")
    row = df.iloc[0]
    # 안전 경로(280m) / 최단 경로(200m) = 1.4
    expected = round(row["safest_m"] / row["shortest_m"], 4)
    assert abs(row["detour_ratio"] - expected) < 1e-3


def test_batch_od_skips_invalid_pairs(tmp_path):
    _, opt = _build_graph()
    od = [("A", "D", "valid", "valid"), ("Z", "D", "invalid", "valid")]
    df = batch_od_evaluation(opt, od, age_group="mid", output_path=tmp_path / "od.json")
    assert len(df) == 1  # Z 노드 없으므로 건너뜀


def test_batch_od_report_written(tmp_path):
    _, opt = _build_graph()
    od = [("A", "D", "학교A", "학원D")]
    batch_od_evaluation(opt, od, output_path=tmp_path / "od.json")
    data = json.loads((tmp_path / "od.json").read_text(encoding="utf-8"))
    assert "total_pairs" in data
    assert "age_profile" in data


def test_batch_od_multiple_age_groups(tmp_path):
    _, opt = _build_graph()
    od = [("A", "D", "s", "d")]
    for group in AGE_PROFILES:
        df = batch_od_evaluation(opt, od, age_group=group,
                                  output_path=tmp_path / f"od_{group}.json")
        assert len(df) == 1
