"""FastAPI 엔드포인트 단위 테스트 (TestClient, 그래프 없이 실행)."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.deps as deps
from api.main import app
from src.route_optimizer import RouteOptimizer
import networkx as nx


def _build_test_optimizer() -> RouteOptimizer:
    G = nx.MultiDiGraph()
    for i, n in enumerate(["A", "B", "C", "D"]):
        G.add_node(n, x=127.38 + i * 0.001, y=36.35)
    scores = pd.DataFrame([
        {"edge_id": "ab", "cmcs": 0.8},
        {"edge_id": "bd", "cmcs": 0.8},
        {"edge_id": "ac", "cmcs": 0.2},
        {"edge_id": "cd", "cmcs": 0.2},
    ])
    G.add_edge("A", "B", edge_id="ab", length=100.0)
    G.add_edge("B", "D", edge_id="bd", length=100.0)
    G.add_edge("A", "C", edge_id="ac", length=140.0)
    G.add_edge("C", "D", edge_id="cd", length=140.0)
    return RouteOptimizer(graph=G, cmcs_data=scores)


@pytest.fixture(autouse=True)
def inject_test_optimizer():
    """실제 80MB 그래프 없이 테스트용 최적화기를 주입한다."""
    deps._state["optimizer"] = _build_test_optimizer()
    deps._state["schools"] = pd.DataFrame([
        {"name": "유성초등학교", "address": "대전광역시 유성구 대학로 1", "lat": 36.35, "lon": 127.381, "district": "유성구"},
        {"name": "갈마초등학교", "address": "대전광역시 서구 갈마동 1", "lat": 36.352, "lon": 127.372, "district": "서구"},
    ])
    deps._state["academies"] = pd.DataFrame([
        {"name": "한솔수학", "address": "대전광역시 유성구 노은로 10", "lat": 36.360, "lon": 127.340, "district": "유성구"},
    ])
    yield
    deps._state.clear()


client = TestClient(app)


# ---------------------------------------------------------------------------
# 헬스체크
# ---------------------------------------------------------------------------

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["graph_loaded"] is True


def test_metrics_returns_cache_stats():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "route_cache" in data
    assert "ready" in data


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def test_search_schools_no_query():
    r = client.get("/search/schools")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_search_schools_name_match():
    r = client.get("/search/schools?q=유성")
    assert r.status_code == 200
    names = [item["name"] for item in r.json()]
    assert "유성초등학교" in names


def test_search_schools_district_filter():
    r = client.get("/search/schools?district=서구")
    assert r.status_code == 200
    items = r.json()
    assert all(item["district"] == "서구" for item in items)


def test_search_academies():
    r = client.get("/search/academies?q=한솔")
    assert r.status_code == 200
    names = [i["name"] for i in r.json()]
    assert "한솔수학" in names


# ---------------------------------------------------------------------------
# 경로 추천 (좌표 기반 최근접 노드 사용)
# ---------------------------------------------------------------------------

VALID_ROUTE_BODY = {
    "origin": {"lat": 36.35, "lon": 127.380},
    "destination": {"lat": 36.35, "lon": 127.383},
    "mode": "balanced",
    "age_group": "mid",
    "hour": 8,
}


def test_route_recommend_returns_200():
    r = client.post("/route/recommend", json=VALID_ROUTE_BODY)
    assert r.status_code == 200


def test_route_recommend_has_required_fields():
    r = client.post("/route/recommend", json=VALID_ROUTE_BODY)
    data = r.json()
    for field in ("total_distance_m", "average_cmcs", "detour_ratio", "mode", "disclaimer"):
        assert field in data, f"필드 누락: {field}"


def test_route_safest_mode():
    body = {**VALID_ROUTE_BODY, "mode": "safest"}
    r = client.post("/route/recommend", json=body)
    assert r.status_code == 200
    assert "safest" in r.json()["mode"].lower() or r.json()["total_distance_m"] > 0


def test_route_disclaimer_present():
    r = client.post("/route/recommend", json=VALID_ROUTE_BODY)
    assert len(r.json()["disclaimer"]) > 10


def test_route_invalid_coordinate_rejected():
    body = {**VALID_ROUTE_BODY, "origin": {"lat": 99.0, "lon": 127.38}}
    r = client.post("/route/recommend", json=body)
    assert r.status_code == 422


def test_route_invalid_hour_rejected():
    body = {**VALID_ROUTE_BODY, "hour": 25}
    r = client.post("/route/recommend", json=body)
    assert r.status_code == 422


def test_route_compare_returns_list():
    r = client.post("/route/compare", json=VALID_ROUTE_BODY)
    assert r.status_code == 200
    data = r.json()
    assert "routes" in data
    assert isinstance(data["routes"], list)


def test_cache_hit_returns_same_result():
    r1 = client.post("/route/recommend", json=VALID_ROUTE_BODY)
    r2 = client.post("/route/recommend", json=VALID_ROUTE_BODY)
    assert r1.json()["total_distance_m"] == r2.json()["total_distance_m"]
