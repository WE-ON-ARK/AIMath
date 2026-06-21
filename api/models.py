"""Pydantic 스키마 — 경로 추천 API 요청/응답."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class RouteMode(str, Enum):
    shortest = "shortest"
    safest = "safest"
    balanced = "balanced"


class AgeGroup(str, Enum):
    low = "low"        # 초등 1-2학년
    mid = "mid"        # 초등 3-4학년
    high = "high"      # 초등 5-6학년
    middle = "middle"  # 중학생


class Coordinate(BaseModel):
    lat: float = Field(..., ge=36.18, le=36.55, description="위도 (대전 범위)")
    lon: float = Field(..., ge=127.29, le=127.62, description="경도 (대전 범위)")


class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    mode: RouteMode = RouteMode.balanced
    age_group: AgeGroup = AgeGroup.mid
    hour: int = Field(8, ge=0, le=23, description="출발 시각 (0-23)")
    lam: float | None = Field(None, ge=0.0, le=1.0,
                               description="balanced 모드의 λ (None이면 연령별 기본값 사용)")


class RouteSegment(BaseModel):
    edge_id: str
    length_m: float
    cmcs: float
    highway: str | None = None


class RouteResponse(BaseModel):
    mode: str
    age_group: str
    hour: int
    time_label: str
    total_distance_m: float
    total_cmcs: float
    average_cmcs: float
    detour_ratio: float
    detour_exceeded: bool
    risk_reduction_pct: float | None
    num_segments: int
    path_nodes: list[Any]
    disclaimer: str = (
        "이 경로는 통계 모델 기반 참고용입니다. "
        "실제 현장 상황과 다를 수 있으며 보호자의 판단을 대체하지 않습니다."
    )


class SearchResult(BaseModel):
    name: str
    address: str
    lat: float | None
    lon: float | None
    district: str | None


class HealthResponse(BaseModel):
    status: str
    graph_loaded: bool
    graph_nodes: int
    graph_edges: int
    model_loaded: bool
    version: str = "0.1.0"
