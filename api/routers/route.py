"""경로 추천 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.cache import route_cache
from api.deps import get_optimizer, is_ready
from api.models import RouteMode, RouteRequest, RouteResponse
from src.route_validator import (
    age_profile,
    apply_time_weights_to_cmcs,
    check_detour_ratio,
    time_weights,
)

router = APIRouter(prefix="/route", tags=["경로"])


def _not_ready():
    raise HTTPException(503, "서비스 초기화 중입니다. 잠시 후 다시 시도해 주세요.")


@router.post("/recommend", response_model=RouteResponse, summary="경로 추천")
def recommend_route(req: RouteRequest) -> RouteResponse:
    """출발지→도착지 경로를 최단·안전·균형 모드로 추천합니다."""
    if not is_ready():
        _not_ready()

    opt = get_optimizer()
    origin = (req.origin.lat, req.origin.lon)
    dest = (req.destination.lat, req.destination.lon)

    cache_key = route_cache.make_key(origin, dest, req.mode, req.age_group, req.hour, req.lam)
    cached = route_cache.get(cache_key)
    if cached:
        return cached

    profile = age_profile(req.age_group)
    tw = time_weights(req.hour)

    try:
        if req.mode == RouteMode.shortest:
            route = opt.shortest_route(origin, dest)
        elif req.mode == RouteMode.safest:
            route = opt.safest_route(origin, dest)
        else:
            lam = req.lam if req.lam is not None else profile.lam
            route = opt.balanced_route(origin, dest, lam)
    except Exception as exc:
        raise HTTPException(422, f"경로를 찾을 수 없습니다: {exc}") from exc

    shortest = opt.shortest_route(origin, dest)
    detour_ratio, detour_exceeded = check_detour_ratio(
        shortest["total_distance_m"],
        route["total_distance_m"],
        profile.max_detour_ratio,
    )
    risk_reduction = None
    if shortest["total_cmcs"] > 0:
        risk_reduction = round(
            (shortest["total_cmcs"] - route["total_cmcs"]) / shortest["total_cmcs"] * 100.0, 2
        )

    response = RouteResponse(
        mode=route["mode"],
        age_group=profile.label,
        hour=req.hour,
        time_label=tw.label,
        total_distance_m=route["total_distance_m"],
        total_cmcs=route["total_cmcs"],
        average_cmcs=route["average_cmcs"],
        detour_ratio=detour_ratio,
        detour_exceeded=detour_exceeded,
        risk_reduction_pct=risk_reduction,
        num_segments=route["num_segments"],
        path_nodes=route["path"],
    )
    route_cache.set(cache_key, response)
    return response


@router.post("/compare", summary="세 가지 모드 경로 비교")
def compare_routes(req: RouteRequest) -> dict:
    """최단·안전·균형 경로를 한 번에 비교합니다."""
    if not is_ready():
        _not_ready()

    opt = get_optimizer()
    origin = (req.origin.lat, req.origin.lon)
    dest = (req.destination.lat, req.destination.lon)

    try:
        comparison = opt.compare_routes(origin, dest)
    except Exception as exc:
        raise HTTPException(422, f"경로 비교 실패: {exc}") from exc

    profile = age_profile(req.age_group)
    return {
        "age_group": profile.label,
        "max_detour_ratio": profile.max_detour_ratio,
        "routes": comparison.to_dict(orient="records"),
        "disclaimer": (
            "이 경로는 통계 모델 기반 참고용입니다. "
            "실제 현장 상황과 다를 수 있으며 보호자의 판단을 대체하지 않습니다."
        ),
    }
