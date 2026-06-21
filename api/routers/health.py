"""헬스체크 및 캐시 통계 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter

from api.cache import route_cache, search_cache
from api.deps import get_optimizer, is_ready
from api.models import HealthResponse

router = APIRouter(tags=["운영"])


@router.get("/health", response_model=HealthResponse, summary="서비스 상태 확인")
def health() -> HealthResponse:
    opt = get_optimizer()
    loaded = opt is not None
    return HealthResponse(
        status="ok" if loaded else "initializing",
        graph_loaded=loaded,
        graph_nodes=opt.G.number_of_nodes() if loaded else 0,
        graph_edges=opt.G.number_of_edges() if loaded else 0,
        model_loaded=loaded,
    )


@router.get("/metrics", summary="캐시 통계")
def metrics() -> dict:
    return {
        "route_cache": route_cache.stats(),
        "search_cache": search_cache.stats(),
        "ready": is_ready(),
    }
