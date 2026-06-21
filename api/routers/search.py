"""학교·학원 검색 엔드포인트 (자동완성)."""
from __future__ import annotations

from fastapi import APIRouter, Query

from api.cache import search_cache
from api.deps import get_academies, get_schools
from api.models import SearchResult

router = APIRouter(prefix="/search", tags=["검색"])


def _fuzzy_filter(df, query: str, name_col: str = "name", max_results: int = 10):
    q = query.strip().lower()
    if not q:
        return df.head(max_results)
    mask = df[name_col].str.lower().str.contains(q, na=False) | \
           df["address"].str.lower().str.contains(q, na=False)
    return df[mask].head(max_results)


@router.get("/schools", response_model=list[SearchResult], summary="학교 검색")
def search_schools(
    q: str = Query("", min_length=0, max_length=50, description="학교명 또는 주소"),
    district: str | None = Query(None, description="구 필터 (예: 유성구)"),
) -> list[SearchResult]:
    key = search_cache.make_key("school", q, district)
    cached = search_cache.get(key)
    if cached is not None:
        return cached

    df = get_schools()
    if district:
        df = df[df["district"].eq(district)]
    df = _fuzzy_filter(df, q)
    results = [
        SearchResult(
            name=row["name"],
            address=row["address"],
            lat=float(row["lat"]) if row["lat"] is not None else None,
            lon=float(row["lon"]) if row["lon"] is not None else None,
            district=row.get("district"),
        )
        for _, row in df.iterrows()
    ]
    search_cache.set(key, results)
    return results


@router.get("/academies", response_model=list[SearchResult], summary="학원 검색")
def search_academies(
    q: str = Query("", min_length=0, max_length=50, description="학원명 또는 주소"),
    district: str | None = Query(None, description="구 필터 (예: 서구)"),
) -> list[SearchResult]:
    key = search_cache.make_key("academy", q, district)
    cached = search_cache.get(key)
    if cached is not None:
        return cached

    df = get_academies()
    if district:
        df = df[df["district"].eq(district)]
    df = _fuzzy_filter(df, q)
    results = [
        SearchResult(
            name=row["name"],
            address=row["address"],
            lat=row.get("lat"),
            lon=row.get("lon"),
            district=row.get("district"),
        )
        for _, row in df.iterrows()
    ]
    search_cache.set(key, results)
    return results
