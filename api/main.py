"""CMCS 어린이 안전 통학 경로 추천 API."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api import deps
from api.routers import health, route, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("서비스 시작 — 그래프·모델 로드 중...")
    try:
        deps.startup()
    except Exception as exc:
        logger.warning("리소스 로드 실패 (경량 모드로 계속): %s", exc)
    yield
    logger.info("서비스 종료")


app = FastAPI(
    title="CMCS 어린이 안전 통학 경로 추천 API",
    description=(
        "대전 보행 도로망과 CMCS(Combined Multi-Criteria Safety) 모델 기반 "
        "어린이 통학 경로 추천 서비스. "
        "**참고용**이며 보호자의 판단을 대체하지 않습니다."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, elapsed)
    response.headers["X-Response-Time-Ms"] = str(elapsed)
    return response


@app.middleware("http")
async def timeout_guard(request: Request, call_next):
    import asyncio
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse({"detail": "요청 처리 시간이 초과되었습니다 (30초)."}, status_code=504)


app.include_router(health.router)
app.include_router(route.router)
app.include_router(search.router)

try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except Exception:
    pass  # static 디렉토리 없으면 스킵
