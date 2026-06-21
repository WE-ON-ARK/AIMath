#!/usr/bin/env python
"""데이터 자동 갱신 스케줄러.

환경변수:
  REFRESH_INTERVAL_HOURS  갱신 주기 (기본 24시간)
  DATA_GO_KR_API_KEY      공공데이터 API 키
  KAKAO_API_KEY           카카오 지오코딩 API 키 (선택)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/refresh.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

INTERVAL_H = float(os.getenv("REFRESH_INTERVAL_HOURS", "24"))


def _collect_real_data() -> dict[str, int]:
    from src.full_pipeline import collect_available_real_data
    return collect_available_real_data(refresh=True)


def _run_quality_check() -> None:
    from src.data_quality import run_data_quality_check
    reports = run_data_quality_check()
    for name, r in reports.items():
        if r.missing > 0 or r.out_of_bbox > 0:
            logger.warning("[품질 경고] %s: missing=%d, out_of_bbox=%d", name, r.missing, r.out_of_bbox)


def _detect_model_drift() -> bool:
    """기존 리포트와 현재 데이터 통계를 비교해 드리프트를 감지한다."""
    import json
    report_path = Path("outputs/reports/real_data_model_report.json")
    quality_path = Path("outputs/reports/data_quality_report.json")
    if not report_path.exists() or not quality_path.exists():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prev_count = report.get("dataset", {}).get("school_count", 0)
    # 학교 수가 10% 이상 변하면 드리프트로 간주
    from src.real_data_pipeline import build_school_risk_dataset
    try:
        dataset, _ = build_school_risk_dataset()
        current_count = len(dataset)
        drift = abs(current_count - prev_count) / max(prev_count, 1) > 0.10
        if drift:
            logger.warning("[드리프트] 학교 수 변화: %d → %d", prev_count, current_count)
        return drift
    except Exception as exc:
        logger.error("[드리프트 검사 실패] %s", exc)
        return False


def run_once() -> None:
    logger.info("=== 데이터 갱신 시작 ===")
    try:
        counts = _collect_real_data()
        logger.info("데이터 수집 완료: %s", counts)
    except Exception as exc:
        logger.error("데이터 수집 실패: %s", exc)

    try:
        _run_quality_check()
    except Exception as exc:
        logger.error("품질 검사 실패: %s", exc)

    try:
        drift = _detect_model_drift()
        if drift:
            logger.warning("모델 재학습이 권장됩니다 (입력 분포 변화 감지).")
    except Exception as exc:
        logger.error("드리프트 감지 실패: %s", exc)

    logger.info("=== 갱신 완료 ===")


if __name__ == "__main__":
    once = os.getenv("RUN_ONCE", "0") == "1"
    if once:
        run_once()
    else:
        logger.info("갱신 스케줄러 시작 (주기: %.0fh)", INTERVAL_H)
        while True:
            run_once()
            next_run = INTERVAL_H * 3600
            logger.info("다음 갱신까지 %.0f초 대기...", next_run)
            time.sleep(next_run)
