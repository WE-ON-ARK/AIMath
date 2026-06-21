from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = DATA_DIR / "graph"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MAP_OUTPUT_DIR = OUTPUT_DIR / "maps"
CHART_OUTPUT_DIR = OUTPUT_DIR / "charts"
REPORT_OUTPUT_DIR = OUTPUT_DIR / "reports"


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("DATA_GO_KR_API_KEY", "")
    base_url: str = "https://apis.data.go.kr"
    region: str = os.getenv("CMCS_REGION", "대전")
    place: str = os.getenv("CMCS_PLACE", "Daejeon, South Korea")
    request_interval_seconds: float = float(
        os.getenv("CMCS_REQUEST_INTERVAL_SECONDS", "0.5")
    )
    request_timeout_seconds: float = 30.0
    default_cmcs: float = 0.5
    risk_floor: float = 0.05


SETTINGS = Settings()


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        GRAPH_DATA_DIR,
        MODEL_DIR,
        MAP_OUTPUT_DIR,
        CHART_OUTPUT_DIR,
        REPORT_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

