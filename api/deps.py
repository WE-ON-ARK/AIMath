"""의존성 주입 — 그래프·모델·검색 인덱스를 앱 수명 동안 한 번만 로드."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import GRAPH_DATA_DIR, PROCESSED_DATA_DIR

_state: dict = {}


def _load_optimizer() -> object:
    from src.route_optimizer import RouteOptimizer

    graph_path = Path(os.getenv("CMCS_GRAPH_PATH", str(GRAPH_DATA_DIR / "daejeon_walk_cmcs.graphml")))
    cmcs_path = Path(os.getenv("CMCS_EDGE_PATH", str(PROCESSED_DATA_DIR / "daejeon_edge_cmcs.csv")))

    if not graph_path.exists():
        graph_path = GRAPH_DATA_DIR / "daejeon_walk.graphml"
    if not graph_path.exists():
        graph_path = GRAPH_DATA_DIR / "demo_walk.graphml"

    cmcs_data = pd.read_csv(cmcs_path) if cmcs_path.exists() else None
    return RouteOptimizer(graph_path=graph_path, cmcs_data=cmcs_data)


def _load_school_index(data_dir: Path) -> pd.DataFrame:
    matches = sorted(data_dir.glob("*초중등학교위치*.csv"))
    if not matches:
        return pd.DataFrame(columns=["name", "address", "lat", "lon", "district"])
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(matches[0], encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return pd.DataFrame(columns=["name", "address", "lat", "lon", "district"])

    from src.real_data_pipeline import extract_district
    df = df[df["시도교육청명"].eq("대전광역시교육청")].dropna(subset=["위도", "경도"])
    return pd.DataFrame({
        "name": df["학교명"].astype(str),
        "address": df["소재지도로명주소"].astype(str),
        "lat": pd.to_numeric(df["위도"], errors="coerce"),
        "lon": pd.to_numeric(df["경도"], errors="coerce"),
        "district": df["소재지도로명주소"].map(extract_district),
    }).dropna(subset=["lat", "lon"]).reset_index(drop=True)


def _load_academy_index(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*교육지원청+학원+및+교습소+현황*.xlsx"))
    if not files:
        return pd.DataFrame(columns=["name", "address", "lat", "lon", "district"])
    from src.real_data_pipeline import extract_district
    rows = []
    for path in files:
        wb = pd.ExcelFile(path)
        for sheet in wb.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet)
            addr_col = next((c for c in frame.columns if "주소" in str(c)), None)
            name_col = next((c for c in frame.columns if str(c) in {"학원명", "교습소명"}), None)
            if addr_col is None or name_col is None:
                continue
            for _, row in frame[[name_col, addr_col]].dropna().drop_duplicates().iterrows():
                addr = str(row[addr_col]).strip()
                rows.append({
                    "name": str(row[name_col]),
                    "address": addr,
                    "lat": None,
                    "lon": None,
                    "district": extract_district(addr),
                })
    return pd.DataFrame(rows).drop_duplicates(subset=["name", "address"]).reset_index(drop=True)


def startup(data_dir: Path | None = None) -> None:
    """앱 시작 시 리소스를 로드한다."""
    from config import DATA_DIR
    data_dir = data_dir or DATA_DIR
    _state["optimizer"] = _load_optimizer()
    _state["schools"] = _load_school_index(data_dir)
    _state["academies"] = _load_academy_index(data_dir)
    n = _state["optimizer"].G.number_of_nodes()
    e = _state["optimizer"].G.number_of_edges()
    print(f"[startup] 그래프 로드 완료: 노드 {n:,}, 간선 {e:,}")
    print(f"[startup] 학교 {len(_state['schools'])}개, 학원 {len(_state['academies'])}개 인덱스 완료")


def get_optimizer():
    return _state.get("optimizer")


def get_schools() -> pd.DataFrame:
    return _state.get("schools", pd.DataFrame())


def get_academies() -> pd.DataFrame:
    return _state.get("academies", pd.DataFrame())


def is_ready() -> bool:
    return "optimizer" in _state
