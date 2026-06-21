"""좌표 데이터 품질 검사 및 보고."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPORT_OUTPUT_DIR

DAEJEON_LAT_MIN = 36.18
DAEJEON_LAT_MAX = 36.55
DAEJEON_LON_MIN = 127.29
DAEJEON_LON_MAX = 127.62

DATA_VINTAGE: dict[str, str] = {
    "횡단보도": "2023-01-05",
    "신호등": "2023-01-05",
    "불법주정차": "2022-12-31",
    "학원·교습소": "2026-05-31",
    "학교위치": "2026-03-20",
    "어린이보호구역": "최근 API 조회",
    "과속방지턱": "최근 API 조회",
    "사고다발지역": "2012~2024",
    "버스정류장(OSM)": "OSM 최신",
    "가로등(OSM)": "OSM 최신",
}


@dataclass
class CoordinateReport:
    name: str
    total: int
    missing: int
    duplicates: int
    out_of_bbox: int
    suspect_indices: list[int] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        bad = self.missing + self.out_of_bbox
        return round(1.0 - bad / self.total, 4)

    def summary(self) -> str:
        return (
            f"{self.name}: total={self.total}, missing={self.missing}, "
            f"duplicates={self.duplicates}, out_of_bbox={self.out_of_bbox}, "
            f"pass_rate={self.pass_rate:.1%}"
        )


def _in_daejeon_bbox(lat: float, lon: float) -> bool:
    return (
        DAEJEON_LAT_MIN <= lat <= DAEJEON_LAT_MAX
        and DAEJEON_LON_MIN <= lon <= DAEJEON_LON_MAX
    )


def check_coordinate_column(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    name: str = "unknown",
) -> CoordinateReport:
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    missing_mask = lat.isna() | lon.isna()
    missing_count = int(missing_mask.sum())

    valid = df[~missing_mask].copy()
    valid_lat = lat[~missing_mask]
    valid_lon = lon[~missing_mask]

    dup_mask = valid_lat.round(6).astype(str) + "," + valid_lon.round(6).astype(str)
    duplicates = int(dup_mask.duplicated().sum())

    bbox_mask = ~np.array(
        [
            _in_daejeon_bbox(la, lo)
            for la, lo in zip(valid_lat, valid_lon)
        ]
    )
    out_of_bbox = int(bbox_mask.sum())
    suspect = valid.index[bbox_mask].tolist()

    return CoordinateReport(
        name=name,
        total=len(df),
        missing=missing_count,
        duplicates=duplicates,
        out_of_bbox=out_of_bbox,
        suspect_indices=suspect[:20],
    )


def check_all_datasets(data_dir: str | Path = "data") -> dict[str, CoordinateReport]:
    data_dir = Path(data_dir)
    results: dict[str, CoordinateReport] = {}

    def _read(pattern: str) -> pd.DataFrame | None:
        matches = sorted(data_dir.glob(pattern))
        if not matches:
            return None
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(matches[0], encoding=enc)
            except UnicodeDecodeError:
                pass
        return None

    crosswalk = _read("*횡단보도*.csv")
    if crosswalk is not None and "위도" in crosswalk.columns:
        results["횡단보도"] = check_coordinate_column(crosswalk, "위도", "경도", "횡단보도")

    signal = _read("*신호등*.csv")
    if signal is not None and "위도" in signal.columns:
        results["신호등"] = check_coordinate_column(signal, "위도", "경도", "신호등")

    school = _read("*초중등학교위치*.csv")
    if school is not None and "위도" in school.columns:
        # 전국 파일 → 대전 초등학교만 필터링 후 검사
        if "시도교육청명" in school.columns:
            school = school[school["시도교육청명"].eq("대전광역시교육청")]
        results["학교위치(대전)"] = check_coordinate_column(school, "위도", "경도", "학교위치(대전)")

    speed_bump_path = data_dir / "raw" / "speed_bump.csv"
    if speed_bump_path.exists():
        sb = _read_path(speed_bump_path)
        if sb is not None and "LATITUDE" in sb.columns:
            results["과속방지턱"] = check_coordinate_column(
                sb, "LATITUDE", "LONGITUDE", "과속방지턱"
            )

    school_zone_path = data_dir / "raw" / "school_zone.csv"
    if school_zone_path.exists():
        sz = _read_path(school_zone_path)
        if sz is not None and "latitude" in sz.columns:
            results["어린이보호구역"] = check_coordinate_column(
                sz, "latitude", "longitude", "어린이보호구역"
            )

    hotspot_path = data_dir / "raw" / "daejeon_schoolzone_accident_hotspots.csv"
    if hotspot_path.exists():
        hs = _read_path(hotspot_path)
        if hs is not None and "la_crd" in hs.columns:
            results["사고다발지역"] = check_coordinate_column(
                hs, "la_crd", "lo_crd", "사고다발지역"
            )

    return results


def _read_path(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    return None


def write_quality_report(
    reports: dict[str, CoordinateReport],
    output_path: str | Path = REPORT_OUTPUT_DIR / "data_quality_report.json",
) -> Path:
    output = {
        "data_vintage": DATA_VINTAGE,
        "coordinate_quality": {
            name: {**asdict(r), "pass_rate": r.pass_rate}
            for name, r in reports.items()
        },
        "summary": {
            "total_datasets": len(reports),
            "all_pass": all(
                r.missing == 0 and r.out_of_bbox == 0 for r in reports.values()
            ),
            "datasets_with_issues": [
                name
                for name, r in reports.items()
                if r.missing > 0 or r.out_of_bbox > 0
            ],
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_data_quality_check(
    data_dir: str | Path = "data",
    output_path: str | Path = REPORT_OUTPUT_DIR / "data_quality_report.json",
) -> dict[str, CoordinateReport]:
    reports = check_all_datasets(data_dir)
    write_quality_report(reports, output_path)
    for r in reports.values():
        print(r.summary())
    return reports
