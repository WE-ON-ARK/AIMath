"""공공데이터 API 원본을 재현 가능한 표준 스키마로 정제한다."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import REPORT_OUTPUT_DIR
from src.data_quality import (
    DAEJEON_LAT_MAX,
    DAEJEON_LAT_MIN,
    DAEJEON_LON_MAX,
    DAEJEON_LON_MIN,
)


API_PREPROCESSING_REPORT_PATH = (
    REPORT_OUTPUT_DIR / "api_preprocessing_report.json"
)


@dataclass(frozen=True)
class APIDatasetSchema:
    name: str
    latitude_aliases: tuple[str, ...] = ()
    longitude_aliases: tuple[str, ...] = ()
    id_aliases: tuple[str, ...] = ()
    date_aliases: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = ()
    deduplicate_columns: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    nonnegative_columns: tuple[str, ...] = ()


API_SCHEMAS: dict[str, APIDatasetSchema] = {
    "speed_bump": APIDatasetSchema(
        name="speed_bump",
        latitude_aliases=("LATITUDE", "latitude", "위도", "lat"),
        longitude_aliases=("LONGITUDE", "longitude", "경도", "lon"),
        id_aliases=("SDHP_SN", "sno", "id", "과속방지턱관리번호"),
        date_aliases=("DATA_STD_DE", "데이터기준일자", "referenceDate"),
        numeric_columns=("LATITUDE", "LONGITUDE", "height", "width", "length"),
        text_columns=("LOCPLC", "address", "소재지도로명주소"),
        deduplicate_columns=("SDHP_SN",),
        nonnegative_columns=("height", "width", "length"),
    ),
    "school_zone": APIDatasetSchema(
        name="school_zone",
        latitude_aliases=("latitude", "LATITUDE", "위도", "lat"),
        longitude_aliases=("longitude", "LONGITUDE", "경도", "lon"),
        id_aliases=("sn", "id", "관리번호", "kidSafeZoneId"),
        date_aliases=("referenceDate", "데이터기준일자", "regDate"),
        numeric_columns=("latitude", "longitude"),
        text_columns=("name", "address", "roadAddress", "cctv"),
        deduplicate_columns=("sn",),
    ),
    "traffic_accident": APIDatasetSchema(
        name="traffic_accident",
        latitude_aliases=("la_crd", "latitude", "위도", "lat"),
        longitude_aliases=("lo_crd", "longitude", "경도", "lon"),
        id_aliases=("afos_fid", "accidentId", "spot_cd", "id"),
        date_aliases=(
            "occrrnc_dt",
            "accidentDate",
            "searchYear",
            "search_year",
            "searchYearCd",
        ),
        numeric_columns=(
            "la_crd",
            "lo_crd",
            "occrrnc_cnt",
            "caslt_cnt",
            "dth_dnv_cnt",
            "se_dnv_cnt",
            "sl_dnv_cnt",
            "wnd_dnv_cnt",
        ),
        text_columns=("spot_nm", "sido_sgg_nm"),
        deduplicate_columns=("afos_fid",),
        nonnegative_columns=(
            "occrrnc_cnt",
            "caslt_cnt",
            "dth_dnv_cnt",
            "se_dnv_cnt",
            "sl_dnv_cnt",
            "wnd_dnv_cnt",
        ),
    ),
}


def _first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    column_set = set(columns)
    return next((alias for alias in aliases if alias in column_set), None)


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def _coordinate_valid_mask(
    latitude: pd.Series,
    longitude: pd.Series,
) -> pd.Series:
    return latitude.between(DAEJEON_LAT_MIN, DAEJEON_LAT_MAX) & longitude.between(
        DAEJEON_LON_MIN, DAEJEON_LON_MAX
    )


def _winsorize_nonnegative(
    series: pd.Series,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> tuple[pd.Series, dict[str, float] | None]:
    numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
    valid = numeric.dropna()
    if len(valid) < 20 or np.isclose(valid.min(), valid.max()):
        return numeric, None
    lower = float(valid.quantile(lower_quantile))
    upper = float(valid.quantile(upper_quantile))
    if np.isclose(lower, upper):
        return numeric, None
    return numeric.clip(lower=lower, upper=upper), {
        "lower": round(lower, 6),
        "upper": round(upper, 6),
    }


def preprocess_api_frame(
    frame: pd.DataFrame,
    dataset_name: str,
    schema: APIDatasetSchema | None = None,
    collected_at: datetime | None = None,
    drop_invalid_coordinates: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """원본 컬럼을 보존하면서 좌표·중복·수치·시점 표준 컬럼을 추가한다."""
    schema = schema or API_SCHEMAS.get(
        dataset_name,
        APIDatasetSchema(name=dataset_name),
    )
    result = frame.copy()
    input_count = len(result)
    result.columns = [str(column).strip() for column in result.columns]

    for column in schema.text_columns:
        if column in result:
            result[column] = _clean_text(result[column])
    numeric_failures: dict[str, int] = {}
    winsorization: dict[str, dict[str, float]] = {}
    for column in schema.numeric_columns:
        if column not in result:
            continue
        original_non_missing = result[column].notna()
        numeric = pd.to_numeric(result[column], errors="coerce")
        numeric_failures[column] = int(
            (original_non_missing & numeric.isna()).sum()
        )
        if column in schema.nonnegative_columns:
            numeric, bounds = _winsorize_nonnegative(numeric)
            if bounds:
                winsorization[column] = bounds
        result[column] = numeric

    latitude_column = _first_existing(result.columns, schema.latitude_aliases)
    longitude_column = _first_existing(result.columns, schema.longitude_aliases)
    coordinate_swaps = 0
    invalid_coordinate_count = 0
    missing_coordinate_count = 0
    if latitude_column and longitude_column:
        latitude = pd.to_numeric(result[latitude_column], errors="coerce")
        longitude = pd.to_numeric(result[longitude_column], errors="coerce")
        swap_mask = latitude.between(
            DAEJEON_LON_MIN, DAEJEON_LON_MAX
        ) & longitude.between(DAEJEON_LAT_MIN, DAEJEON_LAT_MAX)
        coordinate_swaps = int(swap_mask.sum())
        swapped_latitude = latitude.copy()
        swapped_longitude = longitude.copy()
        swapped_latitude.loc[swap_mask] = longitude.loc[swap_mask]
        swapped_longitude.loc[swap_mask] = latitude.loc[swap_mask]
        latitude, longitude = swapped_latitude, swapped_longitude
        result["_latitude"] = latitude
        result["_longitude"] = longitude
        missing_mask = latitude.isna() | longitude.isna()
        valid_mask = _coordinate_valid_mask(latitude, longitude)
        missing_coordinate_count = int(missing_mask.sum())
        invalid_coordinate_count = int((~missing_mask & ~valid_mask).sum())
        result["_coordinate_valid"] = valid_mask
        if drop_invalid_coordinates:
            result = result[valid_mask].copy()
    else:
        result["_latitude"] = np.nan
        result["_longitude"] = np.nan
        result["_coordinate_valid"] = False

    date_column = _first_existing(result.columns, schema.date_aliases)
    if date_column:
        result["_observation_date"] = pd.to_datetime(
            result[date_column], errors="coerce"
        )
    else:
        result["_observation_date"] = pd.NaT

    id_column = _first_existing(result.columns, schema.id_aliases)
    if id_column:
        result["_source_record_id"] = _clean_text(result[id_column])
    else:
        result["_source_record_id"] = pd.NA

    duplicate_subset = [
        column for column in schema.deduplicate_columns if column in result
    ]
    if not duplicate_subset and id_column:
        duplicate_subset = [id_column]
    if not duplicate_subset and latitude_column and longitude_column:
        result["_coordinate_key"] = (
            result["_latitude"].round(6).astype(str)
            + ":"
            + result["_longitude"].round(6).astype(str)
        )
        duplicate_subset = ["_coordinate_key"]
    duplicate_count = int(
        result.duplicated(subset=duplicate_subset, keep="last").sum()
    ) if duplicate_subset else 0
    if duplicate_subset:
        result = result.drop_duplicates(
            subset=duplicate_subset,
            keep="last",
        ).copy()

    processed_time = datetime.now(timezone.utc)
    existing_collection_time = None
    if "_collected_at_utc" in result and result["_collected_at_utc"].notna().any():
        existing_collection_time = str(
            result.loc[result["_collected_at_utc"].notna(), "_collected_at_utc"].iloc[0]
        )
    collection_time = collected_at or processed_time
    result["_source_dataset"] = dataset_name
    result["_collected_at_utc"] = (
        existing_collection_time or collection_time.isoformat()
    )
    result["_processed_at_utc"] = processed_time.isoformat()
    result["_preprocessing_version"] = "2.0"
    result["_row_quality_score"] = (
        result["_coordinate_valid"].astype(float) * 0.60
        + result["_source_record_id"].notna().astype(float) * 0.20
        + result["_observation_date"].notna().astype(float) * 0.20
    ).round(3)

    missing_required = [
        column for column in schema.required_columns if column not in result
    ]
    report: dict[str, object] = {
        "dataset": dataset_name,
        "input_rows": input_count,
        "output_rows": int(len(result)),
        "removed_rows": input_count - int(len(result)),
        "duplicate_rows_removed": duplicate_count,
        "coordinate_columns": {
            "latitude": latitude_column,
            "longitude": longitude_column,
        },
        "coordinate_swaps_corrected": coordinate_swaps,
        "missing_coordinate_rows": missing_coordinate_count,
        "out_of_daejeon_rows": invalid_coordinate_count,
        "numeric_parse_failures": numeric_failures,
        "winsorization": winsorization,
        "missing_required_columns": missing_required,
        "collection_timestamp_utc": (
            existing_collection_time or collection_time.isoformat()
        ),
        "processing_timestamp_utc": processed_time.isoformat(),
        "policy": {
            "coordinate_bbox": {
                "latitude": [DAEJEON_LAT_MIN, DAEJEON_LAT_MAX],
                "longitude": [DAEJEON_LON_MIN, DAEJEON_LON_MAX],
            },
            "coordinate_rounding_for_deduplication": 6,
            "nonnegative_winsorization_quantiles": [0.005, 0.995],
            "raw_columns_preserved": True,
        },
    }
    return result.reset_index(drop=True), report


def update_api_preprocessing_report(
    dataset_report: dict[str, object],
    output_path: str | Path = API_PREPROCESSING_REPORT_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"preprocessing_version": "2.0", "datasets": {}}
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["datasets"][str(dataset_report["dataset"])] = dataset_report
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def preprocess_for_cmcs_analysis(
    edge_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """CMCS 통계 분석에 투입할 edge feature를 정규화·이상치 처리한다.

    단계:
      1. 수치 컬럼 강제 변환 및 결측 대체 (중앙값)
      2. 비음수 컬럼 winsorization (0.5%–99.5%)
      3. 0–1 범위 클리핑 (이미 정규화된 _norm 컬럼)
      4. IQR 기반 이상치 플래깅 (제거하지 않고 마킹)
      5. 전처리 통계 리포트 생성
    """
    result = edge_features.copy()
    report: dict[str, object] = {"input_rows": len(result)}

    norm_columns = [
        col for col in result.columns
        if col.endswith("_norm") or col.startswith("has_") or col.startswith("is_")
    ]
    count_columns = [
        col for col in result.columns
        if col.endswith("_count") and not col.endswith("_norm")
    ]
    numeric_columns = norm_columns + count_columns + ["length_m", "accident_count"]
    numeric_columns = [col for col in numeric_columns if col in result.columns]

    coercion_failures: dict[str, int] = {}
    for col in numeric_columns:
        original_valid = result[col].notna().sum()
        result[col] = pd.to_numeric(result[col], errors="coerce")
        coercion_failures[col] = int(original_valid - result[col].notna().sum())
    report["coercion_failures"] = {
        k: v for k, v in coercion_failures.items() if v > 0
    }

    median_fill: dict[str, float] = {}
    for col in numeric_columns:
        missing = int(result[col].isna().sum())
        if missing > 0:
            fill_value = float(result[col].median()) if result[col].notna().any() else 0.0
            result[col] = result[col].fillna(fill_value)
            median_fill[col] = round(fill_value, 6)
    report["median_imputation"] = median_fill

    winsorization_applied: dict[str, dict[str, float]] = {}
    for col in count_columns:
        if col not in result.columns:
            continue
        series = result[col]
        if series.nunique(dropna=True) < 3:
            continue
        clipped, bounds = _winsorize_nonnegative(series)
        if bounds:
            result[col] = clipped
            winsorization_applied[col] = bounds
    report["winsorization"] = winsorization_applied

    clipped_columns: list[str] = []
    for col in norm_columns:
        if col in result.columns:
            before_min = float(result[col].min())
            before_max = float(result[col].max())
            result[col] = result[col].clip(lower=0.0, upper=1.0)
            if before_min < -0.001 or before_max > 1.001:
                clipped_columns.append(col)
    report["range_clipped_columns"] = clipped_columns

    outlier_flags: dict[str, int] = {}
    for col in count_columns:
        if col not in result.columns:
            continue
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        upper_fence = q3 + 3.0 * iqr
        n_outliers = int((result[col] > upper_fence).sum())
        if n_outliers > 0:
            outlier_flags[col] = n_outliers
            result[f"_outlier_{col}"] = (result[col] > upper_fence).astype(int)
    report["iqr_outlier_flags"] = outlier_flags
    report["output_rows"] = len(result)
    return result, report


def preprocess_existing_api_files(
    raw_dir: str | Path = "data/raw",
) -> dict[str, dict[str, object]]:
    """캐시된 API 파일도 수집 직후와 동일한 정책으로 재정제한다."""
    raw_dir = Path(raw_dir)
    files = {
        "speed_bump": raw_dir / "speed_bump.csv",
        "school_zone": raw_dir / "school_zone.csv",
        "traffic_accident": (
            raw_dir / "daejeon_schoolzone_accident_hotspots.csv"
        ),
    }
    reports: dict[str, dict[str, object]] = {}
    for dataset_name, path in files.items():
        if not path.exists():
            continue
        frame = None
        for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            continue
        clean, report = preprocess_api_frame(frame, dataset_name)
        clean.to_csv(path, index=False, encoding="utf-8-sig")
        update_api_preprocessing_report(report)
        reports[dataset_name] = report
    return reports
