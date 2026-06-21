from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CHART_OUTPUT_DIR,
    MAP_OUTPUT_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORT_OUTPUT_DIR,
    ensure_directories,
)


EARTH_RADIUS_M = 6_371_000.0
DISTRICTS = ("대덕구", "유성구", "동구", "중구", "서구")
MODEL_FEATURES = [
    "crosswalk_count_100m",
    "crosswalk_count_300m",
    "crosswalk_count_500m",
    "signal_count_100m",
    "signal_count_300m",
    "signal_count_500m",
    "nearest_crosswalk_m",
    "nearest_signal_m",
    "pedestrian_crosswalk_signal_ratio_300m",
    "crosswalk_audio_ratio_300m",
    "tactile_block_ratio_300m",
    "raised_crosswalk_ratio_300m",
    "focused_light_ratio_300m",
    "avg_lane_count_300m",
    "actuated_signal_ratio_300m",
    "countdown_signal_ratio_300m",
    "signal_audio_ratio_300m",
    "academy_count_district",
    "illegal_parking_count_district",
]


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"CSV 인코딩을 판별하지 못했습니다: {path}") from last_error


def extract_district(value: object) -> str | None:
    match = re.search(r"(대덕구|유성구|동구|중구|서구)", str(value))
    return match.group(1) if match else None


def haversine_matrix(
    origin_latitudes: Iterable[float],
    origin_longitudes: Iterable[float],
    target_latitudes: Iterable[float],
    target_longitudes: Iterable[float],
) -> np.ndarray:
    origin_lat = np.radians(np.asarray(origin_latitudes, dtype=float))[:, None]
    origin_lon = np.radians(np.asarray(origin_longitudes, dtype=float))[:, None]
    target_lat = np.radians(np.asarray(target_latitudes, dtype=float))[None, :]
    target_lon = np.radians(np.asarray(target_longitudes, dtype=float))[None, :]

    delta_lat = target_lat - origin_lat
    delta_lon = target_lon - origin_lon
    haversine = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(origin_lat)
        * np.cos(target_lat)
        * np.sin(delta_lon / 2.0) ** 2
    )
    return EARTH_RADIUS_M * 2.0 * np.arctan2(
        np.sqrt(haversine), np.sqrt(np.clip(1.0 - haversine, 0.0, None))
    )


def _first_matching_file(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{data_dir}/{pattern} 파일을 찾지 못했습니다.")
    return matches[0]


def _academy_counts(data_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    files = sorted(data_dir.glob("*교육지원청+학원+및+교습소+현황*.xlsx"))
    if not files:
        raise FileNotFoundError("학원·교습소 XLSX 파일이 없습니다.")

    for path in files:
        workbook = pd.ExcelFile(path)
        for sheet_name in workbook.sheet_names:
            frame = pd.read_excel(path, sheet_name=sheet_name)
            address_column = next(
                (column for column in frame.columns if "주소" in str(column)), None
            )
            name_column = next(
                (
                    column
                    for column in frame.columns
                    if str(column) in {"학원명", "교습소명"}
                ),
                None,
            )
            if address_column is None or name_column is None:
                continue
            unique_places = frame[[name_column, address_column]].dropna().drop_duplicates()
            districts = unique_places[address_column].map(extract_district)
            for district, count in districts.value_counts().items():
                counts[str(district)] = counts.get(str(district), 0) + int(count)
    return counts


def _nearby_ratio(
    frame: pd.DataFrame,
    distances: np.ndarray,
    column: str,
    radius_m: float = 300.0,
) -> np.ndarray:
    values = frame[column].astype(str).str.upper().eq("Y").to_numpy()
    nearby = distances <= radius_m
    denominators = nearby.sum(axis=1)
    return np.divide(
        (nearby * values).sum(axis=1),
        denominators,
        out=np.zeros(len(denominators), dtype=float),
        where=denominators > 0,
    )


def _nearby_numeric_mean(
    frame: pd.DataFrame,
    distances: np.ndarray,
    column: str,
    radius_m: float = 300.0,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    means = []
    for distance_row in distances:
        nearby_values = values[distance_row <= radius_m]
        nearby_values = nearby_values[np.isfinite(nearby_values)]
        means.append(float(nearby_values.mean()) if len(nearby_values) else 0.0)
    return np.asarray(means)


def build_school_risk_dataset(
    data_dir: str | Path = "data",
    accident_path: str | Path = "data/raw/daejeon_schoolzone_accident_hotspots.csv",
    label_radius_m: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    school_path = _first_matching_file(data_dir, "*초중등학교위치*.csv")
    crosswalk_path = _first_matching_file(data_dir, "*횡단보도*.csv")
    signal_path = _first_matching_file(data_dir, "*신호등*.csv")
    parking_path = _first_matching_file(data_dir, "*불법주정차*.csv")

    schools = read_csv_flexible(school_path)
    schools = schools[
        schools["시도교육청명"].eq("대전광역시교육청")
        & schools["학교급구분"].eq("초등학교")
        & schools["운영상태"].eq("운영")
    ].copy()
    schools = schools.dropna(subset=["위도", "경도"]).reset_index(drop=True)
    schools["district"] = schools["소재지도로명주소"].map(extract_district)

    crosswalks = read_csv_flexible(crosswalk_path).dropna(
        subset=["위도", "경도"]
    )
    signals = read_csv_flexible(signal_path).dropna(subset=["위도", "경도"])
    hotspots = read_csv_flexible(accident_path).dropna(
        subset=["la_crd", "lo_crd"]
    )
    parking = read_csv_flexible(parking_path)
    parking["district"] = parking["자치구"].map(extract_district)
    parking_counts = dict(
        zip(parking["district"], pd.to_numeric(parking["단속건수"], errors="coerce"))
    )
    academy_counts = _academy_counts(data_dir)

    school_latitudes = schools["위도"].astype(float).to_numpy()
    school_longitudes = schools["경도"].astype(float).to_numpy()
    crosswalk_distances = haversine_matrix(
        school_latitudes,
        school_longitudes,
        crosswalks["위도"],
        crosswalks["경도"],
    )
    signal_distances = haversine_matrix(
        school_latitudes,
        school_longitudes,
        signals["위도"],
        signals["경도"],
    )
    hotspot_distances = haversine_matrix(
        school_latitudes,
        school_longitudes,
        hotspots["la_crd"],
        hotspots["lo_crd"],
    )

    features = pd.DataFrame(
        {
            "school_id": schools["학교ID"].astype(str),
            "school_name": schools["학교명"].astype(str),
            "district": schools["district"],
            "address": schools["소재지도로명주소"].astype(str),
            "latitude": school_latitudes,
            "longitude": school_longitudes,
        }
    )
    for radius in (100, 300, 500):
        features[f"crosswalk_count_{radius}m"] = (
            crosswalk_distances <= radius
        ).sum(axis=1)
        features[f"signal_count_{radius}m"] = (signal_distances <= radius).sum(
            axis=1
        )
    features["nearest_crosswalk_m"] = crosswalk_distances.min(axis=1)
    features["nearest_signal_m"] = signal_distances.min(axis=1)

    crosswalk_ratio_columns = {
        "보행자신호등유무": "pedestrian_crosswalk_signal_ratio_300m",
        "음향신호기설치여부": "crosswalk_audio_ratio_300m",
        "점자블록유무": "tactile_block_ratio_300m",
        "고원식적용여부": "raised_crosswalk_ratio_300m",
        "집중조명시설유무": "focused_light_ratio_300m",
    }
    for source, target in crosswalk_ratio_columns.items():
        features[target] = _nearby_ratio(
            crosswalks, crosswalk_distances, source
        )
    features["avg_lane_count_300m"] = _nearby_numeric_mean(
        crosswalks, crosswalk_distances, "차로수"
    )

    signal_ratio_columns = {
        "보행자작동신호기유무": "actuated_signal_ratio_300m",
        "잔여시간표시기유무": "countdown_signal_ratio_300m",
        "시각장애인용음향신호기유무": "signal_audio_ratio_300m",
    }
    for source, target in signal_ratio_columns.items():
        features[target] = _nearby_ratio(signals, signal_distances, source)

    features["academy_count_district"] = (
        features["district"].map(academy_counts).fillna(0).astype(int)
    )
    features["illegal_parking_count_district"] = (
        features["district"].map(parking_counts).fillna(0).astype(int)
    )

    nearest_indices = hotspot_distances.argmin(axis=1)
    features["nearest_hotspot_m"] = hotspot_distances.min(axis=1)
    features["nearest_hotspot_year"] = hotspots.iloc[nearest_indices][
        "search_year"
    ].to_numpy()
    features["nearest_hotspot_name"] = hotspots.iloc[nearest_indices][
        "spot_nm"
    ].to_numpy()
    features["accident_hotspot_within_radius"] = (
        hotspot_distances <= label_radius_m
    ).any(axis=1).astype(int)
    features["label_radius_m"] = float(label_radius_m)
    return features, hotspots


def _classification_metrics(
    target: pd.Series, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, object]:
    predictions = probabilities >= threshold
    return {
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 6),
        "average_precision": round(
            float(average_precision_score(target, probabilities)), 6
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(target, predictions)), 6
        ),
        "precision": round(
            float(precision_score(target, predictions, zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(target, predictions, zero_division=0)), 6
        ),
        "f1": round(float(f1_score(target, predictions, zero_division=0)), 6),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(target, predictions).tolist(),
    }


def train_school_hotspot_model(
    dataset: pd.DataFrame,
    model_path: str | Path = MODEL_DIR / "real_school_hotspot_model.pkl",
    report_path: str | Path = REPORT_OUTPUT_DIR / "real_data_model_report.json",
) -> tuple[object, dict[str, object], pd.DataFrame]:
    X = dataset[MODEL_FEATURES]
    y = dataset["accident_hotspot_within_radius"].astype(int)
    groups = dataset["district"]
    if y.nunique() < 2:
        raise ValueError("사고 다발지역 라벨에 두 개 클래스가 필요합니다.")

    candidates = {
        "DummyPrior": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        C=0.5,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    validation = LeaveOneGroupOut()
    model_results: dict[str, dict[str, object]] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}
    for name, model in candidates.items():
        probabilities = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=validation,
            method="predict_proba",
        )[:, 1]
        probabilities_by_model[name] = probabilities
        model_results[name] = _classification_metrics(y, probabilities)

    selectable = [name for name in candidates if name != "DummyPrior"]
    best_name = max(
        selectable,
        key=lambda name: model_results[name]["average_precision"],
    )
    best_model = candidates[best_name]
    best_model.fit(X, y)

    scored = dataset.copy()
    scored["out_of_district_cv_probability"] = probabilities_by_model[best_name]
    scored["fitted_probability"] = best_model.predict_proba(X)[:, 1]
    scored = scored.sort_values(
        "out_of_district_cv_probability", ascending=False
    ).reset_index(drop=True)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": MODEL_FEATURES,
            "label": "accident_hotspot_within_radius",
            "validation": "LeaveOneGroupOut by district",
        },
        model_path,
    )

    positive_count = int(y.sum())
    report: dict[str, object] = {
        "dataset": {
            "school_count": int(len(dataset)),
            "positive_count": positive_count,
            "negative_count": int(len(dataset) - positive_count),
            "positive_rate": round(float(y.mean()), 6),
            "district_distribution": dataset.groupby("district")[
                "accident_hotspot_within_radius"
            ]
            .agg(["count", "sum"])
            .to_dict(orient="index"),
        },
        "validation": {
            "method": "LeaveOneGroupOut",
            "group": "district",
            "reason": "같은 구의 공간 패턴이 학습·검증에 동시에 들어가는 누수를 줄이기 위함",
        },
        "models": model_results,
        "best_model": best_name,
        "deployment_ready": (
            model_results[best_name]["roc_auc"] >= 0.7
            and model_results[best_name]["average_precision"]
            >= max(0.3, float(y.mean()) * 2)
        ),
        "limitations": [
            "사고 라벨은 개별 사고 전체가 아니라 공공 API의 어린이보호구역 사고 다발지역이다.",
            "사고 자료는 2012~2024년, 시설 자료는 2022~2026년 스냅샷으로 시점이 일치하지 않는다.",
            "양성 표본이 적어 모델 성능 수치의 불확실성이 크다.",
            "이 결과는 사고 인과관계나 실제 무사고 지역을 증명하지 않는다.",
        ],
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return best_model, report, scored


def verify_repeated_accident_area(
    hotspots: pd.DataFrame,
    schools: pd.DataFrame,
    cluster_radius_m: float = 400.0,
    output_path: str | Path = REPORT_OUTPUT_DIR
    / "verified_accident_locations.csv",
) -> tuple[pd.DataFrame, dict[str, object]]:
    coordinates_radians = np.radians(
        hotspots[["la_crd", "lo_crd"]].astype(float).to_numpy()
    )
    clusterer = DBSCAN(
        eps=cluster_radius_m / EARTH_RADIUS_M,
        min_samples=1,
        metric="haversine",
    )
    clustered = hotspots.copy()
    clustered["cluster_id"] = clusterer.fit_predict(coordinates_radians)

    cluster_summary = (
        clustered.groupby("cluster_id")
        .agg(
            distinct_years=("search_year", "nunique"),
            first_year=("search_year", "min"),
            last_year=("search_year", "max"),
            hotspot_records=("search_year", "size"),
            accident_count=("occrrnc_cnt", "sum"),
            casualty_count=("caslt_cnt", "sum"),
            death_count=("dth_dnv_cnt", "sum"),
            latitude=("la_crd", "mean"),
            longitude=("lo_crd", "mean"),
        )
        .reset_index()
    )
    cluster_summary = cluster_summary.sort_values(
        ["death_count", "distinct_years", "accident_count", "casualty_count"],
        ascending=False,
    )
    selected_cluster = int(cluster_summary.iloc[0]["cluster_id"])
    verified = clustered[clustered["cluster_id"].eq(selected_cluster)].copy()
    verified = verified.sort_values("search_year")

    center_latitude = float(verified["la_crd"].astype(float).mean())
    center_longitude = float(verified["lo_crd"].astype(float).mean())
    school_distances = haversine_matrix(
        [center_latitude],
        [center_longitude],
        schools["latitude"],
        schools["longitude"],
    )[0]
    nearest_school_index = int(np.argmin(school_distances))
    nearest_school = schools.iloc[nearest_school_index]

    summary = {
        "cluster_radius_m": cluster_radius_m,
        "center_latitude": center_latitude,
        "center_longitude": center_longitude,
        "years": sorted(verified["search_year"].astype(int).unique().tolist()),
        "accident_count": int(verified["occrrnc_cnt"].sum()),
        "casualty_count": int(verified["caslt_cnt"].sum()),
        "death_count": int(verified["dth_dnv_cnt"].sum()),
        "nearest_school": str(nearest_school["school_name"]),
        "nearest_school_address": str(nearest_school["address"]),
        "nearest_school_distance_m": round(
            float(school_distances[nearest_school_index]), 1
        ),
        "spot_names": verified["spot_nm"].astype(str).tolist(),
    }
    for key, value in summary.items():
        if isinstance(value, list):
            verified[key] = " | ".join(map(str, value))
        else:
            verified[key] = value

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(output_path, index=False, encoding="utf-8-sig")
    return verified, summary


def create_accident_verification_map(
    verified: pd.DataFrame,
    summary: dict[str, object],
    output_path: str | Path = MAP_OUTPUT_DIR
    / "verified_accident_hotspots.html",
) -> Path:
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("검증 지도 생성에는 folium이 필요합니다.") from exc

    center = [summary["center_latitude"], summary["center_longitude"]]
    map_object = folium.Map(
        location=center, zoom_start=16, tiles="CartoDB positron"
    )
    for _, row in verified.iterrows():
        popup = (
            f"<b>{row['spot_nm']}</b><br>"
            f"기준연도: {int(row['search_year'])}<br>"
            f"사고: {int(row['occrrnc_cnt'])}건<br>"
            f"사상자: {int(row['caslt_cnt'])}명<br>"
            f"사망: {int(row['dth_dnv_cnt'])}명"
        )
        location = [float(row["la_crd"]), float(row["lo_crd"])]
        folium.Circle(
            location=location,
            radius=300,
            color="#dc2626",
            fill=True,
            fill_opacity=0.12,
            popup=popup,
        ).add_to(map_object)
        folium.Marker(
            location=location,
            tooltip=f"{int(row['search_year'])}년 사고 다발지역",
            popup=popup,
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(map_object)

    folium.Marker(
        location=center,
        tooltip=(
            f"인접 학교: {summary['nearest_school']} "
            f"({summary['nearest_school_distance_m']}m)"
        ),
        icon=folium.Icon(color="blue", icon="education"),
    ).add_to(map_object)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(output_path)
    return output_path


def plot_model_feature_weights(
    model: object,
    output_path: str | Path = CHART_OUTPUT_DIR
    / "real_model_feature_importance.png",
) -> Path:
    import matplotlib.pyplot as plt

    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        weights = np.abs(estimator.coef_[0])
    else:
        weights = estimator.feature_importances_
    importance = pd.Series(weights, index=MODEL_FEATURES).sort_values().tail(12)

    figure, axis = plt.subplots(figsize=(9, 6))
    importance.plot.barh(ax=axis, color="#2563eb")
    axis.set_title("Actual-data school hotspot model feature weights")
    axis.set_xlabel("Absolute coefficient / feature importance")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def run_real_data_pipeline(
    data_dir: str | Path = "data",
    accident_path: str | Path = "data/raw/daejeon_schoolzone_accident_hotspots.csv",
    label_radius_m: float = 300.0,
) -> dict[str, object]:
    ensure_directories()
    dataset, hotspots = build_school_risk_dataset(
        data_dir=data_dir,
        accident_path=accident_path,
        label_radius_m=label_radius_m,
    )
    feature_path = PROCESSED_DATA_DIR / "daejeon_school_risk_features.csv"
    dataset.to_csv(feature_path, index=False, encoding="utf-8-sig")

    model, report, scored = train_school_hotspot_model(dataset)
    scored_path = REPORT_OUTPUT_DIR / "school_hotspot_predictions.csv"
    scored.to_csv(scored_path, index=False, encoding="utf-8-sig")

    verified, verification_summary = verify_repeated_accident_area(
        hotspots, dataset
    )
    map_path = create_accident_verification_map(
        verified, verification_summary
    )
    chart_path = plot_model_feature_weights(model)

    summary_path = REPORT_OUTPUT_DIR / "verified_accident_summary.json"
    summary_path.write_text(
        json.dumps(verification_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "dataset": dataset,
        "model_report": report,
        "scored": scored,
        "verification": verification_summary,
        "feature_path": feature_path,
        "scored_path": scored_path,
        "map_path": map_path,
        "chart_path": chart_path,
        "summary_path": summary_path,
    }

