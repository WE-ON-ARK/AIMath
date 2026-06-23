from __future__ import annotations

import ast
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from pypdf import PdfReader
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    CHART_OUTPUT_DIR,
    GRAPH_DATA_DIR,
    MAP_OUTPUT_DIR,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORT_OUTPUT_DIR,
    SETTINGS,
    ensure_directories,
)
from src.api_preprocessing import (
    preprocess_api_frame,
    preprocess_existing_api_files,
    update_api_preprocessing_report,
)
from src.cmcs_calculator import CMCSCalculator
from src.data_driven_cmcs import (
    DATA_DRIVEN_CHART_PATH,
    DATA_DRIVEN_REPORT_PATH,
    DATA_DRIVEN_WEIGHTS_PATH,
    derive_data_driven_cmcs_weights,
)
from src.real_data_pipeline import (
    DISTRICTS,
    extract_district,
    haversine_matrix,
    read_csv_flexible,
)
from src.route_optimizer import RouteOptimizer
from src.route_validator import batch_od_evaluation


PROJECTED_CRS = "EPSG:5179"
FULL_GRAPH_PATH = GRAPH_DATA_DIR / "daejeon_walk.graphml"
SCORED_GRAPH_PATH = GRAPH_DATA_DIR / "daejeon_walk_cmcs.graphml"
EDGE_FEATURE_PATH = PROCESSED_DATA_DIR / "daejeon_edge_features.csv"
EDGE_CMCS_PATH = PROCESSED_DATA_DIR / "daejeon_edge_cmcs.csv"
EDGE_MODEL_PATH = MODEL_DIR / "edge_accident_risk_model.pkl"
EDGE_MODEL_REPORT_PATH = REPORT_OUTPUT_DIR / "edge_model_report.json"
REGIONAL_MODEL_PATH = MODEL_DIR / "regional_xgboost_risk_model.pkl"
REGIONAL_MODEL_REPORT_PATH = REPORT_OUTPUT_DIR / "regional_boosting_report.json"
FULL_PIPELINE_REPORT_PATH = REPORT_OUTPUT_DIR / "full_pipeline_report.json"
REGIONAL_CELL_SIZE_M = 1750
REGIONAL_ENSEMBLE_SEEDS = (7, 17, 29, 42, 61, 91, 113)
REGIONAL_NESTED_SEEDS = (17, 42, 91)

EDGE_MODEL_FEATURES = [
    "traffic_volume_norm",
    "avg_speed_norm",
    "narrow_sidewalk_norm",
    "slope_norm",
    "is_alley",
    "pedestrian_flow_norm",
    "academy_density_norm",
    "bus_stop_nearby_norm",
    "illegal_parking_norm",
    "light_density_norm",
    "has_crosswalk",
    "has_signal",
    "lane_count_norm",
    "has_speed_bump",
    "has_cctv",
    "is_school_zone",
    "crosswalk_count_norm",
    "signal_count_norm",
    "speed_bump_count_norm",
]

DISTRICT_QUERIES = {
    "대덕구": "Daedeok-gu, Daejeon, South Korea",
    "동구": "Dong-gu, Daejeon, South Korea",
    "중구": "Jung-gu, Daejeon, South Korea",
    "서구": "Seo-gu, Daejeon, South Korea",
    "유성구": "Yuseong-gu, Daejeon, South Korea",
}

HIGHWAY_TRAFFIC_PROXY = {
    "motorway": 1.00,
    "motorway_link": 0.95,
    "trunk": 0.95,
    "trunk_link": 0.90,
    "primary": 0.88,
    "primary_link": 0.84,
    "secondary": 0.76,
    "secondary_link": 0.72,
    "tertiary": 0.62,
    "tertiary_link": 0.58,
    "residential": 0.34,
    "unclassified": 0.30,
    "living_street": 0.18,
    "service": 0.16,
    "pedestrian": 0.08,
    "footway": 0.04,
    "path": 0.03,
    "steps": 0.01,
    "track": 0.08,
}

HIGHWAY_SPEED_PROXY = {
    "motorway": 100.0,
    "motorway_link": 60.0,
    "trunk": 80.0,
    "trunk_link": 55.0,
    "primary": 60.0,
    "primary_link": 45.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 35.0,
    "residential": 30.0,
    "unclassified": 30.0,
    "living_street": 15.0,
    "service": 15.0,
    "pedestrian": 5.0,
    "footway": 5.0,
    "path": 5.0,
    "steps": 3.0,
    "track": 10.0,
}

REGIONAL_DISTRIBUTION_FEATURES = [
    "traffic_volume_norm",
    "avg_speed_norm",
    "narrow_sidewalk_norm",
    "slope_norm",
    "pedestrian_flow_norm",
    "bus_stop_nearby_norm",
    "illegal_parking_norm",
    "lane_count_norm",
]

REGIONAL_SUM_FEATURES = [
    "length_m",
    "crosswalk_count",
    "signal_count",
    "speed_bump_count",
    "bus_stop_count",
    "streetlight_count",
    "school_zone_cctv_count",
    "is_school_zone",
    "has_crosswalk",
    "has_signal",
    "has_speed_bump",
    "has_cctv",
]

REGIONAL_HIGHWAY_CLASSES = tuple(HIGHWAY_TRAFFIC_PROXY) + ("other",)
REGIONAL_DERIVED_FEATURES = [
    "log_segment_count",
    "road_length_km",
    "crosswalk_per_km",
    "signal_per_km",
    "speed_bump_per_km",
    "bus_stop_per_km",
    "streetlight_per_km",
    "school_zone_cctv_per_km",
    "school_zone_share",
    "crosswalk_coverage",
    "signal_coverage",
    "speed_bump_coverage",
    "cctv_coverage",
    "structural_hazard_mean",
    "structural_hazard_peak",
]

REGIONAL_MODEL_FEATURES = (
    EDGE_MODEL_FEATURES
    + ["center_x", "center_y", "segment_count"]
    + [
        f"{feature}_{statistic}"
        for statistic in ("max", "std", "q90")
        for feature in REGIONAL_DISTRIBUTION_FEATURES
    ]
    + [f"{feature}_sum" for feature in REGIONAL_SUM_FEATURES]
    + [f"hw_share_{highway}" for highway in REGIONAL_HIGHWAY_CLASSES]
    + REGIONAL_DERIVED_FEATURES
)


def _api_key_from_environment_or_pdf(pdf_path: Path) -> str:
    if SETTINGS.api_key:
        return SETTINGS.api_key
    text = "\n".join(
        page.extract_text() or "" for page in PdfReader(pdf_path).pages
    )
    matches = re.findall(r"\b[0-9a-fA-F]{64}\b", text)
    if not matches:
        raise ValueError(
            f"API 인증키를 찾지 못했습니다. DATA_GO_KR_API_KEY를 설정하세요: {pdf_path}"
        )
    return matches[0]


def collect_speed_bumps(
    output_path: str | Path = RAW_DATA_DIR / "speed_bump.csv",
) -> pd.DataFrame:
    key = _api_key_from_environment_or_pdf(
        Path("data/과속방지턱조회서비스.pdf")
    )
    response = requests.get(
        "https://apis.data.go.kr/6300000/GetSdhpListService/getSdhpList",
        params={
            "serviceKey": key,
            "pageNo": 1,
            "numOfRows": 10000,
            "type": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    header = payload.get("response", {}).get("header", {})
    if str(header.get("resultCode")) != "00":
        raise RuntimeError(f"과속방지턱 API 오류: {header}")
    items = (
        payload.get("response", {})
        .get("body", {})
        .get("items", {})
        .get("item", [])
    )
    if isinstance(items, dict):
        items = [items]
    frame, preprocessing_report = preprocess_api_frame(
        pd.DataFrame(items),
        "speed_bump",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    update_api_preprocessing_report(preprocessing_report)
    return frame


def collect_school_zones(
    output_path: str | Path = RAW_DATA_DIR / "school_zone.csv",
) -> pd.DataFrame:
    key = _api_key_from_environment_or_pdf(Path("data/어린이보호구역 정보.pdf"))
    endpoint = (
        "https://apis.data.go.kr/6300000/"
        "kidSafeDaejeonService/kidSafeDaejeonList"
    )
    rows: list[dict[str, str | None]] = []
    page = 1
    total_count: int | None = None
    while True:
        response = requests.get(
            endpoint,
            params={"serviceKey": key, "pageNo": page, "numOfRows": 100},
            timeout=60,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return_code = root.findtext(".//returnCode")
        if return_code != "00":
            raise RuntimeError(
                f"어린이보호구역 API 오류: {root.findtext('.//returnMessage')}"
            )
        if total_count is None:
            total_count = int(root.findtext(".//totalCount") or 0)
        page_items = [
            {child.tag: child.text for child in node}
            for node in root.findall(".//MsgBody/items")
        ]
        rows.extend(page_items)
        if not page_items or len(rows) >= total_count:
            break
        page += 1
    frame, preprocessing_report = preprocess_api_frame(
        pd.DataFrame(rows),
        "school_zone",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    update_api_preprocessing_report(preprocessing_report)
    return frame


def collect_osm_point_features(
    place: str = "Daejeon, South Korea",
) -> dict[str, Path]:
    """OSM에서 버스 정류장 및 가로등 포인트 데이터를 수집한다.

    가로등은 highway=street_lamp 외에 amenity=street_lamp,
    power=pole+street_lamp=yes 태그도 함께 쿼리해 커버리지를 높인다.
    """
    ox.settings.use_cache = True
    outputs: dict[str, Path] = {}

    tag_sets: dict[str, list[dict[str, object]]] = {
        "bus_stop": [{"highway": "bus_stop"}],
        "streetlight": [
            {"highway": "street_lamp"},
            {"amenity": "street_lamp"},
        ],
    }
    for name, tag_list in tag_sets.items():
        path = RAW_DATA_DIR / f"{name}.geojson"
        frames = []
        for tags in tag_list:
            try:
                feats = ox.features_from_place(place, tags).reset_index()
                feats = feats[feats.geometry.geom_type.eq("Point")]
                columns = [
                    col
                    for col in ("element_type", "osmid", "name", "highway", "amenity", "geometry")
                    if col in feats.columns
                ]
                frames.append(feats[columns])
            except Exception as exc:
                print(f"[OSM 보조 데이터 생략] {name}/{tags}: {exc}")
        if frames:
            import geopandas as _gpd
            combined = _gpd.GeoDataFrame(
                pd.concat(frames, ignore_index=True).drop_duplicates(subset=["osmid"] if "osmid" in frames[0].columns else None),
                crs="EPSG:4326",
            )
            combined.to_file(path, driver="GeoJSON")
            outputs[name] = path
            print(f"[OSM] {name}: {len(combined)}개 수집")
    return outputs


def collect_available_real_data(refresh: bool = False) -> dict[str, int]:
    ensure_directories()
    if not refresh:
        preprocess_existing_api_files(RAW_DATA_DIR)
    counts: dict[str, int] = {}
    speed_path = RAW_DATA_DIR / "speed_bump.csv"
    zone_path = RAW_DATA_DIR / "school_zone.csv"
    if refresh or not speed_path.exists():
        counts["speed_bump"] = len(collect_speed_bumps(speed_path))
    else:
        counts["speed_bump"] = len(read_csv_flexible(speed_path))
    if refresh or not zone_path.exists():
        counts["school_zone"] = len(collect_school_zones(zone_path))
    else:
        counts["school_zone"] = len(read_csv_flexible(zone_path))
    if refresh or not (RAW_DATA_DIR / "bus_stop.geojson").exists():
        collect_osm_point_features()
    for name in ("bus_stop", "streetlight"):
        path = RAW_DATA_DIR / f"{name}.geojson"
        counts[name] = len(gpd.read_file(path)) if path.exists() else 0
    return counts


def build_or_load_walk_graph(
    graph_path: str | Path = FULL_GRAPH_PATH,
    place: str = "Daejeon, South Korea",
    refresh: bool = False,
) -> nx.MultiDiGraph:
    path = Path(graph_path)
    if path.exists() and not refresh:
        return ox.load_graphml(path)
    graph = ox.graph_from_place(place, network_type="walk")
    _assign_edge_identifiers(graph)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=path)
    return graph


def _normalize_osmid(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return "-".join(sorted(map(str, value)))
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return "-".join(sorted(map(str, parsed)))
        except (SyntaxError, ValueError):
            pass
    return re.sub(r"[^0-9A-Za-z_-]+", "-", text)


def canonical_segment_id(u: object, v: object, osmid: object = "") -> str:
    left, right = sorted((str(u), str(v)))
    return f"S-{left}-{right}-{_normalize_osmid(osmid)}"


def _assign_edge_identifiers(graph: nx.MultiDiGraph) -> None:
    for u, v, key, data in graph.edges(keys=True, data=True):
        data["edge_id"] = f"E-{u}-{v}-{key}"
        data["segment_id"] = canonical_segment_id(u, v, data.get("osmid", ""))


def _first_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return str(next(iter(value), ""))
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)) and parsed:
                return str(next(iter(parsed)))
        except (SyntaxError, ValueError):
            pass
    return text


def _parse_number(value: object, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    values = re.findall(r"\d+(?:\.\d+)?", _first_value(value))
    if not values:
        return default
    return float(values[0])


def _highway_value(value: object) -> str:
    return _first_value(value).lower()


def _road_base_features(row: pd.Series) -> dict[str, float]:
    highway = _highway_value(row.get("highway", "unclassified"))
    traffic = HIGHWAY_TRAFFIC_PROXY.get(highway, 0.25)
    default_speed = HIGHWAY_SPEED_PROXY.get(highway, 25.0)
    speed = _parse_number(row.get("maxspeed"), default_speed)
    lanes = _parse_number(row.get("lanes"), 1.0)
    sidewalk = str(row.get("sidewalk", "")).lower()
    foot_only = highway in {"footway", "pedestrian", "path", "steps"}
    arterial = highway in {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "motorway_link",
        "trunk_link",
        "primary_link",
        "secondary_link",
        "tertiary_link",
    }
    if foot_only:
        narrow_sidewalk = 0.05
    elif sidewalk in {"both", "left", "right", "yes", "separate"}:
        narrow_sidewalk = 0.20
    elif sidewalk in {"no", "none"}:
        narrow_sidewalk = 1.0
    elif arterial:
        narrow_sidewalk = 0.72
    else:
        narrow_sidewalk = 0.55

    incline = abs(_parse_number(row.get("incline"), 0.0))
    slope = min(incline / 15.0, 1.0)
    lit_tag = str(row.get("lit", "")).lower()
    if lit_tag in {"yes", "24/7", "automatic"}:
        lit_score = 1.0
    elif lit_tag == "no":
        lit_score = 0.0
    else:
        lit_score = 0.5
    return {
        "traffic_volume_proxy": traffic,
        "avg_speed_kph": speed,
        "lane_count": lanes,
        "narrow_sidewalk_proxy": narrow_sidewalk,
        "slope_proxy": slope,
        "is_alley": float(highway in {"service", "living_street"}),
        "lit_tag_score": lit_score,
    }


def _academy_and_parking_counts(data_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    academy_counts: dict[str, int] = {}
    for path in sorted(data_dir.glob("*교육지원청+학원+및+교습소+현황*.xlsx")):
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
            places = frame[[name_column, address_column]].dropna().drop_duplicates()
            for district, count in (
                places[address_column].map(extract_district).value_counts().items()
            ):
                academy_counts[str(district)] = (
                    academy_counts.get(str(district), 0) + int(count)
                )

    parking_path = next(data_dir.glob("*불법주정차*.csv"))
    parking = read_csv_flexible(parking_path)
    parking["district"] = parking["자치구"].map(extract_district)
    parking_counts = {
        str(district): int(count)
        for district, count in zip(parking["district"], parking["단속건수"])
        if pd.notna(district)
    }
    return academy_counts, parking_counts


def _load_point_datasets(data_dir: Path) -> dict[str, gpd.GeoDataFrame]:
    crosswalk = read_csv_flexible(next(data_dir.glob("*횡단보도*.csv")))
    signal = read_csv_flexible(next(data_dir.glob("*신호등*.csv")))
    speed_bump = read_csv_flexible(RAW_DATA_DIR / "speed_bump.csv")
    school_zone = read_csv_flexible(RAW_DATA_DIR / "school_zone.csv")
    accident_raw = read_csv_flexible(
        RAW_DATA_DIR / "daejeon_schoolzone_accident_hotspots.csv"
    )
    accident, accident_preprocessing_report = preprocess_api_frame(
        accident_raw,
        "traffic_accident",
    )
    update_api_preprocessing_report(accident_preprocessing_report)

    def points(
        frame: pd.DataFrame, lon_column: str, lat_column: str
    ) -> gpd.GeoDataFrame:
        clean = frame.copy()
        clean[lon_column] = pd.to_numeric(clean[lon_column], errors="coerce")
        clean[lat_column] = pd.to_numeric(clean[lat_column], errors="coerce")
        clean = clean.dropna(subset=[lon_column, lat_column])
        return gpd.GeoDataFrame(
            clean,
            geometry=gpd.points_from_xy(clean[lon_column], clean[lat_column]),
            crs="EPSG:4326",
        )

    datasets = {
        "crosswalk": points(crosswalk, "경도", "위도"),
        "signal": points(signal, "경도", "위도"),
        "speed_bump": points(speed_bump, "LONGITUDE", "LATITUDE"),
        "school_zone": points(school_zone, "longitude", "latitude"),
        "accident": points(accident, "lo_crd", "la_crd"),
    }
    for name in ("bus_stop", "streetlight"):
        path = RAW_DATA_DIR / f"{name}.geojson"
        if path.exists():
            frame = gpd.read_file(path)
            frame = frame[frame.geometry.geom_type.eq("Point")].copy()
            datasets[name] = frame.to_crs("EPSG:4326")
    return datasets


def _district_polygons(refresh: bool = False) -> gpd.GeoDataFrame:
    path = RAW_DATA_DIR / "daejeon_districts.geojson"
    if path.exists() and not refresh:
        return gpd.read_file(path)
    rows = []
    for district, query in DISTRICT_QUERIES.items():
        frame = ox.geocode_to_gdf(query)
        frame["district"] = district
        rows.append(frame[["district", "geometry"]])
    result = gpd.GeoDataFrame(
        pd.concat(rows, ignore_index=True), crs="EPSG:4326"
    )
    result.to_file(path, driver="GeoJSON")
    return result


def _minmax(series: pd.Series, log_scale: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    if log_scale:
        values = np.log1p(values.clip(lower=0))
    minimum, maximum = float(values.min()), float(values.max())
    if np.isclose(minimum, maximum):
        return pd.Series(0.0, index=series.index)
    return (values - minimum) / (maximum - minimum)


def _combine_points_for_nearest_edge(
    datasets: dict[str, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    frames = []
    for name in ("crosswalk", "signal", "speed_bump", "bus_stop", "streetlight"):
        if name not in datasets or datasets[name].empty:
            continue
        frame = datasets[name][["geometry"]].copy()
        frame["dataset"] = name
        frame["source_index"] = datasets[name].index.astype(str)
        frames.append(frame)
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs="EPSG:4326"
    )


def build_edge_feature_table(
    graph: nx.MultiDiGraph,
    data_dir: str | Path = "data",
    output_path: str | Path = EDGE_FEATURE_PATH,
) -> tuple[pd.DataFrame, nx.MultiDiGraph]:
    _assign_edge_identifiers(graph)
    graph_projected = ox.project_graph(graph, to_crs=PROJECTED_CRS)
    edges = ox.graph_to_gdfs(graph_projected, nodes=False).reset_index()
    base_records = []
    for _, row in edges.iterrows():
        base = _road_base_features(row)
        base_records.append(
            {
                "edge_id": str(row["edge_id"]),
                "segment_id": str(row["segment_id"]),
                "u": row["u"],
                "v": row["v"],
                "key": row["key"],
                "length_m": float(row.get("length", 1.0)),
                "highway": _highway_value(row.get("highway", "")),
                "geometry": row.geometry,
                **base,
            }
        )
    directed = gpd.GeoDataFrame(base_records, crs=PROJECTED_CRS)
    segments = (
        directed.sort_values("length_m")
        .drop_duplicates("segment_id")
        .reset_index(drop=True)
    )
    segment_centers = segments.geometry.interpolate(0.5, normalized=True)
    segments["center_x"] = segment_centers.x
    segments["center_y"] = segment_centers.y

    datasets = _load_point_datasets(Path(data_dir))
    combined = _combine_points_for_nearest_edge(datasets)
    if not combined.empty:
        combined_projected = combined.to_crs(PROJECTED_CRS)
        nearest = ox.distance.nearest_edges(
            graph_projected,
            X=combined_projected.geometry.x.to_numpy(),
            Y=combined_projected.geometry.y.to_numpy(),
        )
        edge_to_segment = {
            (row.u, row.v, row.key): row.segment_id
            for row in directed.itertuples()
        }
        combined["segment_id"] = [
            edge_to_segment[(u, v, key)] for u, v, key in nearest
        ]
        counts = (
            combined.groupby(["segment_id", "dataset"])
            .size()
            .unstack(fill_value=0)
        )
        for name in ("crosswalk", "signal", "speed_bump", "bus_stop", "streetlight"):
            segments[f"{name}_count"] = segments["segment_id"].map(
                counts[name] if name in counts else {}
            ).fillna(0)
    else:
        for name in ("crosswalk", "signal", "speed_bump", "bus_stop", "streetlight"):
            segments[f"{name}_count"] = 0

    tree = cKDTree(segments[["center_x", "center_y"]].to_numpy())
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)

    school_zones = datasets["school_zone"].copy()
    sx, sy = transformer.transform(
        school_zones.geometry.x.to_numpy(), school_zones.geometry.y.to_numpy()
    )
    segments["is_school_zone"] = 0
    segments["school_zone_cctv_count"] = 0
    cctv_values = school_zones["cctv"].astype(str).str.strip().eq("설치").to_numpy()
    for x, y, has_cctv in zip(sx, sy, cctv_values):
        indices = tree.query_ball_point([x, y], r=200.0)
        if not indices:
            continue
        segments.loc[indices, "is_school_zone"] = 1
        if has_cctv:
            segments.loc[indices, "school_zone_cctv_count"] += 1

    accidents = datasets["accident"].copy()
    ax, ay = transformer.transform(
        accidents.geometry.x.to_numpy(), accidents.geometry.y.to_numpy()
    )
    segments["accident_count"] = 0.0
    segments["casualty_count"] = 0.0
    segments["death_count"] = 0.0
    for point_index, (x, y) in enumerate(zip(ax, ay)):
        indices = tree.query_ball_point([x, y], r=300.0)
        if not indices:
            continue
        row = accidents.iloc[point_index]
        segments.loc[indices, "accident_count"] += float(row["occrrnc_cnt"])
        segments.loc[indices, "casualty_count"] += float(row["caslt_cnt"])
        segments.loc[indices, "death_count"] += float(row["dth_dnv_cnt"])

    polygons = _district_polygons().to_crs(PROJECTED_CRS)
    center_gdf = gpd.GeoDataFrame(
        segments[["segment_id"]],
        geometry=segment_centers,
        crs=PROJECTED_CRS,
    )
    joined = gpd.sjoin(
        center_gdf,
        polygons[["district", "geometry"]],
        how="left",
        predicate="within",
    )
    district_map = (
        joined.dropna(subset=["district"])
        .drop_duplicates("segment_id")
        .set_index("segment_id")["district"]
    )
    segments["district"] = segments["segment_id"].map(district_map)
    missing = segments["district"].isna()
    if missing.any():
        nearest_district = gpd.sjoin_nearest(
            center_gdf.loc[missing],
            polygons[["district", "geometry"]],
            how="left",
        )
        segments.loc[missing, "district"] = nearest_district[
            "district"
        ].to_numpy()

    academy_counts, parking_counts = _academy_and_parking_counts(Path(data_dir))
    segments["academy_density"] = segments["district"].map(academy_counts).fillna(0)
    segments["illegal_parking"] = segments["district"].map(parking_counts).fillna(0)

    segments["has_crosswalk"] = (segments["crosswalk_count"] > 0).astype(int)
    segments["has_signal"] = (segments["signal_count"] > 0).astype(int)
    segments["has_speed_bump"] = (segments["speed_bump_count"] > 0).astype(int)
    segments["has_cctv"] = (
        segments["school_zone_cctv_count"] > 0
    ).astype(int)
    segments["traffic_volume_norm"] = segments["traffic_volume_proxy"].clip(0, 1)
    segments["avg_speed_norm"] = (segments["avg_speed_kph"] / 100.0).clip(0, 1)
    segments["narrow_sidewalk_norm"] = segments[
        "narrow_sidewalk_proxy"
    ].clip(0, 1)
    segments["slope_norm"] = segments["slope_proxy"].clip(0, 1)
    segments["academy_density_norm"] = _minmax(
        segments["academy_density"], log_scale=True
    )
    segments["illegal_parking_norm"] = _minmax(
        segments["illegal_parking"], log_scale=True
    )
    segments["bus_stop_nearby_norm"] = _minmax(
        segments["bus_stop_count"], log_scale=True
    )
    segments["lane_count_norm"] = (
        segments["lane_count"].clip(lower=1, upper=6) / 6.0
    )
    segments["crosswalk_count_norm"] = _minmax(
        segments["crosswalk_count"], log_scale=True
    )
    segments["signal_count_norm"] = _minmax(
        segments["signal_count"], log_scale=True
    )
    segments["speed_bump_count_norm"] = _minmax(
        segments["speed_bump_count"], log_scale=True
    )
    segments["accident_count_norm"] = _minmax(
        segments["accident_count"], log_scale=True
    )
    streetlight_signal = _minmax(
        segments["streetlight_count"], log_scale=True
    )
    segments["light_density_norm"] = np.maximum(
        segments["lit_tag_score"], streetlight_signal
    )
    segments["pedestrian_flow_norm"] = (
        0.45 * segments["academy_density_norm"]
        + 0.35 * segments["bus_stop_nearby_norm"]
        + 0.20 * segments["is_school_zone"]
    ).clip(0, 1)

    score_features = segments.drop(columns="geometry")
    manual_calculator = CMCSCalculator()
    learned_weights, weight_report = derive_data_driven_cmcs_weights(
        score_features
    )
    scored = CMCSCalculator(learned_weights).score(score_features)
    scored["cmcs_manual_legacy"] = manual_calculator.calculate_cmcs(
        score_features
    )
    scored["cmcs_weight_evidence_methods"] = len(
        weight_report["normalized_method_shares"]
    )
    geometry_lookup = segments.set_index("segment_id")["geometry"]
    scored["geometry_wkt"] = scored["segment_id"].map(
        geometry_lookup.map(lambda geometry: geometry.wkt)
    )

    directed_columns = directed[
        ["edge_id", "segment_id", "u", "v", "key", "length_m"]
    ]
    full = directed_columns.merge(
        scored.drop(
            columns=["edge_id", "u", "v", "key", "length_m"],
            errors="ignore",
        ),
        on="segment_id",
        how="left",
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(path, index=False, encoding="utf-8-sig")
    return full, graph


def _optimal_f1_threshold(
    target: pd.Series | np.ndarray, probability: np.ndarray
) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if len(thresholds) == 0:
        return 0.5
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds, dtype=float),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def _threshold_metrics(
    target: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    prediction = probability >= threshold
    return {
        "balanced_accuracy": round(
            float(balanced_accuracy_score(target, prediction)), 6
        ),
        "precision": round(
            float(precision_score(target, prediction, zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(target, prediction, zero_division=0)), 6
        ),
        "f1": round(
            float(f1_score(target, prediction, zero_division=0)), 6
        ),
        "confusion_matrix": confusion_matrix(target, prediction).tolist(),
    }


def _edge_metrics(
    target: pd.Series, probability: np.ndarray
) -> dict[str, object]:
    optimal_threshold = _optimal_f1_threshold(target, probability)
    return {
        "roc_auc": round(float(roc_auc_score(target, probability)), 6),
        "average_precision": round(
            float(average_precision_score(target, probability)), 6
        ),
        "brier_score": round(
            float(brier_score_loss(target, probability)), 6
        ),
        "fixed_threshold": {
            "threshold": 0.5,
            **_threshold_metrics(target, probability, 0.5),
        },
        "optimized_threshold": {
            "threshold": round(optimal_threshold, 6),
            **_threshold_metrics(target, probability, optimal_threshold),
        },
    }


def _build_edge_model_candidates(
    positive_count: int,
    negative_count: int,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "도로 위험 모델 비교에는 xgboost가 필요합니다. "
            "pip install -e '.[ml]' 또는 pip install xgboost를 실행하세요."
        ) from exc

    scale_pos_weight = negative_count / max(positive_count, 1)
    return {
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
                        random_state=random_state,
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
                        n_estimators=350,
                        max_depth=12,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.1,
                        reg_lambda=2.0,
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _select_best_tree_model(
    metrics: dict[str, dict[str, object]],
    ap_tolerance: float = 0.005,
) -> str:
    candidates = ("RandomForest", "XGBoost")
    best_average_precision = max(
        float(metrics[name]["average_precision"]) for name in candidates
    )
    statistically_close = [
        name
        for name in candidates
        if best_average_precision
        - float(metrics[name]["average_precision"])
        <= ap_tolerance
    ]
    return max(
        statistically_close,
        key=lambda name: (
            float(metrics[name]["roc_auc"]),
            -float(metrics[name]["brier_score"]),
            float(metrics[name]["optimized_threshold"]["f1"]),
        ),
    )


def _serializable_model_parameters(
    models: dict[str, Pipeline],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, pipeline in models.items():
        parameters = pipeline.named_steps["model"].get_params()
        serialized: dict[str, object] = {}
        for key, value in parameters.items():
            if value is None or isinstance(value, (str, int, bool)):
                serialized[key] = value
            elif isinstance(value, float):
                serialized[key] = value if math.isfinite(value) else None
        result[name] = serialized
    return result


def train_edge_risk_model(
    edge_features: pd.DataFrame,
    model_path: str | Path = EDGE_MODEL_PATH,
    report_path: str | Path = EDGE_MODEL_REPORT_PATH,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    segments = edge_features.drop_duplicates("segment_id").copy()
    segments["accident_label"] = (segments["accident_count"] > 0).astype(int)
    positive = segments[segments["accident_label"].eq(1)]
    negative = segments[segments["accident_label"].eq(0)]
    negative_limit = min(len(negative), max(25_000, len(positive) * 6))
    negative_sample = negative.sample(
        negative_limit, random_state=random_state
    )
    training = pd.concat([positive, negative_sample]).sample(
        frac=1, random_state=random_state
    )

    X = training[EDGE_MODEL_FEATURES]
    y = training["accident_label"]
    grid_x = np.floor(training["center_x"] / 2000).astype(int)
    grid_y = np.floor(training["center_y"] / 2000).astype(int)
    groups = grid_x.astype(str) + ":" + grid_y.astype(str)
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )

    models = _build_edge_model_candidates(
        positive_count=int(y.sum()),
        negative_count=int((y == 0).sum()),
        random_state=random_state,
    )

    metrics: dict[str, dict[str, object]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        probability = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        probabilities[name] = probability
        metrics[name] = _edge_metrics(y, probability)

    selection_candidates = ("RandomForest", "XGBoost")
    best_name = _select_best_tree_model(metrics)
    best_model = models[best_name]
    best_model.fit(X, y)
    all_probability = best_model.predict_proba(segments[EDGE_MODEL_FEATURES])[:, 1]
    segments["ml_risk_probability"] = all_probability

    best_metrics = metrics[best_name]
    selected_threshold = float(
        best_metrics["optimized_threshold"]["threshold"]
    )
    if best_metrics["roc_auc"] >= 0.70:
        blend_weight = 0.35
    elif best_metrics["roc_auc"] >= 0.60:
        blend_weight = 0.20
    else:
        blend_weight = 0.0
    segments["cmcs_final"] = (
        (1.0 - blend_weight) * segments["cmcs"]
        + blend_weight * segments["ml_risk_probability"]
    ).clip(0, 1)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": EDGE_MODEL_FEATURES,
            "blend_weight": blend_weight,
            "decision_threshold": selected_threshold,
            "best_model_name": best_name,
            "validation_metrics": best_metrics,
            "label": "accident hotspot within 300m of road segment",
        },
        model_path,
    )
    validation_predictions = training[
        ["segment_id", "district", "accident_label"]
    ].copy()
    for name, probability in probabilities.items():
        validation_predictions[f"{name}_probability"] = probability
        threshold = float(metrics[name]["optimized_threshold"]["threshold"])
        validation_predictions[f"{name}_prediction"] = (
            probability >= threshold
        ).astype(int)
    validation_prediction_path = (
        REPORT_OUTPUT_DIR / "edge_model_validation_predictions.csv"
    )
    validation_predictions.to_csv(
        validation_prediction_path, index=False, encoding="utf-8-sig"
    )

    roc_path = CHART_OUTPUT_DIR / "edge_model_roc_pr.png"
    _plot_edge_model_validation(y, probabilities, roc_path)
    explain_path = CHART_OUTPUT_DIR / "edge_model_explainability.png"
    explanation_method = _explain_edge_model(
        best_model,
        training[EDGE_MODEL_FEATURES],
        explain_path,
    )
    prevalence = float(training["accident_label"].mean())
    ap_lift = (
        float(best_metrics["average_precision"]) / prevalence
        if prevalence > 0
        else 0.0
    )
    best_overall_name = max(
        metrics,
        key=lambda name: (
            metrics[name]["average_precision"],
            metrics[name]["roc_auc"],
        ),
    )
    report = {
        "segment_count": int(len(segments)),
        "training_count": int(len(training)),
        "positive_segment_count": int(segments["accident_label"].sum()),
        "positive_rate_full": round(float(segments["accident_label"].mean()), 6),
        "validation": "5-fold StratifiedGroupKFold with 2km spatial grid groups",
        "features": EDGE_MODEL_FEATURES,
        "models": metrics,
        "model_parameters": _serializable_model_parameters(models),
        "best_model": best_name,
        "best_overall_validation_model": best_overall_name,
        "baseline_model": "LogisticRegression",
        "baseline_outperformed_selected_tree": (
            best_overall_name == "LogisticRegression"
        ),
        "selection_candidates": list(selection_candidates),
        "selection_metric": (
            "average_precision primary; AP difference <=0.005 is treated as "
            "a tie, then roc_auc, lower brier_score, and optimized_f1"
        ),
        "selected_decision_threshold": round(selected_threshold, 6),
        "cmcs_ml_blend_weight": blend_weight,
        "average_precision_lift_over_training_prevalence": round(ap_lift, 4),
        "research_validation_passed": (
            best_metrics["roc_auc"] >= 0.70 and ap_lift >= 2.0
        ),
        "production_deployment_ready": False,
        "explanation_method": explanation_method,
        "artifacts": {
            "validation_predictions": str(validation_prediction_path),
            "roc_pr_chart": str(roc_path),
            "explainability_chart": str(explain_path),
        },
        "limitations": [
            "사고 라벨은 개별 사고 원장이 아니라 어린이보호구역 사고 다발지역 중심 반경 300m이다.",
            "사고와 시설 데이터의 기준연도가 완전히 일치하지 않는다.",
            "교통량은 OSM 도로 등급 기반 프록시이며 실측 교통량이 아니다.",
            "학원 밀도와 불법주정차는 구 단위 집계라 도로별 국지 변동을 충분히 표현하지 못한다.",
        ],
    }
    leaderboard_rows = []
    for name, model_metrics in metrics.items():
        optimized = model_metrics["optimized_threshold"]
        fixed = model_metrics["fixed_threshold"]
        leaderboard_rows.append(
            {
                "model": name,
                "roc_auc": model_metrics["roc_auc"],
                "average_precision": model_metrics["average_precision"],
                "brier_score": model_metrics["brier_score"],
                "optimized_threshold": optimized["threshold"],
                "optimized_f1": optimized["f1"],
                "optimized_precision": optimized["precision"],
                "optimized_recall": optimized["recall"],
                "optimized_balanced_accuracy": optimized[
                    "balanced_accuracy"
                ],
                "fixed_f1": fixed["f1"],
                "fixed_precision": fixed["precision"],
                "fixed_recall": fixed["recall"],
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    )
    leaderboard_path = REPORT_OUTPUT_DIR / "edge_model_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False, encoding="utf-8-sig")
    report["artifacts"]["leaderboard"] = str(leaderboard_path)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return segments, report


def _aggregate_regional_features(
    segment_scores: pd.DataFrame,
    cell_size_m: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = segment_scores.drop_duplicates("segment_id").copy()
    segments = segments.drop(
        columns=[
            "region_x",
            "region_y",
            "regional_risk_probability",
            "regional_risk_prediction",
            "oof_risk_probability",
            "oof_probability_std",
            "nested_oof_probability",
            "nested_oof_prediction",
            "nested_decision_threshold",
        ],
        errors="ignore",
    )
    segments["district"] = segments["district"].fillna("미분류").astype(str)
    segments["highway"] = segments.get(
        "highway", pd.Series("other", index=segments.index)
    ).fillna("other").astype(str)
    segments["highway_group"] = segments["highway"].where(
        segments["highway"].isin(REGIONAL_HIGHWAY_CLASSES[:-1]),
        "other",
    )
    for feature in set(
        EDGE_MODEL_FEATURES
        + REGIONAL_DISTRIBUTION_FEATURES
        + REGIONAL_SUM_FEATURES
    ):
        if feature not in segments:
            segments[feature] = 1.0 if feature == "length_m" else 0.0
        segments[feature] = pd.to_numeric(
            segments[feature], errors="coerce"
        ).fillna(0.0)
    segments["region_x"] = (
        pd.to_numeric(segments["center_x"], errors="coerce").fillna(0)
        // cell_size_m
    ).astype(int)
    segments["region_y"] = (
        pd.to_numeric(segments["center_y"], errors="coerce").fillna(0)
        // cell_size_m
    ).astype(int)
    segments["accident_label"] = (
        pd.to_numeric(segments["accident_count"], errors="coerce").fillna(0) > 0
    ).astype(int)

    key_columns = ["district", "region_x", "region_y"]
    aggregations: dict[str, object] = {
        feature: "mean" for feature in EDGE_MODEL_FEATURES
    }
    aggregations.update(
        {
            "center_x": "mean",
            "center_y": "mean",
            "segment_id": "size",
            "accident_label": "max",
        }
    )
    regions = (
        segments.groupby(
            key_columns,
            as_index=False,
        )
        .agg(aggregations)
        .rename(columns={"segment_id": "segment_count"})
    )

    grouped = segments.groupby(key_columns)
    for statistic in ("max", "std"):
        values = getattr(
            grouped[REGIONAL_DISTRIBUTION_FEATURES],
            statistic,
        )()
        if statistic == "std":
            values = values.fillna(0.0)
        values = values.add_suffix(f"_{statistic}").reset_index()
        regions = regions.merge(values, on=key_columns, how="left")
    quantiles = (
        grouped[REGIONAL_DISTRIBUTION_FEATURES]
        .quantile(0.90)
        .add_suffix("_q90")
        .reset_index()
    )
    regions = regions.merge(quantiles, on=key_columns, how="left")
    sums = (
        grouped[REGIONAL_SUM_FEATURES]
        .sum()
        .add_suffix("_sum")
        .reset_index()
    )
    regions = regions.merge(sums, on=key_columns, how="left")

    highway_shares = pd.crosstab(
        [segments[column] for column in key_columns],
        segments["highway_group"],
        normalize="index",
    ).reset_index()
    for highway in REGIONAL_HIGHWAY_CLASSES:
        if highway not in highway_shares:
            highway_shares[highway] = 0.0
    highway_shares = highway_shares.rename(
        columns={
            highway: f"hw_share_{highway}"
            for highway in REGIONAL_HIGHWAY_CLASSES
        }
    )
    regions = regions.merge(
        highway_shares[
            key_columns
            + [
                f"hw_share_{highway}"
                for highway in REGIONAL_HIGHWAY_CLASSES
            ]
        ],
        on=key_columns,
        how="left",
    )

    regions["log_segment_count"] = np.log1p(regions["segment_count"])
    regions["road_length_km"] = regions["length_m_sum"] / 1000.0
    road_length = regions["road_length_km"].clip(lower=0.05)
    for source, output in (
        ("crosswalk_count_sum", "crosswalk_per_km"),
        ("signal_count_sum", "signal_per_km"),
        ("speed_bump_count_sum", "speed_bump_per_km"),
        ("bus_stop_count_sum", "bus_stop_per_km"),
        ("streetlight_count_sum", "streetlight_per_km"),
        ("school_zone_cctv_count_sum", "school_zone_cctv_per_km"),
    ):
        regions[output] = regions[source] / road_length
    segment_count = regions["segment_count"].clip(lower=1)
    for source, output in (
        ("is_school_zone_sum", "school_zone_share"),
        ("has_crosswalk_sum", "crosswalk_coverage"),
        ("has_signal_sum", "signal_coverage"),
        ("has_speed_bump_sum", "speed_bump_coverage"),
        ("has_cctv_sum", "cctv_coverage"),
    ):
        regions[output] = regions[source] / segment_count

    # 사고 라벨 또는 사고 건수에서 파생된 CMCS를 넣지 않는 비누수 구조 위험도다.
    regions["structural_hazard_mean"] = (
        0.18 * regions["traffic_volume_norm"]
        + 0.12 * regions["avg_speed_norm"]
        + 0.15 * regions["narrow_sidewalk_norm"]
        + 0.08 * regions["slope_norm"]
        + 0.12 * regions["pedestrian_flow_norm"]
        + 0.08 * regions["illegal_parking_norm"]
        + 0.10 * (1.0 - regions["light_density_norm"])
        + 0.08 * (1.0 - regions["has_crosswalk"])
        + 0.06 * (1.0 - regions["has_signal"])
        + 0.03 * regions["lane_count_norm"]
    )
    regions["structural_hazard_peak"] = (
        0.25 * regions["traffic_volume_norm_max"]
        + 0.15 * regions["avg_speed_norm_max"]
        + 0.20 * regions["narrow_sidewalk_norm_max"]
        + 0.10 * regions["pedestrian_flow_norm_max"]
        + 0.10 * regions["illegal_parking_norm_max"]
        + 0.10 * (1.0 - regions["light_density_norm"])
        + 0.10 * regions["lane_count_norm_max"]
    )
    regions[REGIONAL_MODEL_FEATURES] = regions[
        REGIONAL_MODEL_FEATURES
    ].replace([np.inf, -np.inf], np.nan)
    return segments, regions


def _regional_xgb_pipeline(
    positive_count: int,
    negative_count: int,
    random_state: int,
) -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=2,
                    learning_rate=0.035,
                    min_child_weight=2,
                    subsample=0.90,
                    colsample_bytree=0.85,
                    reg_alpha=0.10,
                    reg_lambda=5.0,
                    scale_pos_weight=negative_count / max(positive_count, 1),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _metrics_from_predictions(
    target: pd.Series,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    probability_metrics: dict[str, object]
    if target.nunique() >= 2:
        probability_metrics = {
            "roc_auc": round(float(roc_auc_score(target, probability)), 6),
            "average_precision": round(
                float(average_precision_score(target, probability)), 6
            ),
        }
    else:
        probability_metrics = {
            "roc_auc": None,
            "average_precision": None,
        }
    return {
        **probability_metrics,
        "brier_score": round(
            float(brier_score_loss(target, probability)), 6
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(target, prediction)), 6
        ),
        "precision": round(
            float(precision_score(target, prediction, zero_division=0)), 6
        ),
        "recall": round(
            float(recall_score(target, prediction, zero_division=0)), 6
        ),
        "f1": round(
            float(f1_score(target, prediction, zero_division=0)), 6
        ),
        "confusion_matrix": confusion_matrix(target, prediction).tolist(),
    }


def _nested_regional_validation(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    probability = np.zeros(len(target), dtype=float)
    prediction = np.zeros(len(target), dtype=int)
    row_threshold = np.zeros(len(target), dtype=float)
    fold_reports: list[dict[str, object]] = []
    outer_splitter = LeaveOneGroupOut()

    for train_index, test_index in outer_splitter.split(
        features, target, groups
    ):
        train_features = features.iloc[train_index]
        train_target = target.iloc[train_index]
        train_groups = groups.iloc[train_index]
        positive_count = int(train_target.sum())
        negative_count = int((train_target == 0).sum())

        inner_probabilities = []
        for seed in REGIONAL_NESTED_SEEDS:
            inner_model = _regional_xgb_pipeline(
                positive_count,
                negative_count,
                seed,
            )
            inner_probabilities.append(
                cross_val_predict(
                    inner_model,
                    train_features,
                    train_target,
                    groups=train_groups,
                    cv=LeaveOneGroupOut(),
                    method="predict_proba",
                    n_jobs=1,
                )[:, 1]
            )
        inner_probability = np.mean(inner_probabilities, axis=0)
        threshold = float(
            np.clip(
                _optimal_f1_threshold(train_target, inner_probability),
                0.20,
                0.70,
            )
        )

        test_probabilities = []
        for seed in REGIONAL_NESTED_SEEDS:
            model = _regional_xgb_pipeline(
                positive_count,
                negative_count,
                seed,
            )
            model.fit(train_features, train_target)
            test_probabilities.append(
                model.predict_proba(features.iloc[test_index])[:, 1]
            )
        fold_probability = np.mean(test_probabilities, axis=0)
        fold_prediction = (fold_probability >= threshold).astype(int)
        probability[test_index] = fold_probability
        prediction[test_index] = fold_prediction
        row_threshold[test_index] = threshold
        held_out_district = str(groups.iloc[test_index[0]])
        fold_metrics = _metrics_from_predictions(
            target.iloc[test_index],
            fold_probability,
            fold_prediction,
        )
        fold_reports.append(
            {
                "held_out_district": held_out_district,
                "test_count": int(len(test_index)),
                "positive_count": int(target.iloc[test_index].sum()),
                "threshold": round(threshold, 6),
                **fold_metrics,
            }
        )
    return probability, prediction, row_threshold, fold_reports


def train_regional_boosting_model(
    segment_scores: pd.DataFrame,
    model_path: str | Path = REGIONAL_MODEL_PATH,
    report_path: str | Path = REGIONAL_MODEL_REPORT_PATH,
    cell_size_m: int = REGIONAL_CELL_SIZE_M,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "권역 부스팅 모델에는 xgboost가 필요합니다."
        ) from exc

    segments, regions = _aggregate_regional_features(
        segment_scores,
        cell_size_m,
    )
    X = regions[REGIONAL_MODEL_FEATURES]
    y = regions["accident_label"].astype(int)
    groups = regions["district"].astype(str)
    if y.nunique() < 2 or groups.nunique() < 2:
        raise ValueError("권역 부스팅 학습에 필요한 클래스 또는 자치구가 부족합니다.")

    positive_count = int(y.sum())
    negative_count = int((y == 0).sum())
    seed_probabilities: dict[int, np.ndarray] = {}
    for seed in REGIONAL_ENSEMBLE_SEEDS:
        seed_model = _regional_xgb_pipeline(
            positive_count,
            negative_count,
            seed,
        )
        seed_probabilities[seed] = cross_val_predict(
            seed_model,
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    oof_probability = np.mean(list(seed_probabilities.values()), axis=0)
    metrics = _edge_metrics(y, oof_probability)
    decision_threshold = float(
        metrics["optimized_threshold"]["threshold"]
    )

    nested_probability, nested_prediction, nested_thresholds, fold_reports = (
        _nested_regional_validation(X, y, groups)
    )
    nested_metrics = _metrics_from_predictions(
        y,
        nested_probability,
        nested_prediction,
    )

    models = []
    fitted_probabilities = []
    seed_optimized_f1: dict[str, float] = {}
    seed_fixed_threshold_f1: dict[str, float] = {}
    for seed, seed_probability in seed_probabilities.items():
        seed_metrics = _edge_metrics(y, seed_probability)
        seed_optimized_f1[str(seed)] = float(
            seed_metrics["optimized_threshold"]["f1"]
        )
        seed_fixed_threshold_f1[str(seed)] = float(
            f1_score(
                y,
                seed_probability >= decision_threshold,
                zero_division=0,
            )
        )
        model = _regional_xgb_pipeline(
            positive_count,
            negative_count,
            seed,
        )
        model.fit(X, y)
        models.append(model)
        fitted_probabilities.append(model.predict_proba(X)[:, 1])
    fitted_probability = np.mean(fitted_probabilities, axis=0)
    validation_columns = pd.DataFrame(
        {
            "regional_risk_probability": fitted_probability,
            "regional_risk_prediction": (
                fitted_probability >= decision_threshold
            ).astype(int),
            "oof_risk_probability": oof_probability,
            "oof_probability_std": np.std(
                list(seed_probabilities.values()),
                axis=0,
            ),
            "nested_oof_probability": nested_probability,
            "nested_oof_prediction": nested_prediction,
            "nested_decision_threshold": nested_thresholds,
        },
        index=regions.index,
    )
    regions = pd.concat([regions.copy(), validation_columns], axis=1)

    key_columns = ["district", "region_x", "region_y"]
    segments = segments.merge(
        regions[
            key_columns
            + ["regional_risk_probability", "regional_risk_prediction"]
        ],
        on=key_columns,
        how="left",
    )
    f1_target_achieved = (
        float(nested_metrics["f1"]) >= 0.5
        and min(seed_fixed_threshold_f1.values()) >= 0.5
    )
    regional_blend_weight = 0.25 if f1_target_achieved else 0.0
    segments["cmcs_final"] = (
        (1.0 - regional_blend_weight) * segments["cmcs_final"]
        + regional_blend_weight * segments["regional_risk_probability"]
    ).clip(0, 1)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_columns": REGIONAL_MODEL_FEATURES,
            "cell_size_m": cell_size_m,
            "decision_threshold": decision_threshold,
            "regional_blend_weight": regional_blend_weight,
            "validation_metrics": metrics,
            "nested_validation_metrics": nested_metrics,
            "ensemble_seeds": list(REGIONAL_ENSEMBLE_SEEDS),
            "validation": (
                "Nested LeaveOneGroupOut by Daejeon district with "
                "fold-specific train-only threshold"
            ),
        },
        model_path,
    )

    regional_predictions_path = (
        REPORT_OUTPUT_DIR / "regional_boosting_predictions.csv"
    )
    regions.to_csv(
        regional_predictions_path, index=False, encoding="utf-8-sig"
    )
    chart_path = CHART_OUTPUT_DIR / "regional_boosting_roc_pr.png"
    _plot_regional_boosting_validation(y, oof_probability, chart_path)
    explain_path = CHART_OUTPUT_DIR / "regional_boosting_shap.png"
    explanation_method = _explain_model_with_features(
        models[0],
        X,
        REGIONAL_MODEL_FEATURES,
        explain_path,
    )
    threshold_candidates = np.linspace(0.05, 0.95, 181)
    stable_thresholds = [
        float(threshold)
        for threshold in threshold_candidates
        if f1_score(y, oof_probability >= threshold, zero_division=0) >= 0.5
    ]
    report = {
        "model": "XGBoost seed ensemble",
        "task": (
            f"{cell_size_m}m 통학 위험 권역 내 사고 다발지역 존재 여부"
        ),
        "region_count": int(len(regions)),
        "positive_region_count": positive_count,
        "positive_rate": round(float(y.mean()), 6),
        "validation": (
            "Nested LeaveOneGroupOut by district; outer district is never "
            "used for model fitting or threshold selection"
        ),
        "region_definition": (
            f"{cell_size_m}m grid clipped by district boundary"
        ),
        "held_out_groups": sorted(groups.unique().tolist()),
        "metrics": metrics,
        "nested_validation_metrics": nested_metrics,
        "per_district_nested_metrics": fold_reports,
        "stability": {
            "ensemble_seeds": list(REGIONAL_ENSEMBLE_SEEDS),
            "seed_f1_at_shared_threshold": {
                seed: round(value, 6)
                for seed, value in seed_fixed_threshold_f1.items()
            },
            "seed_optimized_f1": {
                seed: round(value, 6)
                for seed, value in seed_optimized_f1.items()
            },
            "minimum_seed_f1": round(
                min(seed_fixed_threshold_f1.values()),
                6,
            ),
            "maximum_seed_f1": round(
                max(seed_fixed_threshold_f1.values()),
                6,
            ),
            "seed_f1_standard_deviation": round(
                float(np.std(list(seed_fixed_threshold_f1.values()))),
                6,
            ),
            "f1_at_least_0_5_threshold_range": [
                round(min(stable_thresholds), 4),
                round(max(stable_thresholds), 4),
            ]
            if stable_thresholds
            else None,
            "nested_threshold_range": [
                round(float(nested_thresholds.min()), 6),
                round(float(nested_thresholds.max()), 6),
            ],
        },
        "f1_target": 0.5,
        "f1_target_achieved": f1_target_achieved,
        "decision_threshold": round(decision_threshold, 6),
        "regional_blend_weight": regional_blend_weight,
        "explanation_method": explanation_method,
        "artifacts": {
            "model": str(model_path),
            "predictions": str(regional_predictions_path),
            "roc_pr_chart": str(chart_path),
            "shap_chart": str(explain_path),
        },
        "scope_note": (
            f"F1은 개별 도로가 아니라 {cell_size_m / 1000:g}km 통학 위험 "
            "권역 분류 기준이다. "
            "도로별 위험은 이 확률을 AHP·도로 모델과 결합해 산출한다."
        ),
        "leakage_control": (
            "사고 건수, 사상자 수, 사고 라벨 및 사고 건수에서 파생된 CMCS "
            "구성값은 입력 피처에서 제외했다."
        ),
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return segments, report


def _plot_regional_boosting_validation(
    target: pd.Series,
    probability: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    false_positive_rate, true_positive_rate, _ = roc_curve(
        target, probability
    )
    precision, recall, _ = precision_recall_curve(target, probability)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        false_positive_rate,
        true_positive_rate,
        color="#2563eb",
        label=f"XGBoost AUC={roc_auc_score(target, probability):.3f}",
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[0].set_title("District holdout ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()
    axes[1].plot(
        recall,
        precision,
        color="#16a34a",
        label=(
            f"XGBoost AP="
            f"{average_precision_score(target, probability):.3f}"
        ),
    )
    axes[1].axhline(
        float(target.mean()), color="#6b7280", linestyle="--", label="Prevalence"
    )
    axes[1].set_title("District holdout Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_edge_model_validation(
    target: pd.Series,
    probabilities: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {
        "LogisticRegression": "#6b7280",
        "RandomForest": "#16a34a",
        "XGBoost": "#2563eb",
    }
    for model_name, probability in probabilities.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            target, probability
        )
        precision, recall, _ = precision_recall_curve(target, probability)
        roc_auc = roc_auc_score(target, probability)
        average_precision = average_precision_score(target, probability)
        color = colors.get(model_name)
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            label=f"{model_name} (AUC={roc_auc:.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            label=f"{model_name} (AP={average_precision:.3f})",
        )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.35)
    axes[0].set_title("Spatial CV ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    axes[1].axhline(
        float(pd.Series(target).mean()),
        color="#6b7280",
        linestyle="--",
        label="Prevalence",
    )
    axes[1].set_title("Spatial CV Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _explain_edge_model(
    model: Pipeline,
    features: pd.DataFrame,
    output_path: Path,
) -> str:
    return _explain_model_with_features(
        model,
        features,
        EDGE_MODEL_FEATURES,
        output_path,
    )


def _explain_model_with_features(
    model: Pipeline,
    features: pd.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> str:
    import matplotlib.pyplot as plt

    estimator = model.named_steps["model"]
    sample = features.sample(min(2500, len(features)), random_state=42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import shap

        transformed = model.named_steps["imputer"].transform(sample)
        if "scaler" in model.named_steps:
            transformed = model.named_steps["scaler"].transform(transformed)
        transformed_frame = pd.DataFrame(
            transformed, columns=feature_names
        )
        if hasattr(estimator, "feature_importances_"):
            values = shap.TreeExplainer(estimator)(transformed_frame)
            if values.values.ndim == 3:
                values = values[:, :, 1]
        else:
            values = shap.LinearExplainer(estimator, transformed_frame)(
                transformed_frame
            )
        shap.plots.bar(values, max_display=14, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return "SHAP"
    except Exception:
        plt.close("all")
        if hasattr(estimator, "coef_"):
            importance = pd.Series(
                np.abs(estimator.coef_[0]), index=feature_names
            ).sort_values().tail(14)
            figure, axis = plt.subplots(figsize=(9, 6))
            importance.plot.barh(ax=axis, color="#2563eb")
            axis.set_title("Absolute standardized logistic coefficients")
            figure.tight_layout()
            figure.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            return "absolute logistic coefficients (SHAP fallback)"
        if hasattr(estimator, "feature_importances_"):
            importance = pd.Series(
                estimator.feature_importances_, index=feature_names
            ).sort_values().tail(14)
            figure, axis = plt.subplots(figsize=(9, 6))
            importance.plot.barh(ax=axis, color="#2563eb")
            axis.set_title("Tree model feature importance")
            figure.tight_layout()
            figure.savefig(output_path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            return "tree feature importance (SHAP fallback)"
        if output_path.exists():
            output_path.unlink()
        return "not available"


def apply_scores_to_graph(
    graph: nx.MultiDiGraph,
    segment_scores: pd.DataFrame,
    output_path: str | Path = SCORED_GRAPH_PATH,
) -> nx.MultiDiGraph:
    score_map = segment_scores.set_index("segment_id")[
        [
            "cmcs",
            "cmcs_final",
            "ml_risk_probability",
            "regional_risk_probability",
        ]
    ].to_dict(orient="index")
    for _, _, _, data in graph.edges(keys=True, data=True):
        scores = score_map.get(str(data["segment_id"]), {})
        data["cmcs_ahp"] = float(scores.get("cmcs", 0.5))
        data["cmcs"] = float(scores.get("cmcs_final", data["cmcs_ahp"]))
        data["ml_risk_probability"] = float(
            scores.get("ml_risk_probability", 0.5)
        )
        data["regional_risk_probability"] = float(
            scores.get("regional_risk_probability", 0.5)
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=path)
    return graph


def create_city_risk_map(
    segment_scores: pd.DataFrame,
    output_path: str | Path = MAP_OUTPUT_DIR / "daejeon_cmcs_risk_map.html",
    max_segments: int = 5000,
) -> Path:
    import folium

    unique = segment_scores.drop_duplicates("segment_id").copy()
    unique = unique.nlargest(min(max_segments, len(unique)), "cmcs_final")
    geometry = gpd.GeoSeries.from_wkt(unique["geometry_wkt"], crs=PROJECTED_CRS)
    risk_gdf = gpd.GeoDataFrame(
        unique[
            [
                "segment_id",
                "district",
                "highway",
                "cmcs",
                "cmcs_final",
                "ml_risk_probability",
                "regional_risk_probability",
            ]
        ].copy(),
        geometry=geometry,
        crs=PROJECTED_CRS,
    ).to_crs("EPSG:4326")
    risk_gdf["cmcs_final"] = risk_gdf["cmcs_final"].round(4)
    risk_gdf["ml_risk_probability"] = risk_gdf[
        "ml_risk_probability"
    ].round(4)
    risk_gdf["regional_risk_probability"] = risk_gdf[
        "regional_risk_probability"
    ].round(4)
    map_object = folium.Map(
        location=[36.3504, 127.3845],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    def style_function(feature):
        value = float(feature["properties"]["cmcs_final"])
        if value >= 0.65:
            color = "#b91c1c"
        elif value >= 0.55:
            color = "#ef4444"
        elif value >= 0.45:
            color = "#f59e0b"
        else:
            color = "#84cc16"
        return {"color": color, "weight": 3, "opacity": 0.72}

    folium.GeoJson(
        json.loads(risk_gdf.to_json()),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "district",
                "highway",
                "cmcs_final",
                "ml_risk_probability",
                "regional_risk_probability",
            ],
            aliases=[
                "자치구",
                "도로 유형",
                "최종 CMCS",
                "도로 ML 위험 확률",
                "권역 XGBoost 위험 확률",
            ],
        ),
        name="상위 위험 도로",
    ).add_to(map_object)
    legend = """
    <div style="position:fixed;bottom:25px;left:25px;z-index:9999;
    background:white;padding:10px 12px;border:1px solid #bbb;border-radius:6px">
    <b>CMCS 상위 위험 도로</b><br>
    <span style="color:#b91c1c">━</span> 0.65 이상<br>
    <span style="color:#ef4444">━</span> 0.55~0.65<br>
    <span style="color:#f59e0b">━</span> 0.45~0.55<br>
    <span style="color:#84cc16">━</span> 0.45 미만
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(map_object)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(path)
    return path


def create_pareto_chart(
    pareto: pd.DataFrame,
    output_path: str | Path = CHART_OUTPUT_DIR / "actual_route_pareto.html",
) -> Path:
    import plotly.express as px

    figure = px.line(
        pareto.sort_values("lambda"),
        x="distance_m",
        y="cmcs",
        color="lambda",
        markers=True,
        title="Actual route Pareto frontier: distance vs risk exposure",
        labels={
            "distance_m": "Walking distance (m)",
            "cmcs": "CMCS risk exposure",
            "lambda": "Distance weight λ",
        },
    )
    figure.update_layout(template="plotly_white")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)
    return path


def create_district_safety_outputs(
    segment_scores: pd.DataFrame,
    csv_path: str | Path = REPORT_OUTPUT_DIR / "district_safety_summary.csv",
    chart_path: str | Path = CHART_OUTPUT_DIR / "district_safety_radar.html",
) -> tuple[pd.DataFrame, Path]:
    import plotly.graph_objects as go

    unique = segment_scores.drop_duplicates("segment_id").copy()
    unique["high_risk"] = (unique["cmcs_final"] >= 0.6).astype(int)
    unique["accident_labeled"] = (unique["accident_count"] > 0).astype(int)
    unique["infrastructure_coverage"] = (
        unique[["has_crosswalk", "has_signal", "has_speed_bump", "has_cctv"]]
        .mean(axis=1)
        .clip(0, 1)
    )
    unique["walking_comfort"] = (1 - unique["narrow_sidewalk_norm"]).clip(0, 1)
    summary = (
        unique.groupby("district")
        .agg(
            average_cmcs=("cmcs_final", "mean"),
            high_risk_ratio=("high_risk", "mean"),
            accident_labeled_ratio=("accident_labeled", "mean"),
            infrastructure_coverage=("infrastructure_coverage", "mean"),
            walking_comfort=("walking_comfort", "mean"),
            segment_count=("segment_id", "size"),
        )
        .reset_index()
    )
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    categories = [
        "average_cmcs",
        "high_risk_ratio",
        "accident_labeled_ratio",
        "infrastructure_coverage",
        "walking_comfort",
    ]
    figure = go.Figure()
    for _, row in summary.iterrows():
        figure.add_trace(
            go.Scatterpolar(
                r=[float(row[column]) for column in categories],
                theta=categories,
                fill="toself",
                name=row["district"],
            )
        )
    figure.update_layout(
        title="Daejeon district CMCS and walking-safety indicators",
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        template="plotly_white",
    )
    chart_path = Path(chart_path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(chart_path)
    return summary, chart_path


def _actual_route_endpoints(data_dir: Path) -> dict[str, object]:
    schools = read_csv_flexible(next(data_dir.glob("*초중등학교위치*.csv")))
    origin_row = schools[schools["학교명"].eq("대전문정초등학교")].iloc[0]
    academy_file = next(data_dir.glob("서부교육지원청+학원+및+교습소+현황*.xlsx"))
    academies = pd.read_excel(academy_file, sheet_name="학원")
    address_column = next(
        column for column in academies.columns if "학원주소" in str(column)
    )
    academy_row = academies[
        academies["학원명"].eq("둔산씨엠에스학원")
    ].iloc[0]
    # Nominatim이 한국어 상세 주소를 안정적으로 인식하지 못해 같은 도로명 주소를
    # 영문 질의로 지오코딩한 좌표를 재현 가능한 기본값으로 사용한다.
    destination_query = "136 Dunsan-ro, Seo-gu, Daejeon, South Korea"
    try:
        destination_latitude, destination_longitude = ox.geocode(
            destination_query
        )
    except Exception:
        destination_latitude, destination_longitude = (
            36.3513309,
            127.3807830,
        )
    return {
        "origin_name": str(origin_row["학교명"]),
        "origin_address": str(origin_row["소재지도로명주소"]),
        "origin": (float(origin_row["위도"]), float(origin_row["경도"])),
        "destination_name": str(academy_row["학원명"]),
        "destination_address": str(academy_row[address_column]),
        "destination": (
            float(destination_latitude),
            float(destination_longitude),
        ),
        "destination_geocode_query": destination_query,
    }


def _school_route_validation_pairs(
    data_dir: Path,
    pair_count: int = 25,
) -> list[tuple[tuple[float, float], tuple[float, float], str, str]]:
    schools = read_csv_flexible(next(data_dir.glob("*초중등학교위치*.csv")))
    schools = schools[
        schools["소재지도로명주소"].astype(str).str.contains("대전광역시")
        & schools["학교급구분"].eq("초등학교")
        & schools["운영상태"].eq("운영")
    ].copy()
    schools["위도"] = pd.to_numeric(schools["위도"], errors="coerce")
    schools["경도"] = pd.to_numeric(schools["경도"], errors="coerce")
    schools = schools.dropna(subset=["위도", "경도"]).drop_duplicates("학교ID")
    schools["district"] = schools["소재지도로명주소"].map(extract_district)
    distances = haversine_matrix(
        schools["위도"],
        schools["경도"],
        schools["위도"],
        schools["경도"],
    )
    index_lookup = {index: position for position, index in enumerate(schools.index)}
    pairs = []
    used: set[tuple[str, str]] = set()
    per_district = max(1, math.ceil(pair_count / max(len(DISTRICTS), 1)))
    for district in DISTRICTS:
        district_schools = schools[schools["district"].eq(district)].sort_values(
            "학교명"
        )
        if district_schools.empty:
            continue
        selected_positions = np.linspace(
            0,
            len(district_schools) - 1,
            min(per_district, len(district_schools)),
            dtype=int,
        )
        for position in selected_positions:
            origin = district_schools.iloc[int(position)]
            origin_position = index_lookup[origin.name]
            candidate_positions = np.where(
                (distances[origin_position] >= 700.0)
                & (distances[origin_position] <= 3000.0)
            )[0]
            if len(candidate_positions) == 0:
                continue
            destination_position = int(
                candidate_positions[
                    np.argmin(
                        np.abs(
                            distances[origin_position, candidate_positions]
                            - 1600.0
                        )
                    )
                ]
            )
            destination = schools.iloc[destination_position]
            pair_key = tuple(
                sorted((str(origin["학교ID"]), str(destination["학교ID"])))
            )
            if pair_key in used:
                continue
            used.add(pair_key)
            pairs.append(
                (
                    (float(origin["위도"]), float(origin["경도"])),
                    (
                        float(destination["위도"]),
                        float(destination["경도"]),
                    ),
                    str(origin["학교명"]),
                    str(destination["학교명"]),
                )
            )
            if len(pairs) >= pair_count:
                return pairs
    return pairs


def create_actual_route_map(
    graph: nx.MultiDiGraph,
    routes: list[dict[str, object]],
    endpoints: dict[str, object],
    output_path: str | Path = MAP_OUTPUT_DIR / "actual_safe_route.html",
) -> Path:
    import folium

    origin = endpoints["origin"]
    destination = endpoints["destination"]
    map_object = folium.Map(
        location=[
            (origin[0] + destination[0]) / 2,
            (origin[1] + destination[1]) / 2,
        ],
        zoom_start=15,
        tiles="CartoDB positron",
    )
    colors = {
        "최단거리": "#2563eb",
        "최저위험": "#16a34a",
        "균형": "#f59e0b",
    }
    for route in routes:
        coordinates = [
            (float(graph.nodes[node]["y"]), float(graph.nodes[node]["x"]))
            for node in route["path"]
        ]
        label = str(route["mode"])
        color = next(
            (
                value
                for prefix, value in colors.items()
                if label.startswith(prefix)
            ),
            "#6b7280",
        )
        popup = (
            f"<b>{label}</b><br>"
            f"거리: {route['total_distance_m']:.0f}m<br>"
            f"평균 CMCS: {route['average_cmcs']:.3f}<br>"
            f"위험 노출량: {route['total_cmcs']:.2f}"
        )
        folium.PolyLine(
            coordinates,
            color=color,
            weight=7,
            opacity=0.78,
            tooltip=label,
            popup=popup,
        ).add_to(map_object)
    folium.Marker(
        origin,
        tooltip=f"출발: {endpoints['origin_name']}",
        popup=endpoints["origin_address"],
        icon=folium.Icon(color="blue", icon="education"),
    ).add_to(map_object)
    folium.Marker(
        destination,
        tooltip=f"도착: {endpoints['destination_name']}",
        popup=endpoints["destination_address"],
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(map_object)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(path)
    return path


def run_actual_route_recommendation(
    graph: nx.MultiDiGraph,
    edge_scores: pd.DataFrame,
    data_dir: str | Path = "data",
) -> dict[str, object]:
    endpoints = _actual_route_endpoints(Path(data_dir))
    cmcs = edge_scores[["edge_id", "cmcs_final"]].rename(
        columns={"cmcs_final": "cmcs"}
    )
    optimizer = RouteOptimizer(graph=graph, cmcs_data=cmcs)
    origin = endpoints["origin"]
    destination = endpoints["destination"]
    shortest = optimizer.shortest_route(origin, destination)
    safest = optimizer.safest_route(origin, destination)
    balanced = optimizer.balanced_route(origin, destination, lam=0.65)
    routes = [shortest, safest, balanced]

    comparison = optimizer.compare_routes(origin, destination)
    balanced_row = {
        "mode": balanced["mode"],
        "total_distance_m": balanced["total_distance_m"],
        "total_cmcs": balanced["total_cmcs"],
        "average_cmcs": balanced["average_cmcs"],
        "extra_distance_m": round(
            balanced["total_distance_m"] - shortest["total_distance_m"], 2
        ),
        "cmcs_reduction_pct": round(
            (
                shortest["total_cmcs"] - balanced["total_cmcs"]
            )
            / shortest["total_cmcs"]
            * 100
            if shortest["total_cmcs"] > 0
            else 0,
            2,
        ),
        "num_segments": balanced["num_segments"],
    }
    comparison = pd.concat(
        [
            comparison[~comparison["mode"].str.startswith("균형")],
            pd.DataFrame([balanced_row]),
        ],
        ignore_index=True,
    )
    comparison_path = REPORT_OUTPUT_DIR / "actual_route_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pareto_path = REPORT_OUTPUT_DIR / "actual_route_pareto.csv"
    pareto = optimizer.generate_pareto_front(
        origin, destination, steps=21, save_path=pareto_path
    )
    pareto_chart_path = create_pareto_chart(pareto)
    shortest_ids = set(optimizer.path_edge_ids(shortest))
    safest_ids = set(optimizer.path_edge_ids(safest))
    avoided_ids = shortest_ids - safest_ids
    avoided = edge_scores[
        edge_scores["edge_id"].astype(str).isin(avoided_ids)
    ][
        [
            "edge_id",
            "segment_id",
            "district",
            "highway",
            "length_m",
            "cmcs_final",
            "ml_risk_probability",
            "regional_risk_probability",
        ]
    ].sort_values("cmcs_final", ascending=False)
    avoided_path = REPORT_OUTPUT_DIR / "actual_route_avoided_segments.csv"
    avoided.to_csv(avoided_path, index=False, encoding="utf-8-sig")
    map_path = create_actual_route_map(
        optimizer.G, routes, endpoints
    )
    endpoint_path = REPORT_OUTPUT_DIR / "actual_route_endpoints.json"
    endpoint_path.write_text(
        json.dumps(endpoints, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    route_stability_path = REPORT_OUTPUT_DIR / "route_stability_evaluation.json"
    validation_pairs = _school_route_validation_pairs(
        Path(data_dir),
        pair_count=25,
    )
    route_stability = batch_od_evaluation(
        optimizer,
        validation_pairs,
        age_group="mid",
        hour=8,
        max_detour_ratio=1.6,
        output_path=route_stability_path,
    )
    evaluated_count = int(len(route_stability))
    positive_reduction_ratio = (
        float((route_stability["risk_reduction_pct"] > 0).mean())
        if evaluated_count
        else 0.0
    )
    route_stability_summary = {
        "ground_truth_route_f1": None,
        "f1_not_applicable_reason": (
            "정답 통학 경로 라벨이 없어 경로 선정 자체의 F1은 계산하지 않는다."
        ),
        "requested_pairs": int(len(validation_pairs)),
        "evaluated_pairs": evaluated_count,
        "mean_risk_reduction_pct": round(
            float(route_stability["risk_reduction_pct"].mean()),
            4,
        )
        if evaluated_count
        else 0.0,
        "median_risk_reduction_pct": round(
            float(route_stability["risk_reduction_pct"].median()),
            4,
        )
        if evaluated_count
        else 0.0,
        "positive_risk_reduction_ratio": round(
            positive_reduction_ratio,
            4,
        ),
        "mean_detour_ratio": round(
            float(route_stability["detour_ratio"].mean()),
            4,
        )
        if evaluated_count
        else 0.0,
        "p90_detour_ratio": round(
            float(route_stability["detour_ratio"].quantile(0.90)),
            4,
        )
        if evaluated_count
        else 0.0,
        "detour_exceeded_count": int(
            route_stability["detour_exceeded"].sum()
        )
        if evaluated_count
        else 0,
    }
    route_stability_summary["route_selection_stability_passed"] = (
        evaluated_count >= 20
        and positive_reduction_ratio >= 0.80
        and route_stability_summary["median_risk_reduction_pct"] > 0
        and route_stability_summary["detour_exceeded_count"] == 0
    )
    route_stability_payload = {
        **route_stability_summary,
        "pairs": route_stability.to_dict(orient="records"),
    }
    route_stability_path.write_text(
        json.dumps(route_stability_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "endpoints": endpoints,
        "shortest": shortest,
        "safest": safest,
        "balanced": balanced,
        "comparison": comparison,
        "pareto": pareto,
        "map_path": map_path,
        "comparison_path": comparison_path,
        "pareto_path": pareto_path,
        "pareto_chart_path": pareto_chart_path,
        "avoided_path": avoided_path,
        "endpoint_path": endpoint_path,
        "route_stability": route_stability_summary,
        "route_stability_path": route_stability_path,
    }


def run_full_pipeline(
    data_dir: str | Path = "data",
    refresh_data: bool = False,
    refresh_network: bool = False,
) -> dict[str, object]:
    ensure_directories()
    collected = collect_available_real_data(refresh=refresh_data)
    graph = build_or_load_walk_graph(refresh=refresh_network)
    edge_features, graph = build_edge_feature_table(graph, data_dir=data_dir)
    segment_scores, edge_model_report = train_edge_risk_model(edge_features)
    segment_scores, regional_model_report = train_regional_boosting_model(
        segment_scores
    )
    score_columns = segment_scores[
        [
            "segment_id",
            "cmcs",
            "cmcs_final",
            "ml_risk_probability",
            "regional_risk_probability",
            "regional_risk_prediction",
            "region_x",
            "region_y",
        ]
    ]
    final_edges = edge_features.drop(
        columns=[
            "cmcs_final",
            "ml_risk_probability",
            "regional_risk_probability",
            "regional_risk_prediction",
            "region_x",
            "region_y",
        ],
        errors="ignore",
    ).merge(score_columns, on=["segment_id", "cmcs"], how="left")
    final_edges.to_csv(EDGE_CMCS_PATH, index=False, encoding="utf-8-sig")
    graph = apply_scores_to_graph(graph, segment_scores)
    risk_map_path = create_city_risk_map(segment_scores)
    district_summary, district_chart_path = create_district_safety_outputs(
        segment_scores
    )
    route_result = run_actual_route_recommendation(
        graph, final_edges, data_dir=data_dir
    )
    cmcs_weight_report = json.loads(
        DATA_DRIVEN_REPORT_PATH.read_text(encoding="utf-8")
    )

    report = {
        "status": "completed",
        "data_counts": collected,
        "graph": {
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "unique_segments": int(final_edges["segment_id"].nunique()),
        },
        "cmcs": {
            "mean": round(float(final_edges["cmcs_final"].mean()), 6),
            "median": round(float(final_edges["cmcs_final"].median()), 6),
            "high_risk_edge_count": int(
                (final_edges["cmcs_final"] >= 0.6).sum()
            ),
            "weight_source": "data_driven_statistical",
            "dimension_weights": cmcs_weight_report["weights"][
                "dimension_weights"
            ],
        },
        "model": edge_model_report,
        "edge_model": edge_model_report,
        "regional_boosting_model": regional_model_report,
        "f1_target_achieved": regional_model_report[
            "f1_target_achieved"
        ],
        "route": {
            "origin": route_result["endpoints"]["origin_name"],
            "destination": route_result["endpoints"]["destination_name"],
            "shortest_distance_m": route_result["shortest"][
                "total_distance_m"
            ],
            "safest_distance_m": route_result["safest"]["total_distance_m"],
            "shortest_average_cmcs": route_result["shortest"][
                "average_cmcs"
            ],
            "safest_average_cmcs": route_result["safest"]["average_cmcs"],
            "risk_reduction_pct": round(
                (
                    route_result["shortest"]["total_cmcs"]
                    - route_result["safest"]["total_cmcs"]
                )
                / route_result["shortest"]["total_cmcs"]
                * 100
                if route_result["shortest"]["total_cmcs"] > 0
                else 0,
                2,
            ),
            "avoided_segment_count": int(
                len(
                    pd.read_csv(
                        route_result["avoided_path"],
                        encoding="utf-8-sig",
                    )
                )
            ),
            "stability_validation": route_result["route_stability"],
        },
        "artifacts": {
            "edge_features": str(EDGE_FEATURE_PATH),
            "edge_cmcs": str(EDGE_CMCS_PATH),
            "scored_graph": str(SCORED_GRAPH_PATH),
            "edge_model": str(EDGE_MODEL_PATH),
            "regional_model": str(REGIONAL_MODEL_PATH),
            "regional_model_report": str(REGIONAL_MODEL_REPORT_PATH),
            "cmcs_data_driven_weights": str(DATA_DRIVEN_WEIGHTS_PATH),
            "cmcs_weight_evidence_report": str(DATA_DRIVEN_REPORT_PATH),
            "cmcs_weight_evidence_chart": str(DATA_DRIVEN_CHART_PATH),
            "route_map": str(route_result["map_path"]),
            "route_comparison": str(route_result["comparison_path"]),
            "city_risk_map": str(risk_map_path),
            "route_pareto_chart": str(route_result["pareto_chart_path"]),
            "avoided_segments": str(route_result["avoided_path"]),
            "route_stability_evaluation": str(
                route_result["route_stability_path"]
            ),
            "district_summary": str(
                REPORT_OUTPUT_DIR / "district_safety_summary.csv"
            ),
            "district_radar": str(district_chart_path),
        },
    }
    FULL_PIPELINE_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
