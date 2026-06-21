from __future__ import annotations

from pathlib import Path
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd


DATASET_FEATURES: dict[str, tuple[str, float]] = {
    "traffic_accident": ("accident_count", 30.0),
    "crosswalk": ("crosswalk_count", 20.0),
    "traffic_signal": ("signal_count", 20.0),
    "speed_bump": ("speed_bump_count", 20.0),
    "cctv": ("cctv_count", 50.0),
    "streetlight": ("streetlight_count", 20.0),
    "academy": ("academy_density", 200.0),
    "bus_stop": ("bus_stop_nearby", 50.0),
    "school_zone": ("in_school_zone", 50.0),
}


class Preprocessor:
    """좌표 변환, 도로 간선 공간 매칭, 결측 처리와 정규화를 담당한다."""

    def __init__(
        self,
        graph: nx.Graph | None = None,
        graph_path: str | Path | None = None,
    ):
        if graph is None and graph_path is None:
            raise ValueError("graph 또는 graph_path 중 하나가 필요합니다.")
        self.G = graph if graph is not None else self._load_graph(graph_path)
        self._ensure_edge_ids()

    @staticmethod
    def _load_graph(path: str | Path | None) -> nx.Graph:
        try:
            import osmnx as ox
        except ImportError:
            return nx.read_graphml(Path(path or ""))
        return ox.load_graphml(Path(path or ""))

    def _ensure_edge_ids(self) -> None:
        iterator = (
            self.G.edges(keys=True, data=True)
            if self.G.is_multigraph()
            else ((u, v, 0, data) for u, v, data in self.G.edges(data=True))
        )
        for index, (_, _, _, data) in enumerate(iterator):
            data.setdefault("edge_id", f"E{index:07d}")

    @staticmethod
    def unify_crs(
        df: pd.DataFrame,
        lon_col: str = "경도",
        lat_col: str = "위도",
        source_crs: str = "EPSG:4326",
        target_crs: str = "EPSG:4326",
    ):
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError as exc:
            raise RuntimeError(
                "공간 전처리에는 geopandas와 shapely가 필요합니다."
            ) from exc

        missing = {lon_col, lat_col} - set(df.columns)
        if missing:
            raise ValueError(f"좌표 컬럼이 없습니다: {sorted(missing)}")
        valid = df.dropna(subset=[lon_col, lat_col]).copy()
        geometry = [
            Point(float(lon), float(lat))
            for lon, lat in zip(valid[lon_col], valid[lat_col])
        ]
        return gpd.GeoDataFrame(valid, geometry=geometry, crs=source_crs).to_crs(
            target_crs
        )

    def edges_gdf(self):
        try:
            import osmnx as ox
        except ImportError as exc:
            raise RuntimeError("간선 공간 변환에는 osmnx가 필요합니다.") from exc
        return ox.graph_to_gdfs(self.G, nodes=False).reset_index()

    def spatial_join_to_edges(
        self,
        points,
        buffer_m: float = 30.0,
        agg_col: str | None = None,
        agg_func: str = "count",
        projected_crs: str = "EPSG:5179",
    ) -> pd.Series:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise RuntimeError("공간 조인에는 geopandas가 필요합니다.") from exc

        edges = self.edges_gdf().to_crs(projected_crs)
        points_projected = points.to_crs(projected_crs)
        edge_lookup = edges[["edge_id", "geometry"]].copy()

        joined = gpd.sjoin_nearest(
            points_projected,
            edge_lookup,
            how="inner",
            max_distance=buffer_m,
            distance_col="_distance_m",
        )
        if joined.empty:
            return pd.Series(dtype=float)
        if agg_col and agg_func != "count":
            if agg_col not in joined:
                raise ValueError(f"집계 컬럼 {agg_col!r}이 없습니다.")
            return joined.groupby("edge_id")[agg_col].agg(agg_func)
        return joined.groupby("edge_id").size()

    def build_edge_features(
        self,
        datasets: Mapping[str, object],
        save_path: str | Path | None = "data/processed/edge_features.csv",
    ) -> pd.DataFrame:
        records = []
        iterator = (
            self.G.edges(keys=True, data=True)
            if self.G.is_multigraph()
            else ((u, v, 0, data) for u, v, data in self.G.edges(data=True))
        )
        for u, v, key, data in iterator:
            records.append(
                {
                    "edge_id": str(data["edge_id"]),
                    "u": u,
                    "v": v,
                    "key": key,
                    "length_m": float(data.get("length", 0.0)),
                }
            )
        features = pd.DataFrame(records)

        for dataset_name, (feature_name, radius) in DATASET_FEATURES.items():
            if dataset_name not in datasets:
                continue
            counts = self.spatial_join_to_edges(datasets[dataset_name], radius)
            features = features.merge(
                counts.rename(feature_name).reset_index(), on="edge_id", how="left"
            )

        count_columns = [feature for feature, _ in DATASET_FEATURES.values()]
        for column in count_columns:
            if column not in features:
                features[column] = 0.0
        features[count_columns] = features[count_columns].fillna(0.0)

        features["light_density"] = features["streetlight_count"] / features[
            "length_m"
        ].clip(lower=1.0)
        features["has_crosswalk"] = (features["crosswalk_count"] > 0).astype(int)
        features["has_signal"] = (features["signal_count"] > 0).astype(int)
        features["has_speed_bump"] = (features["speed_bump_count"] > 0).astype(int)
        features["has_cctv"] = (features["cctv_count"] > 0).astype(int)
        features["is_school_zone"] = (features["in_school_zone"] > 0).astype(int)

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(path, index=False)
        return features

    @staticmethod
    def normalize_features(
        df: pd.DataFrame,
        columns: list[str],
        suffix: str = "_norm",
    ) -> pd.DataFrame:
        result = df.copy()
        for column in columns:
            values = pd.to_numeric(result[column], errors="coerce")
            minimum = values.min()
            maximum = values.max()
            if pd.isna(minimum) or np.isclose(maximum, minimum):
                result[f"{column}{suffix}"] = 0.0
            else:
                result[f"{column}{suffix}"] = (
                    (values - minimum) / (maximum - minimum)
                ).fillna(0.0)
        return result

