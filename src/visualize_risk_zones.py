from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

from config import CHART_OUTPUT_DIR, MAP_OUTPUT_DIR, REPORT_OUTPUT_DIR


PROJECTED_CRS = "EPSG:5179"
WGS84_CRS = "EPSG:4326"
PILOT_BUFFER_M = 700.0


def _artifact_ref(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _edge_gdf_from_graph(graph: nx.Graph) -> gpd.GeoDataFrame:
    import osmnx as ox

    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    edges = edges.reset_index()
    if edges.crs is None:
        edges = edges.set_crs(WGS84_CRS)
    return edges


def _edge_gdf_from_csv(path: Path) -> gpd.GeoDataFrame:
    frame = pd.read_csv(path)
    if "geometry" in frame.columns:
        geometry_column = "geometry"
    elif "geometry_wkt" in frame.columns:
        geometry_column = "geometry_wkt"
    else:
        raise ValueError(f"{path} does not contain geometry or geometry_wkt")
    geometry = frame[geometry_column].map(wkt.loads)
    edges = gpd.GeoDataFrame(frame.copy(), geometry=geometry, crs=PROJECTED_CRS)
    return edges


def _ensure_edge_gdf(edge_source: Any) -> gpd.GeoDataFrame:
    if isinstance(edge_source, gpd.GeoDataFrame):
        edges = edge_source.copy()
    elif isinstance(edge_source, nx.Graph):
        edges = _edge_gdf_from_graph(edge_source)
    else:
        path = Path(edge_source)
        if path.suffix.lower() == ".graphml":
            import osmnx as ox

            edges = _edge_gdf_from_graph(ox.load_graphml(path))
        elif path.suffix.lower() == ".csv":
            edges = _edge_gdf_from_csv(path)
        else:
            edges = gpd.read_file(path)
    if edges.crs is None:
        edges = edges.set_crs(WGS84_CRS)
    return edges


def _score_edges(edges: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    scored = edges.copy()
    for candidate in ("cmcs", "cmcs_final"):
        if candidate in scored.columns and scored[candidate].notna().any():
            scored["risk_score"] = pd.to_numeric(
                scored[candidate], errors="coerce"
            )
            return scored.dropna(subset=["risk_score"]), candidate

    if {"safety_cost", "length"}.issubset(scored.columns):
        length = pd.to_numeric(scored["length"], errors="coerce").replace(0, np.nan)
        scored["risk_score"] = pd.to_numeric(
            scored["safety_cost"], errors="coerce"
        ) / length
        return scored.dropna(subset=["risk_score"]), "safety_cost/length"

    if {"risk_exposure", "length"}.issubset(scored.columns):
        length = pd.to_numeric(scored["length"], errors="coerce").replace(0, np.nan)
        scored["risk_score"] = pd.to_numeric(
            scored["risk_exposure"], errors="coerce"
        ) / length
        return scored.dropna(subset=["risk_score"]), "risk_exposure/length"

    raise ValueError("No CMCS or length-normalized risk field is available")


def _classify_edges(edges: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, float, float]:
    classified = edges.copy()
    q90 = float(classified["risk_score"].quantile(0.90))
    q70 = float(classified["risk_score"].quantile(0.70))
    classified["risk_class"] = np.select(
        [
            classified["risk_score"] >= q90,
            classified["risk_score"] >= q70,
        ],
        ["high", "mid"],
        default="normal",
    )
    return classified, q90, q70


def _prepare_classified_edges(
    edge_source: Any,
) -> tuple[gpd.GeoDataFrame, str, float, float]:
    edges = _ensure_edge_gdf(edge_source)
    scored, field = _score_edges(edges)
    classified, q90, q70 = _classify_edges(scored)
    return classified, field, q90, q70


def _to_projected(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.crs is None:
        edges = edges.set_crs(WGS84_CRS)
    return edges.to_crs(PROJECTED_CRS)


def _to_wgs84(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if edges.crs is None:
        edges = edges.set_crs(PROJECTED_CRS)
    return edges.to_crs(WGS84_CRS)


def _route_edge_gdf(
    classified_edges: gpd.GeoDataFrame, route: dict[str, Any] | None
) -> gpd.GeoDataFrame:
    if not route:
        return classified_edges.iloc[0:0].copy()
    if "edge_gdf" in route and isinstance(route["edge_gdf"], gpd.GeoDataFrame):
        route_edges = route["edge_gdf"].copy()
        if route_edges.crs is None:
            route_edges = route_edges.set_crs(classified_edges.crs)
        return route_edges

    edge_path = route.get("edge_path") or []
    if not edge_path:
        return classified_edges.iloc[0:0].copy()

    edges = classified_edges.copy()
    for column in ("u", "v", "key"):
        if column not in edges.columns:
            edges[column] = None
    route_keys = {(str(u), str(v), str(k)) for u, v, k in edge_path}
    reverse_keys = {(str(v), str(u), str(k)) for u, v, k in edge_path}
    keys = list(
        zip(
            edges["u"].astype(str),
            edges["v"].astype(str),
            edges["key"].astype(str),
        )
    )
    mask = pd.Series(
        [key in route_keys or key in reverse_keys for key in keys],
        index=edges.index,
    )
    route_edges = edges.loc[mask].copy()
    if route_edges.empty and "edge_id" in edges.columns:
        ids = set(map(str, route.get("edge_ids", [])))
        if ids:
            route_edges = edges[edges["edge_id"].astype(str).isin(ids)].copy()
    return route_edges


def _safe_union(geometries: gpd.GeoSeries):
    if geometries.empty:
        return None
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def _split_classes(edges: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    return {
        name: edges[edges["risk_class"].eq(name)]
        for name in ("normal", "mid", "high")
    }


def _plot_risk_layers(axis, edges: gpd.GeoDataFrame) -> None:
    layers = _split_classes(edges)
    if not layers["normal"].empty:
        layers["normal"].plot(
            ax=axis, color="#9ca3af", linewidth=0.18, alpha=0.32
        )
    if not layers["mid"].empty:
        layers["mid"].plot(
            ax=axis, color="#f59e0b", linewidth=0.55, alpha=0.72
        )
    if not layers["high"].empty:
        layers["high"].plot(
            ax=axis, color="#dc2626", linewidth=0.95, alpha=0.92
        )


def plot_daejeon_cmcs_risk_overview(
    edge_source: Any,
    output_path: str | Path = CHART_OUTPUT_DIR / "daejeon_cmcs_risk_overview.png",
) -> Path:
    classified, _, _, _ = _prepare_classified_edges(edge_source)
    edges = _to_projected(classified)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 8))
    _plot_risk_layers(axis, edges)
    axis.set_axis_off()
    axis.set_title(
        "Daejeon CMCS Risk Zones: top 10% and 70-90% road segments",
        fontsize=13,
        pad=12,
    )
    figure.tight_layout(pad=0.1)
    figure.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return path


def _add_edge_layer(
    folium_module,
    map_object,
    edges: gpd.GeoDataFrame,
    name: str,
    color: str,
    weight: float,
    opacity: float,
) -> None:
    if edges.empty:
        return
    tooltip_fields = [
        column
        for column in ("edge_id", "segment_id", "risk_score", "risk_class")
        if column in edges.columns
    ]
    export_edges = edges[[*tooltip_fields, "geometry"]].copy()
    if "risk_score" in export_edges.columns:
        export_edges["risk_score"] = export_edges["risk_score"].round(6)
    layer = folium_module.FeatureGroup(name=name, show=name != "Normal risk roads")
    folium_module.GeoJson(
        json.loads(export_edges.to_json()),
        style_function=lambda _feature, color=color, weight=weight, opacity=opacity: {
            "color": color,
            "weight": weight,
            "opacity": opacity,
        },
        tooltip=folium_module.GeoJsonTooltip(fields=tooltip_fields)
        if tooltip_fields
        else None,
    ).add_to(layer)
    layer.add_to(map_object)


def _folium_center(edges: gpd.GeoDataFrame) -> list[float]:
    if edges.empty:
        return [36.3504, 127.3845]
    minx, miny, maxx, maxy = edges.total_bounds
    return [(miny + maxy) / 2, (minx + maxx) / 2]


def make_interactive_daejeon_cmcs_risk_overview(
    edge_source: Any,
    output_path: str | Path = MAP_OUTPUT_DIR / "daejeon_cmcs_risk_overview.html",
) -> Path:
    import folium

    classified, _, _, _ = _prepare_classified_edges(edge_source)
    edges = _to_wgs84(classified)
    layers = _split_classes(edges)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    map_object = folium.Map(
        location=_folium_center(edges),
        zoom_start=12,
        tiles="CartoDB positron",
    )
    normal = layers["normal"]
    if len(normal) > 3500:
        normal = normal.sample(3500, random_state=42)
    _add_edge_layer(
        folium,
        map_object,
        normal,
        "Normal risk roads",
        "#9ca3af",
        1.0,
        0.22,
    )
    _add_edge_layer(
        folium,
        map_object,
        layers["mid"],
        "Mid risk roads (q70-q90)",
        "#f59e0b",
        2.2,
        0.74,
    )
    _add_edge_layer(
        folium,
        map_object,
        layers["high"],
        "High risk roads (q90+)",
        "#dc2626",
        3.1,
        0.90,
    )
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(path)
    return path


def _pilot_context(
    classified_edges: gpd.GeoDataFrame,
    shortest_route: dict[str, Any] | None,
    safest_route: dict[str, Any] | None,
    buffer_m: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    edges = _to_projected(classified_edges)
    shortest_edges = _to_projected(_route_edge_gdf(classified_edges, shortest_route))
    safest_edges = _to_projected(_route_edge_gdf(classified_edges, safest_route))
    route_union = _safe_union(pd.concat([shortest_edges, safest_edges]).geometry)
    if route_union is None or route_union.is_empty:
        warnings.warn(
            "Pilot route geometry is unavailable; pilot map falls back to all edges.",
            RuntimeWarning,
            stacklevel=2,
        )
        return edges, shortest_edges, safest_edges
    buffer_geometry = route_union.buffer(buffer_m)
    pilot_edges = edges[edges.intersects(buffer_geometry)].copy()
    return pilot_edges, shortest_edges, safest_edges


def plot_pilot_cmcs_risk_zone_map(
    edge_source: Any,
    shortest_route: dict[str, Any] | None = None,
    safest_route: dict[str, Any] | None = None,
    endpoints: dict[str, Any] | None = None,
    output_path: str | Path = CHART_OUTPUT_DIR / "pilot_cmcs_risk_zone_map.png",
    buffer_m: float = PILOT_BUFFER_M,
) -> Path:
    classified, _, _, _ = _prepare_classified_edges(edge_source)
    pilot_edges, shortest_edges, safest_edges = _pilot_context(
        classified, shortest_route, safest_route, buffer_m
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 8))
    _plot_risk_layers(axis, pilot_edges)
    if not shortest_edges.empty:
        shortest_edges.plot(
            ax=axis, color="#2563eb", linewidth=2.7, alpha=0.92, label="Shortest"
        )
    if not safest_edges.empty:
        safest_edges.plot(
            ax=axis, color="#16a34a", linewidth=2.7, alpha=0.92, label="Safest"
        )
    if endpoints:
        marker_rows = []
        for label, color in (("origin", "#2563eb"), ("destination", "#dc2626")):
            value = endpoints.get(label)
            if value:
                marker_rows.append(
                    {
                        "label": label,
                        "color": color,
                        "geometry": Point(float(value[1]), float(value[0])),
                    }
                )
        if marker_rows:
            markers = gpd.GeoDataFrame(marker_rows, crs=WGS84_CRS).to_crs(
                PROJECTED_CRS
            )
            markers.plot(
                ax=axis,
                color=markers["color"].tolist(),
                markersize=58,
                zorder=5,
                edgecolor="white",
                linewidth=0.8,
            )
    axis.set_axis_off()
    axis.set_title(
        "Pilot route corridor: CMCS risk zones and selected paths",
        fontsize=13,
        pad=12,
    )
    axis.legend(loc="lower left", frameon=True) if axis.get_legend_handles_labels()[0] else None
    figure.tight_layout(pad=0.1)
    figure.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return path


def _add_route_layer(
    folium_module,
    map_object,
    route_edges: gpd.GeoDataFrame,
    name: str,
    color: str,
) -> None:
    if route_edges.empty:
        return
    layer = folium_module.FeatureGroup(name=name, show=True)
    folium_module.GeoJson(
        json.loads(route_edges[["geometry"]].to_json()),
        style_function=lambda _feature, color=color: {
            "color": color,
            "weight": 5,
            "opacity": 0.9,
        },
        tooltip=name,
    ).add_to(layer)
    layer.add_to(map_object)


def make_interactive_pilot_cmcs_risk_zone_map(
    edge_source: Any,
    shortest_route: dict[str, Any] | None = None,
    safest_route: dict[str, Any] | None = None,
    endpoints: dict[str, Any] | None = None,
    output_path: str | Path = MAP_OUTPUT_DIR / "pilot_cmcs_risk_zone_map.html",
    buffer_m: float = PILOT_BUFFER_M,
) -> Path:
    import folium

    classified, _, _, _ = _prepare_classified_edges(edge_source)
    pilot_edges, shortest_edges, safest_edges = _pilot_context(
        classified, shortest_route, safest_route, buffer_m
    )
    pilot_edges_wgs = _to_wgs84(pilot_edges)
    shortest_wgs = _to_wgs84(shortest_edges)
    safest_wgs = _to_wgs84(safest_edges)
    layers = _split_classes(pilot_edges_wgs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    map_object = folium.Map(
        location=_folium_center(pilot_edges_wgs),
        zoom_start=15,
        tiles="CartoDB positron",
    )
    _add_edge_layer(
        folium,
        map_object,
        layers["normal"],
        "Normal risk roads in pilot buffer",
        "#9ca3af",
        1.0,
        0.24,
    )
    _add_edge_layer(
        folium,
        map_object,
        layers["mid"],
        "Mid risk roads in pilot buffer",
        "#f59e0b",
        2.0,
        0.74,
    )
    _add_edge_layer(
        folium,
        map_object,
        layers["high"],
        "High risk roads in pilot buffer",
        "#dc2626",
        3.0,
        0.9,
    )
    _add_route_layer(folium, map_object, shortest_wgs, "Shortest route", "#2563eb")
    _add_route_layer(folium, map_object, safest_wgs, "Safest route", "#16a34a")
    if endpoints:
        for label, color in (("origin", "blue"), ("destination", "red")):
            value = endpoints.get(label)
            if value:
                folium.Marker(
                    location=[float(value[0]), float(value[1])],
                    tooltip=label,
                    icon=folium.Icon(color=color),
                ).add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.save(path)
    return path


def _overlap_count(route_edges: gpd.GeoDataFrame, candidate_edges: gpd.GeoDataFrame) -> int:
    if route_edges.empty or candidate_edges.empty:
        return 0
    if "edge_id" in route_edges.columns and "edge_id" in candidate_edges.columns:
        route_ids = set(route_edges["edge_id"].astype(str))
        candidate_ids = set(candidate_edges["edge_id"].astype(str))
        return len(route_ids & candidate_ids)
    return int(route_edges.intersects(_safe_union(candidate_edges.geometry)).sum())


def summarize_risk_zone_maps(
    edge_source: Any,
    shortest_route: dict[str, Any] | None = None,
    safest_route: dict[str, Any] | None = None,
    generated_files: dict[str, str | Path] | None = None,
    output_path: str | Path = REPORT_OUTPUT_DIR / "risk_zone_map_summary.json",
    buffer_m: float = PILOT_BUFFER_M,
) -> Path:
    classified, field, q90, q70 = _prepare_classified_edges(edge_source)
    pilot_edges, shortest_edges, safest_edges = _pilot_context(
        classified, shortest_route, safest_route, buffer_m
    )
    high_edges = classified[classified["risk_class"].eq("high")]
    mid_edges = classified[classified["risk_class"].eq("mid")]
    pilot_high = pilot_edges[pilot_edges["risk_class"].eq("high")]
    pilot_mid = pilot_edges[pilot_edges["risk_class"].eq("mid")]
    shortest_high_overlap = _overlap_count(shortest_edges, high_edges)
    safest_high_overlap = _overlap_count(safest_edges, high_edges)
    shortest_mid_overlap = _overlap_count(shortest_edges, mid_edges)
    safest_mid_overlap = _overlap_count(safest_edges, mid_edges)
    files = {
        key: _artifact_ref(value)
        for key, value in (generated_files or {}).items()
    }
    payload = {
        "risk_score_field": field,
        "total_edges": int(len(classified)),
        "high_risk_threshold_q90": round(q90, 6),
        "mid_risk_threshold_q70": round(q70, 6),
        "high_risk_edge_count": int(len(high_edges)),
        "mid_risk_edge_count": int(len(mid_edges)),
        "high_risk_edge_ratio": round(len(high_edges) / max(len(classified), 1), 6),
        "mid_risk_edge_ratio": round(len(mid_edges) / max(len(classified), 1), 6),
        "pilot_buffer_m": float(buffer_m),
        "pilot_edges_in_buffer": int(len(pilot_edges)),
        "pilot_high_risk_edges_in_buffer": int(len(pilot_high)),
        "pilot_mid_risk_edges_in_buffer": int(len(pilot_mid)),
        "shortest_route_edge_count": int(len(shortest_edges)),
        "safest_route_edge_count": int(len(safest_edges)),
        "shortest_high_risk_overlap_count": int(shortest_high_overlap),
        "safest_high_risk_overlap_count": int(safest_high_overlap),
        "shortest_mid_risk_overlap_count": int(shortest_mid_overlap),
        "safest_mid_risk_overlap_count": int(safest_mid_overlap),
        "high_risk_edges_avoided_by_safest": int(
            shortest_high_overlap - safest_high_overlap
        ),
        "mid_risk_edges_avoided_by_safest": int(
            shortest_mid_overlap - safest_mid_overlap
        ),
        "generated_files": files,
        "captions": {
            "daejeon_overview": (
                "그림 VIII-1. 대전 전체 보행 도로망의 CMCS 기반 위험구간 분포. "
                "상위 10% 위험구간은 적색, 70-90% 구간은 주황색으로 표시하였다."
            ),
            "pilot_zone": (
                "그림 VIII-2. 파일럿 경로 주변 700m 버퍼 내 CMCS 위험구간과 "
                "최단경로(청색), 최저위험경로(녹색)의 비교."
            ),
            "report_placement": (
                "최종 보고서 VIII장 결과/시각화 절에는 전체 대전 위험지도, "
                "파일럿 경로 주변 위험지도 순서로 삽입한다."
            ),
        },
    }
    for key, value in list(payload.items()):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            payload[key] = None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def generate_risk_zone_map_artifacts(
    edge_source: Any,
    shortest_route: dict[str, Any] | None = None,
    safest_route: dict[str, Any] | None = None,
    endpoints: dict[str, Any] | None = None,
    buffer_m: float = PILOT_BUFFER_M,
) -> dict[str, Path]:
    overview_png = plot_daejeon_cmcs_risk_overview(edge_source)
    overview_html = make_interactive_daejeon_cmcs_risk_overview(edge_source)
    pilot_png = plot_pilot_cmcs_risk_zone_map(
        edge_source,
        shortest_route=shortest_route,
        safest_route=safest_route,
        endpoints=endpoints,
        buffer_m=buffer_m,
    )
    pilot_html = make_interactive_pilot_cmcs_risk_zone_map(
        edge_source,
        shortest_route=shortest_route,
        safest_route=safest_route,
        endpoints=endpoints,
        buffer_m=buffer_m,
    )
    summary = summarize_risk_zone_maps(
        edge_source,
        shortest_route=shortest_route,
        safest_route=safest_route,
        generated_files={
            "daejeon_cmcs_risk_overview_png": overview_png,
            "daejeon_cmcs_risk_overview_html": overview_html,
            "pilot_cmcs_risk_zone_map_png": pilot_png,
            "pilot_cmcs_risk_zone_map_html": pilot_html,
        },
        buffer_m=buffer_m,
    )
    return {
        "daejeon_cmcs_risk_overview_png": overview_png,
        "daejeon_cmcs_risk_overview_html": overview_html,
        "pilot_cmcs_risk_zone_map_png": pilot_png,
        "pilot_cmcs_risk_zone_map_html": pilot_html,
        "risk_zone_map_summary": summary,
    }


def generate_risk_zone_maps_from_saved_artifacts() -> dict[str, Path]:
    from src.full_pipeline import (
        EDGE_CMCS_PATH,
        ROUTE_TIME_BUDGET_S,
        SCORED_GRAPH_PATH,
        _actual_route_endpoints,
    )
    from src.route_optimizer import RouteOptimizer

    import osmnx as ox

    graph = ox.load_graphml(SCORED_GRAPH_PATH)
    edge_scores = pd.read_csv(EDGE_CMCS_PATH)
    cmcs = edge_scores[["edge_id", "cmcs_final"]].rename(
        columns={"cmcs_final": "cmcs"}
    )
    optimizer = RouteOptimizer(
        graph=graph,
        cmcs_data=cmcs,
        route_time_budget_s=ROUTE_TIME_BUDGET_S,
    )
    endpoints = _actual_route_endpoints(Path("data"))
    origin = endpoints["origin"]
    destination = endpoints["destination"]
    shortest = optimizer.shortest_route(origin, destination)
    max_distance_m = shortest["total_distance_m"] * 1.6
    safest = optimizer.safest_route(
        origin,
        destination,
        max_distance_m=max_distance_m,
    )
    return generate_risk_zone_map_artifacts(
        optimizer.G,
        shortest_route=shortest,
        safest_route=safest,
        endpoints=endpoints,
    )
