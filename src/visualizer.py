from __future__ import annotations

from pathlib import Path
from typing import Sequence

import networkx as nx
import pandas as pd


def _node_coordinate(graph: nx.Graph, node) -> tuple[float, float]:
    data = graph.nodes[node]
    return float(data["y"]), float(data["x"])


class Visualizer:
    """CMCS 및 경로 비교 산출물을 생성한다."""

    @staticmethod
    def route_comparison_map(
        graph: nx.Graph,
        shortest_info: dict,
        safest_info: dict,
        save_path: str | Path = "outputs/maps/route_comparison.html",
    ) -> Path:
        try:
            import folium
        except ImportError as exc:
            raise RuntimeError("지도 생성에는 folium이 필요합니다.") from exc

        shortest_coords = [
            _node_coordinate(graph, node) for node in shortest_info["path"]
        ]
        safest_coords = [_node_coordinate(graph, node) for node in safest_info["path"]]
        center = shortest_coords[len(shortest_coords) // 2]
        map_object = folium.Map(
            location=center, zoom_start=16, tiles="CartoDB positron"
        )
        folium.PolyLine(
            shortest_coords,
            color="#3498db",
            weight=6,
            opacity=0.8,
            tooltip=(
                f"최단거리 {shortest_info['total_distance_m']:.0f}m / "
                f"위험노출 {shortest_info['total_cmcs']:.2f}"
            ),
        ).add_to(map_object)
        folium.PolyLine(
            safest_coords,
            color="#2ecc71",
            weight=6,
            opacity=0.85,
            tooltip=(
                f"안전경로 {safest_info['total_distance_m']:.0f}m / "
                f"위험노출 {safest_info['total_cmcs']:.2f}"
            ),
        ).add_to(map_object)
        folium.Marker(shortest_coords[0], tooltip="출발").add_to(map_object)
        folium.Marker(shortest_coords[-1], tooltip="도착").add_to(map_object)

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        map_object.save(path)
        return path

    @staticmethod
    def cmcs_network_map(
        graph: nx.Graph,
        save_path: str | Path = "outputs/maps/cmcs_heatmap.html",
    ) -> Path:
        try:
            import folium
        except ImportError as exc:
            raise RuntimeError("지도 생성에는 folium이 필요합니다.") from exc

        nodes = list(graph.nodes)
        center = _node_coordinate(graph, nodes[len(nodes) // 2])
        map_object = folium.Map(
            location=center, zoom_start=15, tiles="CartoDB positron"
        )
        iterator = (
            graph.edges(keys=True, data=True)
            if graph.is_multigraph()
            else ((u, v, 0, data) for u, v, data in graph.edges(data=True))
        )
        for u, v, _, data in iterator:
            cmcs = float(data.get("cmcs", 0.5))
            color = "#2ecc71" if cmcs < 0.3 else "#f39c12" if cmcs < 0.6 else "#e74c3c"
            folium.PolyLine(
                [_node_coordinate(graph, u), _node_coordinate(graph, v)],
                color=color,
                weight=4,
                opacity=0.65,
                tooltip=f"{data.get('edge_id', '')}: CMCS {cmcs:.3f}",
            ).add_to(map_object)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        map_object.save(path)
        return path

    @staticmethod
    def pareto_front_chart(
        pareto_df: pd.DataFrame,
        save_path: str | Path = "outputs/charts/pareto_front.html",
    ) -> Path:
        try:
            import plotly.express as px
        except ImportError as exc:
            raise RuntimeError("파레토 차트에는 plotly가 필요합니다.") from exc

        figure = px.line(
            pareto_df,
            x="distance_m",
            y="cmcs",
            color="lambda",
            markers=True,
            title="거리–CMCS 위험 노출 파레토 프론트",
            labels={
                "distance_m": "이동 거리 (m)",
                "cmcs": "CMCS 위험 노출량",
                "lambda": "거리 가중치 λ",
            },
        )
        figure.update_layout(template="plotly_white")
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path)
        return path

    @staticmethod
    def district_radar(
        district_stats: pd.DataFrame,
        categories: Sequence[str],
        save_path: str | Path = "outputs/charts/district_radar.html",
    ) -> Path:
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise RuntimeError("레이더 차트에는 plotly가 필요합니다.") from exc

        figure = go.Figure()
        for _, row in district_stats.iterrows():
            figure.add_trace(
                go.Scatterpolar(
                    r=[row[column] for column in categories],
                    theta=list(categories),
                    fill="toself",
                    name=row["district"],
                )
            )
        figure.update_layout(
            title="구별 안전 지표 비교",
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            template="plotly_white",
        )
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path)
        return path

