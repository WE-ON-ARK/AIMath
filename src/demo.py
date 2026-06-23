from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd

from config import (
    CHART_OUTPUT_DIR,
    GRAPH_DATA_DIR,
    MAP_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    REPORT_OUTPUT_DIR,
    ensure_directories,
)
from src.cmcs_calculator import CMCSCalculator
from src.route_optimizer import RouteOptimizer
from src.visualizer import Visualizer


def build_demo_graph(size: int = 5) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """대전 중심 좌표 부근의 작은 격자 도로망과 합성 안전 피처를 만든다."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    base_lat, base_lon = 36.3504, 127.3845
    lat_step, lon_step = 0.00082, 0.00102

    for x in range(size):
        for y in range(size):
            node = f"{x}-{y}"
            graph.add_node(
                node,
                x=base_lon + (x - size // 2) * lon_step,
                y=base_lat + (y - size // 2) * lat_step,
            )

    features: list[dict] = []
    edge_index = 0
    for x in range(size):
        for y in range(size):
            neighbors = []
            if x + 1 < size:
                neighbors.append((x + 1, y, 90.0))
            if y + 1 < size:
                neighbors.append((x, y + 1, 110.0))

            for nx_x, nx_y, length in neighbors:
                for source, target in (
                    (f"{x}-{y}", f"{nx_x}-{nx_y}"),
                    (f"{nx_x}-{nx_y}", f"{x}-{y}"),
                ):
                    edge_id = f"DEMO-E{edge_index:04d}"
                    edge_index += 1
                    graph.add_edge(
                        source,
                        target,
                        edge_id=edge_id,
                        length=length,
                    )

                    middle_y = (y + nx_y) / 2
                    is_risky_corridor = 1.75 <= middle_y <= 2.25
                    is_safe_corridor = middle_y <= 1.25
                    features.append(
                        {
                            "edge_id": edge_id,
                            "length_m": length,
                            "accident_count_norm": 0.90 if is_risky_corridor else 0.08,
                            "traffic_volume_norm": 0.85 if is_risky_corridor else 0.18,
                            "avg_speed_norm": 0.80 if is_risky_corridor else 0.20,
                            "narrow_sidewalk_norm": 0.75 if is_risky_corridor else 0.18,
                            "slope_norm": 0.12,
                            "is_alley": int(is_risky_corridor),
                            "pedestrian_flow_norm": 0.70 if is_risky_corridor else 0.22,
                            "academy_density_norm": 0.82 if is_risky_corridor else 0.15,
                            "bus_stop_nearby_norm": 0.65 if is_risky_corridor else 0.10,
                            "illegal_parking_norm": 0.88 if is_risky_corridor else 0.10,
                            "light_density_norm": 0.28 if is_risky_corridor else 0.88,
                            "has_crosswalk": int(not is_risky_corridor),
                            "has_signal": int(not is_risky_corridor),
                            "lane_count_norm": 0.80 if is_risky_corridor else 0.15,
                            "has_speed_bump": int(is_safe_corridor),
                            "has_cctv": int(is_safe_corridor),
                            "is_school_zone": int(is_safe_corridor),
                        }
                    )
    return graph, pd.DataFrame(features)


def run_demo(with_visuals: bool = True) -> dict[str, object]:
    ensure_directories()
    graph, features = build_demo_graph()
    scored = CMCSCalculator().score(features)
    scored.to_csv(PROCESSED_DATA_DIR / "edge_cmcs.csv", index=False)

    optimizer = RouteOptimizer(graph=graph, cmcs_data=scored)
    origin, destination = "0-2", "4-2"
    shortest = optimizer.shortest_route(origin, destination)
    safest = optimizer.safest_route(origin, destination)
    comparison = optimizer.compare_routes(origin, destination)
    comparison.to_csv(REPORT_OUTPUT_DIR / "route_comparison.csv", index=False)
    pareto = optimizer.generate_pareto_front(
        origin,
        destination,
        steps=11,
        save_path=REPORT_OUTPUT_DIR / "pareto_front.csv",
    )
    pulse_metrics = comparison[
        [
            "mode",
            "algorithm",
            "runtime_ms",
            "pulses_generated",
            "states_expanded",
            "bound_prunes",
            "resource_prunes",
            "dominance_prunes",
            "optimality_proven",
        ]
    ]
    pulse_metrics.to_csv(
        REPORT_OUTPUT_DIR / "pulse_algorithm_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (REPORT_OUTPUT_DIR / "pulse_algorithm_evaluation.json").write_text(
        json.dumps(
            {
                "algorithm": "pulse",
                "dijkstra_used": False,
                "all_optimality_proven": bool(
                    pulse_metrics["optimality_proven"].all()
                ),
                "routes": pulse_metrics.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    nx.write_graphml(optimizer.G, GRAPH_DATA_DIR / "demo_walk.graphml")

    visual_outputs: list[Path] = []
    if with_visuals:
        try:
            visual_outputs.extend(
                [
                    Visualizer.cmcs_network_map(
                        optimizer.G, MAP_OUTPUT_DIR / "cmcs_heatmap.html"
                    ),
                    Visualizer.route_comparison_map(
                        optimizer.G,
                        shortest,
                        safest,
                        MAP_OUTPUT_DIR / "route_comparison.html",
                    ),
                    Visualizer.pareto_front_chart(
                        pareto, CHART_OUTPUT_DIR / "pareto_front.html"
                    ),
                ]
            )
        except RuntimeError as exc:
            print(f"[시각화 생략] {exc}")

    return {
        "graph": optimizer.G,
        "features": scored,
        "shortest": shortest,
        "safest": safest,
        "comparison": comparison,
        "pareto": pareto,
        "visual_outputs": visual_outputs,
    }

