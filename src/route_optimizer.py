from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from src.aco_pareto_rcsp import HybridACOParetoRCSP


Coordinate = tuple[float, float]
EdgeRef = tuple[object, object, object]


class RouteOptimizer:
    """거리와 CMCS 위험 노출량을 함께 다루는 경로 최적화 엔진."""

    def __init__(
        self,
        graph: nx.Graph | None = None,
        cmcs_data: pd.DataFrame | None = None,
        graph_path: str | Path | None = None,
        cmcs_path: str | Path | None = None,
        default_cmcs: float = 0.5,
        risk_floor: float = 0.05,
        algorithm: str = "aco_pareto_rcsp",
        route_time_budget_s: float | None = None,
    ):
        if graph is None and graph_path is None:
            raise ValueError("graph 또는 graph_path 중 하나가 필요합니다.")
        self.G = graph.copy() if graph is not None else self._load_graph(graph_path)
        self.default_cmcs = float(default_cmcs)
        self.risk_floor = float(risk_floor)
        if algorithm not in {"aco_pareto_rcsp", "bidirectional", "pulse"}:
            raise ValueError("algorithm must be 'aco_pareto_rcsp'")
        self._algorithm_name = "aco_pareto_rcsp"
        self._requested_algorithm = algorithm
        # 경로 1건당 탐색 시간 상한(anytime 안전망). 초과 시 현재까지 찾은
        # 최적 incumbent(시드가 빠르게 최적값을 찾으므로 보통 최적값)를 반환한다.
        self._route_time_budget_s = route_time_budget_s

        if cmcs_data is None and cmcs_path is not None:
            cmcs_data = pd.read_csv(cmcs_path)
        self._assign_cmcs_weights(cmcs_data)
        self.router = HybridACOParetoRCSP(self.G)

    @staticmethod
    def _load_graph(graph_path: str | Path | None) -> nx.Graph:
        path = Path(graph_path or "")
        try:
            import osmnx as ox
        except ImportError:
            return nx.read_graphml(path)
        return ox.load_graphml(path)

    def _assign_cmcs_weights(self, cmcs_data: pd.DataFrame | None) -> None:
        cmcs_by_edge: dict[str, float] = {}
        if cmcs_data is not None:
            required = {"edge_id", "cmcs"}
            if not required.issubset(cmcs_data.columns):
                raise ValueError("cmcs_data에는 edge_id와 cmcs 컬럼이 필요합니다.")
            cmcs_by_edge = {
                str(edge_id): float(np.clip(cmcs, 0.0, 1.0))
                for edge_id, cmcs in zip(cmcs_data["edge_id"], cmcs_data["cmcs"])
            }

        for _, _, _, data in self._iter_edges():
            edge_id = str(data.get("edge_id", ""))
            cmcs = cmcs_by_edge.get(edge_id, data.get("cmcs", self.default_cmcs))
            length = max(float(data.get("length", 1.0)), 0.01)
            data["length"] = length
            data["cmcs"] = float(np.clip(cmcs, 0.0, 1.0))
            data["risk_exposure"] = length * data["cmcs"]
            data["safety_cost"] = length * (self.risk_floor + data["cmcs"])

    def _iter_edges(self):
        if self.G.is_multigraph():
            yield from self.G.edges(keys=True, data=True)
        else:
            for u, v, data in self.G.edges(data=True):
                yield u, v, 0, data

    def find_nearest_node(self, lat: float, lon: float):
        try:
            import osmnx as ox
        except ImportError:
            ox = None
        if ox is not None:
            try:
                return ox.nearest_nodes(self.G, lon, lat)
            except Exception:
                pass

        candidates = []
        for node, data in self.G.nodes(data=True):
            if "x" in data and "y" in data:
                distance_sq = (float(data["x"]) - lon) ** 2 + (
                    float(data["y"]) - lat
                ) ** 2
                candidates.append((distance_sq, node))
        if not candidates:
            raise ValueError("최근접 노드 탐색을 위한 x/y 노드 좌표가 없습니다.")
        return min(candidates, key=lambda item: item[0])[1]

    def shortest_route(
        self, origin: Coordinate | object, destination: Coordinate | object
    ) -> dict:
        return self._route(origin, destination, "length", "최단거리")

    def safest_route(
        self,
        origin: Coordinate | object,
        destination: Coordinate | object,
        max_distance_m: float | None = None,
    ) -> dict:
        return self._route(
            origin,
            destination,
            "safety_cost",
            "최저위험",
            max_distance_m=max_distance_m,
        )

    def balanced_route(
        self,
        origin: Coordinate | object,
        destination: Coordinate | object,
        lam: float = 0.5,
        max_distance_m: float | None = None,
    ) -> dict:
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lambda는 0과 1 사이여야 합니다.")
        for _, _, _, data in self._iter_edges():
            risk_multiplier = self.risk_floor + data["cmcs"]
            data["balanced_cost"] = data["length"] * (
                lam + (1.0 - lam) * risk_multiplier
            )
        return self._route(
            origin,
            destination,
            "balanced_cost",
            f"균형 (λ={lam:.2f})",
            extra={"lambda": lam},
            max_distance_m=max_distance_m,
        )

    def _resolve_node(self, value: Coordinate | object):
        if isinstance(value, tuple) and len(value) == 2:
            return self.find_nearest_node(float(value[0]), float(value[1]))
        if value not in self.G:
            raise nx.NodeNotFound(f"그래프에 노드 {value!r}가 없습니다.")
        return value

    def _route(
        self,
        origin: Coordinate | object,
        destination: Coordinate | object,
        cost_attribute: str,
        mode: str,
        extra: dict | None = None,
        max_distance_m: float | None = None,
    ) -> dict:
        origin_node = self._resolve_node(origin)
        destination_node = self._resolve_node(destination)
        search = self.router.solve(
            origin_node,
            destination_node,
            cost_attribute,
            resource_attribute="length",
            max_resource=max_distance_m,
            time_budget_s=self._route_time_budget_s,
        )
        path = search.path
        edge_path = search.edge_path
        total_distance = self._sum_edges(edge_path, "length")
        total_risk_exposure = self._sum_edges(edge_path, "risk_exposure")
        average_cmcs = (
            total_risk_exposure / total_distance if total_distance else 0.0
        )
        result = {
            "mode": mode,
            "path": path,
            "edge_path": edge_path,
            "total_distance_m": round(total_distance, 2),
            "total_cmcs": round(total_risk_exposure, 4),
            "average_cmcs": round(average_cmcs, 4),
            "num_segments": len(edge_path),
            "algorithm": search.search_stats["algorithm"],
            "selected_source": search.selected_source,
            "optimality_proven": search.optimality_proven,
            "optimality_claim_scope": search.optimality_claim_scope,
            "aco_found_feasible": search.aco_found_feasible,
            "aco_objective": search.aco_objective,
            "aco_total_distance": search.aco_total_distance,
            "aco_total_cmcs": search.aco_total_cmcs,
            "aco_feasible_solutions": search.aco_feasible_solutions,
            "pure_aco_feasible_solutions": search.pure_aco_feasible_solutions,
            "seeded_feasible_solutions": search.seeded_feasible_solutions,
            "rcsp_objective": search.rcsp_objective,
            "rcsp_total_distance": search.rcsp_total_distance,
            "rcsp_total_cmcs": search.rcsp_total_cmcs,
            "rcsp_used_aco_upper_bound": search.rcsp_used_aco_upper_bound,
            "initial_upper_bound_source": search.initial_upper_bound_source,
            "detour_ratio": search.detour_ratio,
            "detour_constraint_satisfied": search.detour_constraint_satisfied,
            "risk_reduction_pct_against_shortest": (
                search.risk_reduction_pct_against_shortest
            ),
            "distance_increase_pct_against_shortest": (
                search.distance_increase_pct_against_shortest
            ),
            "objective_cost": round(search.objective_cost, 6),
            "search_stats": search.search_stats,
            "aco_stats": search.aco_stats,
            "rcsp_stats": search.rcsp_stats,
            "gap_pct": search.gap_pct,
            "max_distance_m": max_distance_m,
        }
        if extra:
            result.update(extra)
        return result

    def _select_edge_path(
        self, path: Sequence[object], cost_attribute: str
    ) -> list[EdgeRef]:
        selected: list[EdgeRef] = []
        for u, v in zip(path, path[1:]):
            if self.G.is_multigraph():
                edges = self.G.get_edge_data(u, v) or {}
                key, _ = min(
                    edges.items(),
                    key=lambda item: float(item[1].get(cost_attribute, np.inf)),
                )
            else:
                if self.G.get_edge_data(u, v) is None:
                    raise nx.NetworkXNoPath(f"{u!r} -> {v!r} 간선이 없습니다.")
                key = 0
            selected.append((u, v, key))
        return selected

    def _edge_data(self, edge: EdgeRef) -> dict:
        u, v, key = edge
        if self.G.is_multigraph():
            return self.G[u][v][key]
        return self.G[u][v]

    def _sum_edges(self, edge_path: Iterable[EdgeRef], attribute: str) -> float:
        return float(
            sum(float(self._edge_data(edge).get(attribute, 0.0)) for edge in edge_path)
        )

    def path_edge_ids(self, route: dict) -> list[str]:
        return [
            str(self._edge_data(edge).get("edge_id", f"{edge[0]}-{edge[1]}"))
            for edge in route["edge_path"]
        ]

    def generate_pareto_front(
        self,
        origin: Coordinate | object,
        destination: Coordinate | object,
        steps: int = 11,
        save_path: str | Path | None = None,
        max_distance_m: float | None = None,
    ) -> pd.DataFrame:
        if steps < 2:
            raise ValueError("steps는 2 이상이어야 합니다.")
        records = []
        for lam in np.linspace(0.0, 1.0, steps):
            route = self.balanced_route(
                origin,
                destination,
                float(lam),
                max_distance_m=max_distance_m,
            )
            records.append(
                {
                    "lambda": round(float(lam), 4),
                    "distance_m": route["total_distance_m"],
                    "cmcs": route["total_cmcs"],
                    "average_cmcs": route["average_cmcs"],
                    "num_segments": route["num_segments"],
                    "runtime_ms": route["search_stats"]["runtime_ms"],
                    "aco_runtime_ms": route["search_stats"]["aco_runtime_ms"],
                    "rcsp_runtime_ms": route["search_stats"]["rcsp_runtime_ms"],
                    "selected_source": route["selected_source"],
                    "gap_pct": route["gap_pct"],
                    "ants_total": route["search_stats"]["ants_total"],
                    "labels_created": route["search_stats"]["labels_created"],
                    "labels_expanded": route["search_stats"]["labels_expanded"],
                    "dominance_prunes": route["search_stats"][
                        "dominance_prunes"
                    ],
                    "resource_prunes": route["search_stats"][
                        "resource_prunes"
                    ],
                    "upper_bound_prunes": route["search_stats"][
                        "upper_bound_prunes"
                    ],
                    "optimality_proven": route["optimality_proven"],
                    "detour_ratio": route["detour_ratio"],
                    "detour_constraint_satisfied": route[
                        "detour_constraint_satisfied"
                    ],
                }
            )
        result = pd.DataFrame(records).drop_duplicates(
            subset=["distance_m", "cmcs"], keep="first"
        )
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)
        return result

    def compare_routes(
        self,
        origin: Coordinate | object,
        destination: Coordinate | object,
        max_detour_ratio: float | None = None,
    ) -> pd.DataFrame:
        shortest = self.shortest_route(origin, destination)
        max_distance_m = (
            shortest["total_distance_m"] * max_detour_ratio
            if max_detour_ratio is not None
            else None
        )
        routes = [
            shortest,
            self.safest_route(
                origin,
                destination,
                max_distance_m=max_distance_m,
            ),
            self.balanced_route(
                origin,
                destination,
                0.5,
                max_distance_m=max_distance_m,
            ),
        ]
        baseline = routes[0]
        rows = []
        for route in routes:
            risk_reduction = 0.0
            if baseline["total_cmcs"] > 0:
                risk_reduction = (
                    baseline["total_cmcs"] - route["total_cmcs"]
                ) / baseline["total_cmcs"] * 100.0
            rows.append(
                {
                    "mode": route["mode"],
                    "path": route["path"],
                    "edge_path": route["edge_path"],
                    "total_distance_m": route["total_distance_m"],
                    "total_cmcs": route["total_cmcs"],
                    "average_cmcs": route["average_cmcs"],
                    "objective_cost": route["objective_cost"],
                    "extra_distance_m": round(
                        route["total_distance_m"] - baseline["total_distance_m"], 2
                    ),
                    "cmcs_reduction_pct": round(risk_reduction, 2),
                    "num_segments": route["num_segments"],
                    "algorithm": route["algorithm"],
                    "selected_source": route["selected_source"],
                    "optimality_proven": route["optimality_proven"],
                    "optimality_claim_scope": route["optimality_claim_scope"],
                    "aco_found_feasible": route["aco_found_feasible"],
                    "aco_objective": route["aco_objective"],
                    "aco_total_distance": route["aco_total_distance"],
                    "aco_total_cmcs": route["aco_total_cmcs"],
                    "aco_feasible_solutions": route["aco_feasible_solutions"],
                    "pure_aco_feasible_solutions": route[
                        "pure_aco_feasible_solutions"
                    ],
                    "seeded_feasible_solutions": route[
                        "seeded_feasible_solutions"
                    ],
                    "rcsp_objective": route["rcsp_objective"],
                    "rcsp_total_distance": route["rcsp_total_distance"],
                    "rcsp_total_cmcs": route["rcsp_total_cmcs"],
                    "rcsp_used_aco_upper_bound": route[
                        "rcsp_used_aco_upper_bound"
                    ],
                    "initial_upper_bound_source": route[
                        "initial_upper_bound_source"
                    ],
                    "detour_ratio": route["detour_ratio"],
                    "detour_constraint_satisfied": route[
                        "detour_constraint_satisfied"
                    ],
                    "risk_reduction_pct_against_shortest": route[
                        "risk_reduction_pct_against_shortest"
                    ],
                    "distance_increase_pct_against_shortest": route[
                        "distance_increase_pct_against_shortest"
                    ],
                    "gap_pct": route["gap_pct"],
                    "runtime_ms": route["search_stats"]["runtime_ms"],
                    "aco_runtime_ms": route["search_stats"]["aco_runtime_ms"],
                    "rcsp_runtime_ms": route["search_stats"]["rcsp_runtime_ms"],
                    "ants_total": route["search_stats"]["ants_total"],
                    "labels_created": route["search_stats"]["labels_created"],
                    "labels_expanded": route["search_stats"]["labels_expanded"],
                    "dominance_prunes": route["search_stats"][
                        "dominance_prunes"
                    ],
                    "resource_prunes": route["search_stats"][
                        "resource_prunes"
                    ],
                    "upper_bound_prunes": route["search_stats"][
                        "upper_bound_prunes"
                    ],
                    "search_stats": route["search_stats"],
                    "aco_stats": route["aco_stats"],
                    "rcsp_stats": route["rcsp_stats"],
                }
            )
        return pd.DataFrame(rows)

    def aggregate_avoided_segments(
        self,
        origins: Sequence[Coordinate | object],
        destinations: Sequence[Coordinate | object],
        save_path: str | Path | None = None,
    ) -> pd.DataFrame:
        counts: dict[str, int] = {}
        for origin in origins:
            for destination in destinations:
                try:
                    shortest = set(self.path_edge_ids(self.shortest_route(origin, destination)))
                    safest = set(self.path_edge_ids(self.safest_route(origin, destination)))
                except (nx.NetworkXException, ValueError):
                    continue
                for edge_id in shortest - safest:
                    counts[edge_id] = counts.get(edge_id, 0) + 1
        result = pd.DataFrame(
            [{"edge_id": edge_id, "avoidance_count": count} for edge_id, count in counts.items()]
        )
        if not result.empty:
            result = result.sort_values("avoidance_count", ascending=False)
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)
        return result

