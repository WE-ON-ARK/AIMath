"""순수 Pulse 기반 단일 목적·거리 자원 제약 경로 탐색.

외부 최단경로 구현이나 별도 휴리스틱 탐색 전처리를 사용하지 않는다.
목적함수 비용과 거리 자원을 누적하며, 기하학적 하한·dominance·cycle
검사로 가지치기하는 depth-first Pulse 알고리즘이다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from time import perf_counter
from typing import Callable, Iterator

import networkx as nx


EdgeRef = tuple[object, object, object]


@dataclass
class PulseSearchStats:
    algorithm: str = "pulse"
    optimality_proven: bool = False
    runtime_ms: float = 0.0
    pulses_generated: int = 0
    states_expanded: int = 0
    edges_considered: int = 0
    feasible_solutions: int = 0
    incumbent_updates: int = 0
    bound_prunes: int = 0
    resource_prunes: int = 0
    dominance_prunes: int = 0
    cycle_prunes: int = 0
    dead_end_prunes: int = 0
    max_stack_size: int = 0
    max_labels_at_node: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PulseSearchResult:
    path: list[object]
    edge_path: list[EdgeRef]
    objective_cost: float
    total_distance: float
    stats: PulseSearchStats


@dataclass(frozen=True)
class _PulseState:
    node: object
    objective: float
    distance: float
    path: tuple[object, ...]
    edge_path: tuple[EdgeRef, ...]
    visited: frozenset[object]


class PulseAlgorithm:
    """비음수 간선 비용 그래프에서 정확한 Pulse 탐색을 수행한다."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._distance_scale = self._coordinate_distance_scale()

    def solve(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        *,
        resource_attribute: str = "length",
        max_resource: float | None = None,
    ) -> PulseSearchResult:
        if origin not in self.graph:
            raise nx.NodeNotFound(f"그래프에 출발 노드 {origin!r}가 없습니다.")
        if destination not in self.graph:
            raise nx.NodeNotFound(f"그래프에 도착 노드 {destination!r}가 없습니다.")
        if origin == destination:
            stats = PulseSearchStats(optimality_proven=True)
            return PulseSearchResult([origin], [], 0.0, 0.0, stats)

        minimum_objective_ratio = self._minimum_cost_ratio(
            objective_attribute,
            resource_attribute,
        )
        started = perf_counter()
        stats = PulseSearchStats(pulses_generated=1, max_stack_size=1)
        incumbent_objective = inf
        incumbent_distance = inf
        incumbent_path: tuple[object, ...] | None = None
        incumbent_edges: tuple[EdgeRef, ...] | None = None
        labels: dict[object, list[tuple[float, float]]] = {}
        stack = [
            _PulseState(
                node=origin,
                objective=0.0,
                distance=0.0,
                path=(origin,),
                edge_path=(),
                visited=frozenset((origin,)),
            )
        ]

        while stack:
            stats.max_stack_size = max(stats.max_stack_size, len(stack))
            state = stack.pop()

            remaining_distance = self._distance_lower_bound(
                state.node,
                destination,
            )
            if (
                max_resource is not None
                and state.distance + remaining_distance > max_resource + 1e-9
            ):
                stats.resource_prunes += 1
                continue

            objective_lower_bound = (
                state.objective
                + remaining_distance * minimum_objective_ratio
            )
            if objective_lower_bound > incumbent_objective + 1e-12:
                stats.bound_prunes += 1
                continue

            if self._is_dominated(
                labels.setdefault(state.node, []),
                state.distance,
                state.objective,
            ):
                stats.dominance_prunes += 1
                continue
            self._record_label(
                labels[state.node],
                state.distance,
                state.objective,
            )
            stats.max_labels_at_node = max(
                stats.max_labels_at_node,
                len(labels[state.node]),
            )

            if state.node == destination:
                stats.feasible_solutions += 1
                if self._is_better_solution(
                    state.objective,
                    state.distance,
                    incumbent_objective,
                    incumbent_distance,
                ):
                    incumbent_objective = state.objective
                    incumbent_distance = state.distance
                    incumbent_path = state.path
                    incumbent_edges = state.edge_path
                    stats.incumbent_updates += 1
                continue

            stats.states_expanded += 1
            candidates: list[
                tuple[tuple[float, float, str, str], _PulseState]
            ] = []
            for edge, next_node, data in self._outgoing_edges(state.node):
                stats.edges_considered += 1
                if next_node in state.visited:
                    stats.cycle_prunes += 1
                    continue

                edge_distance = self._edge_value(
                    data,
                    resource_attribute,
                )
                edge_objective = self._edge_value(
                    data,
                    objective_attribute,
                )
                next_distance = state.distance + edge_distance
                if (
                    max_resource is not None
                    and next_distance > max_resource + 1e-9
                ):
                    stats.resource_prunes += 1
                    continue

                next_objective = state.objective + edge_objective
                if next_objective > incumbent_objective + 1e-12:
                    stats.bound_prunes += 1
                    continue

                next_remaining = self._distance_lower_bound(
                    next_node,
                    destination,
                )
                priority = (
                    next_objective
                    + next_remaining * minimum_objective_ratio,
                    next_distance + next_remaining,
                    str(next_node),
                    str(edge[2]),
                )
                candidates.append(
                    (
                        priority,
                        _PulseState(
                            node=next_node,
                            objective=next_objective,
                            distance=next_distance,
                            path=state.path + (next_node,),
                            edge_path=state.edge_path + (edge,),
                            visited=state.visited | frozenset((next_node,)),
                        ),
                    )
                )

            if not candidates:
                stats.dead_end_prunes += 1
                continue

            # 스택은 LIFO이므로 우선순위가 낮은 Pulse가 먼저 나오도록 역순 삽입한다.
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, candidate in candidates:
                stack.append(candidate)
                stats.pulses_generated += 1

        stats.runtime_ms = round((perf_counter() - started) * 1000.0, 4)
        stats.optimality_proven = incumbent_path is not None
        if incumbent_path is None or incumbent_edges is None:
            raise nx.NetworkXNoPath(
                f"{origin!r}에서 {destination!r}까지 Pulse 경로가 없습니다."
            )
        return PulseSearchResult(
            path=list(incumbent_path),
            edge_path=list(incumbent_edges),
            objective_cost=float(incumbent_objective),
            total_distance=float(incumbent_distance),
            stats=stats,
        )

    @staticmethod
    def _edge_value(data: dict, attribute: str) -> float:
        value = float(data.get(attribute, inf))
        if value < 0:
            raise ValueError(
                f"Pulse 알고리즘은 비음수 비용만 지원합니다: {attribute}={value}"
            )
        return value

    @staticmethod
    def _is_better_solution(
        objective: float,
        distance: float,
        incumbent_objective: float,
        incumbent_distance: float,
    ) -> bool:
        return (
            objective < incumbent_objective - 1e-12
            or (
                abs(objective - incumbent_objective) <= 1e-12
                and distance < incumbent_distance - 1e-9
            )
        )

    @staticmethod
    def _is_dominated(
        labels: list[tuple[float, float]],
        distance: float,
        objective: float,
    ) -> bool:
        return any(
            known_distance <= distance + 1e-9
            and known_objective <= objective + 1e-12
            for known_distance, known_objective in labels
        )

    @staticmethod
    def _record_label(
        labels: list[tuple[float, float]],
        distance: float,
        objective: float,
    ) -> None:
        labels[:] = [
            (known_distance, known_objective)
            for known_distance, known_objective in labels
            if not (
                distance <= known_distance + 1e-9
                and objective <= known_objective + 1e-12
            )
        ]
        labels.append((distance, objective))

    def _outgoing_edges(
        self,
        node: object,
    ) -> Iterator[tuple[EdgeRef, object, dict]]:
        if self.graph.is_multigraph():
            if self.graph.is_directed():
                for _, v, key, data in self.graph.out_edges(
                    node,
                    keys=True,
                    data=True,
                ):
                    yield (node, v, key), v, data
            else:
                for u, v, key, data in self.graph.edges(
                    node,
                    keys=True,
                    data=True,
                ):
                    next_node = v if u == node else u
                    yield (node, next_node, key), next_node, data
            return

        if self.graph.is_directed():
            for _, v, data in self.graph.out_edges(node, data=True):
                yield (node, v, 0), v, data
        else:
            for u, v, data in self.graph.edges(node, data=True):
                next_node = v if u == node else u
                yield (node, next_node, 0), next_node, data

    def _minimum_cost_ratio(
        self,
        objective_attribute: str,
        resource_attribute: str,
    ) -> float:
        ratios = []
        for _, _, _, data in self._all_edges():
            resource = self._edge_value(data, resource_attribute)
            objective = self._edge_value(data, objective_attribute)
            if resource > 0:
                ratios.append(objective / resource)
        return min(ratios, default=0.0)

    def _coordinate_distance_scale(self) -> float:
        """좌표 직선거리를 실제 간선 거리의 안전한 하한으로 변환한다."""
        ratios = []
        for u, v, _, data in self._all_edges():
            coordinate_distance = self._coordinate_distance(u, v)
            if coordinate_distance <= 0:
                continue
            length = float(data.get("length", 0.0))
            if length > 0:
                ratios.append(length / coordinate_distance)
        return min(ratios, default=0.0)

    def _distance_lower_bound(self, node: object, destination: object) -> float:
        return self._coordinate_distance(node, destination) * self._distance_scale

    def _coordinate_distance(self, left: object, right: object) -> float:
        left_data = self.graph.nodes[left]
        right_data = self.graph.nodes[right]
        if not {"x", "y"}.issubset(left_data) or not {"x", "y"}.issubset(
            right_data
        ):
            return 0.0
        dx = float(left_data["x"]) - float(right_data["x"])
        dy = float(left_data["y"]) - float(right_data["y"])
        return (dx * dx + dy * dy) ** 0.5

    def _all_edges(self) -> Iterator[tuple[object, object, object, dict]]:
        if self.graph.is_multigraph():
            yield from self.graph.edges(keys=True, data=True)
        else:
            for u, v, data in self.graph.edges(data=True):
                yield u, v, 0, data
