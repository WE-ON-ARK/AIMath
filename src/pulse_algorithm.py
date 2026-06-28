"""Pulse 기반 단일/양방향 자원 제약 경로 탐색.

단방향 Pulse: 출발지에서 깊이 우선 탐색으로 도착지까지 탐색.
양방향 Pulse: 출발지·도착지 양쪽에서 동시에 탐색 후 만남점에서 경로 결합.
  - 전진 탐색: 원본 그래프에서 origin → node
  - 후진 탐색: 역방향 그래프(directed reverse / undirected 동일)에서 destination → node
  - 만남점 m에서 (origin→m) + (m→destination) 결합
  - 공유 상한값(incumbent)으로 양방향 동시 가지치기
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from time import perf_counter
from typing import Callable, Iterator

import networkx as nx


EdgeRef = tuple[object, object, object]


# ---------------------------------------------------------------------------
# 통계 데이터클래스
# ---------------------------------------------------------------------------

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


@dataclass
class BiPulseSearchStats:
    """양방향 Pulse 탐색 통계.

    기존 코드와의 호환성을 위해 단방향 필드(pulses_generated, states_expanded 등)를
    전진+후진 합산값으로 유지한다.
    """
    algorithm: str = "bidirectional_pulse"
    optimality_proven: bool = False
    runtime_ms: float = 0.0
    # 합산값 (기존 코드 호환)
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
    # 양방향 전용 필드
    forward_pulses: int = 0
    backward_pulses: int = 0
    forward_states_expanded: int = 0
    backward_states_expanded: int = 0
    forward_edges_considered: int = 0
    backward_edges_considered: int = 0
    meeting_points_found: int = 0
    max_forward_stack_size: int = 0
    max_backward_stack_size: int = 0
    # 그리디 시딩(초기 상한값 확보) 통계
    seed_succeeded: bool = False
    seed_states_expanded: int = 0
    seed_objective: float = float("inf")
    budget_exceeded: bool = False
    loop_iterations: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 결과 및 내부 상태
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PulseSearchResult:
    path: list[object]
    edge_path: list[EdgeRef]
    objective_cost: float
    total_distance: float
    stats: object  # PulseSearchStats | BiPulseSearchStats


@dataclass(frozen=True)
class _PulseState:
    node: object
    objective: float
    distance: float
    path: tuple[object, ...]
    edge_path: tuple[EdgeRef, ...]
    visited: frozenset[object]


class _BiState:
    """양방향 Pulse용 경량 상태.

    경로 전체를 튜플로 누적하지 않고 부모 포인터만 보관해 메모리·복사 비용을
    O(경로길이)에서 O(1)로 줄인다. 단순경로(visited) 검사를 위해 frozenset만
    누적하며, 실제 경로·간선열은 만남 시점에 부모 체인을 거슬러 재구성한다.
    """

    __slots__ = ("node", "objective", "distance", "visited", "parent", "edge")

    def __init__(
        self,
        node: object,
        objective: float,
        distance: float,
        visited: frozenset,
        parent: "_BiState | None",
        edge: EdgeRef | None,
    ):
        self.node = node
        self.objective = objective
        self.distance = distance
        self.visited = visited
        self.parent = parent
        self.edge = edge  # parent → node 간선 (탐색 그래프 방향 기준)


# ---------------------------------------------------------------------------
# 단방향 Pulse 알고리즘 (기존 구현 유지)
# ---------------------------------------------------------------------------

class PulseAlgorithm:
    """비음수 간선 비용 그래프에서 정확한 단방향 Pulse 탐색을 수행한다."""

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
        time_budget_s: float | None = None,
    ) -> PulseSearchResult:
        # time_budget_s는 양방향 구현과의 API 호환을 위해 받지만 단방향에서는
        # 사용하지 않는다(파이프라인 기본 경로 엔진은 양방향이다).
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

            if _is_dominated(
                labels.setdefault(state.node, []),
                state.distance,
                state.objective,
            ):
                stats.dominance_prunes += 1
                continue
            _record_label(
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
                if _is_better_solution(
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
            for edge, next_node, data in _outgoing_edges(state.node, self.graph):
                stats.edges_considered += 1
                if next_node in state.visited:
                    stats.cycle_prunes += 1
                    continue

                edge_distance = _edge_value(data, resource_attribute)
                edge_objective = _edge_value(data, objective_attribute)
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

    def _minimum_cost_ratio(
        self,
        objective_attribute: str,
        resource_attribute: str,
    ) -> float:
        ratios = []
        for _, _, _, data in _all_edges(self.graph):
            resource = _edge_value(data, resource_attribute)
            objective = _edge_value(data, objective_attribute)
            if resource > 0:
                ratios.append(objective / resource)
        return min(ratios, default=0.0)

    def _coordinate_distance_scale(self) -> float:
        return _coordinate_distance_scale(self.graph)

    def _distance_lower_bound(self, node: object, target: object) -> float:
        return _coordinate_distance(self.graph, node, target) * self._distance_scale


# ---------------------------------------------------------------------------
# 양방향 Pulse 알고리즘
# ---------------------------------------------------------------------------

class BidirectionalPulseAlgorithm:
    """양방향 Pulse 탐색.

    전진 탐색(origin→)과 후진 탐색(destination→, 역방향 그래프)을 교대로 수행하고
    만남점에서 두 부분 경로를 결합해 최적 완성 경로를 구한다.

    유향 그래프: G.reverse()로 역방향 그래프 생성.
    무향 그래프: 원본과 동일한 그래프를 후진 탐색에 사용.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.reversed_graph = (
            graph.reverse(copy=True) if graph.is_directed() else graph
        )
        self._distance_scale = _coordinate_distance_scale(graph)

    def _greedy_seed(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        resource_attribute: str,
        max_resource: float | None,
        min_ratio: float,
    ) -> tuple[float, float, tuple, tuple, int] | None:
        """거리 기준 그리디 다이브로 첫 feasible 경로를 빠르게 찾아 초기 상한값을 만든다.

        핵심: 정렬 기준을 **목적함수가 아니라 목적지까지의 기하 거리**로 둔다.
        목적함수(예: safety_cost)는 하한 계수가 약해(min_ratio≈0.05) 목적함수로
        정렬하면 목적지를 향하지 못하고 헤매다 폭발한다. 거리 기준으로 다이브하면
        어떤 목적함수든 목적지에 빠르게 도달한다. 찾은 경로의 실제 목적함수 누적값이
        본 양방향 탐색의 유효한 초기 incumbent(상한)가 된다.

        거리 기준 dominance(노드별 최소 도달 거리)로 같은 노드의 중복 확장을 막아
        O(V+E)로 종료를 보장한다. 외부 최단경로(다익스트라 등)는 쓰지 않는다.

        반환: (objective, distance, node_path, edge_path, states_expanded) 또는
        경로가 없으면 None.
        """
        graph = self.graph
        scale = self._distance_scale
        coord = _coordinate_distance
        states_expanded = 0
        # 노드별 최소 도달 거리(거리 기준 dominance)
        best_distance: dict[object, float] = {origin: 0.0}
        stack = [_BiState(origin, 0.0, 0.0, frozenset({origin}), None, None)]
        while stack:
            state = stack.pop()
            if state.node == destination:
                path, edges = _reconstruct_state(state)
                return state.objective, state.distance, path, edges, states_expanded
            # 이미 더 짧은 거리로 이 노드를 확장했다면 건너뛴다.
            recorded = best_distance.get(state.node)
            if recorded is not None and recorded < state.distance - 1e-9:
                continue
            states_expanded += 1
            children: list[tuple[tuple, _BiState]] = []
            for edge, next_node, data in _outgoing_edges(state.node, graph):
                if next_node in state.visited:
                    continue
                edge_dist = _edge_value(data, resource_attribute)
                next_dist = state.distance + edge_dist
                if max_resource is not None and next_dist > max_resource + 1e-9:
                    continue
                prev = best_distance.get(next_node)
                if prev is not None and prev <= next_dist + 1e-9:
                    continue  # 이미 같거나 더 짧게 도달 → 시드에서 가지치기
                best_distance[next_node] = next_dist
                edge_obj = _edge_value(data, objective_attribute)
                next_obj = state.objective + edge_obj
                nxt_rem = coord(graph, next_node, destination) * scale
                # 목적지까지의 추정 총거리로 정렬(목적함수 무관, 빠른 도달 유도)
                priority = (next_dist + nxt_rem, str(next_node), str(edge[2]))
                children.append(
                    (
                        priority,
                        _BiState(
                            next_node,
                            next_obj,
                            next_dist,
                            state.visited | frozenset({next_node}),
                            state,
                            edge,
                        ),
                    )
                )
            # 스택은 LIFO이므로 거리가 가까운(우선순위 작은) 자식이 먼저 pop되도록 역순 push
            children.sort(key=lambda x: x[0], reverse=True)
            for _, cand in children:
                stack.append(cand)
        return None

    def solve(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        *,
        resource_attribute: str = "length",
        max_resource: float | None = None,
        time_budget_s: float | None = None,
    ) -> PulseSearchResult:
        if origin not in self.graph:
            raise nx.NodeNotFound(f"그래프에 출발 노드 {origin!r}가 없습니다.")
        if destination not in self.graph:
            raise nx.NodeNotFound(f"그래프에 도착 노드 {destination!r}가 없습니다.")
        if origin == destination:
            stats = BiPulseSearchStats(optimality_proven=True)
            return PulseSearchResult([origin], [], 0.0, 0.0, stats)

        min_ratio = _minimum_cost_ratio(
            self.graph, objective_attribute, resource_attribute
        )
        scale = self._distance_scale
        graph = self.graph
        reversed_graph = self.reversed_graph
        coord = _coordinate_distance
        started = perf_counter()
        stats = BiPulseSearchStats(
            forward_pulses=1,
            backward_pulses=1,
            pulses_generated=2,
            max_forward_stack_size=1,
            max_backward_stack_size=1,
            max_stack_size=2,
        )

        incumbent_objective: float = inf
        incumbent_distance: float = inf
        incumbent_path: tuple[object, ...] | None = None
        incumbent_edges: tuple[EdgeRef, ...] | None = None

        # ── 그리디 시딩: 초기 상한값 확보(문제 #1 해결의 핵심) ──────────────
        # 양방향 탐색이 만남 전부터 가지치기할 수 있도록 feasible 경로 하나를
        # 먼저 빠르게 찾아 incumbent를 세운다.
        seed = self._greedy_seed(
            origin,
            destination,
            objective_attribute,
            resource_attribute,
            max_resource,
            min_ratio,
        )
        if seed is not None:
            seed_obj, seed_dist, seed_path, seed_edges, seed_states = seed
            incumbent_objective = seed_obj
            incumbent_distance = seed_dist
            incumbent_path = seed_path
            incumbent_edges = seed_edges
            stats.seed_succeeded = True
            stats.seed_states_expanded = seed_states
            stats.seed_objective = round(float(seed_obj), 6)

        # 지배 검사용 레이블: node → list[(dist, obj)]
        fwd_labels: dict[object, list[tuple[float, float]]] = {}
        bwd_labels: dict[object, list[tuple[float, float]]] = {}
        # 만남점 결합용 레코드(경량): node → list[(dist, obj, _BiState)]
        # 경로 전체가 아니라 상태 참조만 보관하고 만남 시점에 부모 체인으로 재구성한다.
        fwd_records: dict[object, list[tuple[float, float, _BiState]]] = {}
        bwd_records: dict[object, list[tuple[float, float, _BiState]]] = {}
        # 노드별 방향 최소 목적값 — 만남 검사 가지치기용.
        # (state.objective + 반대편 최소목적) ≥ incumbent면 그 노드의 만남은
        # 어떤 조합도 incumbent를 개선할 수 없으므로 전부 건너뛴다.
        fwd_best_obj: dict[object, float] = {}
        bwd_best_obj: dict[object, float] = {}

        fwd_stack = [
            _BiState(origin, 0.0, 0.0, frozenset({origin}), None, None)
        ]
        bwd_stack = [
            _BiState(destination, 0.0, 0.0, frozenset({destination}), None, None)
        ]

        def _try_meet(fwd_state: _BiState, bwd_state: _BiState) -> None:
            nonlocal incumbent_objective, incumbent_distance
            nonlocal incumbent_path, incumbent_edges
            comb_obj = fwd_state.objective + bwd_state.objective
            comb_dist = fwd_state.distance + bwd_state.distance
            if max_resource is not None and comb_dist > max_resource + 1e-9:
                return
            # 두 frontier가 feasible하게 만난 시점(상한 개선 여부와 무관하게 집계)
            stats.meeting_points_found += 1
            stats.feasible_solutions += 1
            if not _is_better_solution(
                comb_obj, comb_dist, incumbent_objective, incumbent_distance
            ):
                return
            fwd_path, fwd_edges = _reconstruct_state(fwd_state)
            bwd_path, bwd_edges = _reconstruct_state(bwd_state)
            incumbent_objective = comb_obj
            incumbent_distance = comb_dist
            incumbent_path, incumbent_edges = _combine_paths(
                graph, fwd_path, fwd_edges, bwd_path, bwd_edges
            )
            stats.incumbent_updates += 1

        def _step(forward: bool) -> None:
            """한 방향에서 스택 top 하나를 처리한다(교대 확장용)."""
            nonlocal incumbent_objective
            if forward:
                stack = fwd_stack
                labels = fwd_labels
                records = fwd_records
                opposite_records = bwd_records
                best_obj = fwd_best_obj
                opposite_best_obj = bwd_best_obj
                search_graph = graph
                heuristic_target = destination
                expand_until = destination
            else:
                stack = bwd_stack
                labels = bwd_labels
                records = bwd_records
                opposite_records = fwd_records
                best_obj = bwd_best_obj
                opposite_best_obj = fwd_best_obj
                search_graph = reversed_graph
                heuristic_target = origin
                expand_until = origin

            state = stack.pop()
            remaining = coord(graph, state.node, heuristic_target) * scale
            if (
                max_resource is not None
                and state.distance + remaining > max_resource + 1e-9
            ):
                stats.resource_prunes += 1
                return
            if state.objective + remaining * min_ratio > incumbent_objective + 1e-12:
                stats.bound_prunes += 1
                return
            node_labels = labels.setdefault(state.node, [])
            if _is_dominated(node_labels, state.distance, state.objective):
                stats.dominance_prunes += 1
                return
            _record_label(node_labels, state.distance, state.objective)
            stats.max_labels_at_node = max(
                stats.max_labels_at_node, len(node_labels)
            )
            records.setdefault(state.node, []).append(
                (state.distance, state.objective, state)
            )
            prev_best = best_obj.get(state.node)
            if prev_best is None or state.objective < prev_best:
                best_obj[state.node] = state.objective

            # 만남점 탐지: 반대 방향이 이미 이 노드에 도달했으면 결합 시도.
            # 단, 가장 좋은 반대편 경로와 합쳐도 incumbent를 못 넘으면 전부 건너뛴다
            # (만남 검사 폭발 방지의 핵심 가지치기).
            opp = opposite_records.get(state.node)
            if opp:
                opp_best = opposite_best_obj[state.node]
                if state.objective + opp_best <= incumbent_objective + 1e-12:
                    for _, _, other_state in opp:
                        if forward:
                            _try_meet(state, other_state)
                        else:
                            _try_meet(other_state, state)

            if state.node == expand_until:
                return

            if forward:
                stats.forward_states_expanded += 1
            else:
                stats.backward_states_expanded += 1
            stats.states_expanded += 1

            candidates: list[tuple[tuple, _BiState]] = []
            for edge, next_node, data in _outgoing_edges(state.node, search_graph):
                if forward:
                    stats.forward_edges_considered += 1
                else:
                    stats.backward_edges_considered += 1
                stats.edges_considered += 1
                if next_node in state.visited:
                    stats.cycle_prunes += 1
                    continue
                edge_dist = _edge_value(data, resource_attribute)
                edge_obj = _edge_value(data, objective_attribute)
                next_dist = state.distance + edge_dist
                if max_resource is not None and next_dist > max_resource + 1e-9:
                    stats.resource_prunes += 1
                    continue
                next_obj = state.objective + edge_obj
                if next_obj > incumbent_objective + 1e-12:
                    stats.bound_prunes += 1
                    continue
                nxt_rem = coord(graph, next_node, heuristic_target) * scale
                priority = (
                    next_obj + nxt_rem * min_ratio,
                    next_dist + nxt_rem,
                    str(next_node),
                    str(edge[2]),
                )
                candidates.append(
                    (
                        priority,
                        _BiState(
                            next_node,
                            next_obj,
                            next_dist,
                            state.visited | frozenset({next_node}),
                            state,
                            edge,
                        ),
                    )
                )

            if not candidates:
                stats.dead_end_prunes += 1
                return
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, cand in candidates:
                stack.append(cand)
                if forward:
                    stats.forward_pulses += 1
                else:
                    stats.backward_pulses += 1
                stats.pulses_generated += 1

        deadline = (started + time_budget_s) if time_budget_s is not None else None
        budget_check_interval = 2048
        while fwd_stack or bwd_stack:
            stats.loop_iterations += 1
            stats.max_forward_stack_size = max(
                stats.max_forward_stack_size, len(fwd_stack)
            )
            stats.max_backward_stack_size = max(
                stats.max_backward_stack_size, len(bwd_stack)
            )
            stats.max_stack_size = max(
                stats.max_stack_size, len(fwd_stack) + len(bwd_stack)
            )
            # 시간 예산(anytime): 초과 시 현재까지의 최적 incumbent로 종료한다.
            if (
                deadline is not None
                and stats.loop_iterations % budget_check_interval == 0
                and perf_counter() > deadline
            ):
                stats.budget_exceeded = True
                break
            # 교대 확장: 더 작은 frontier를 우선 확장(Pohl의 cardinality 비교).
            # 한쪽 탐색이 폭발적으로 커지는 것을 막아 양방향 이득을 유지한다.
            if fwd_stack and (not bwd_stack or len(fwd_stack) <= len(bwd_stack)):
                _step(forward=True)
            else:
                _step(forward=False)

        stats.runtime_ms = round((perf_counter() - started) * 1000.0, 4)
        # 예산 내 자연 종료 시에만 최적성 증명. 예산 초과 종료는 incumbent가
        # 최적이라는 보장이 없다(시딩 경로일 수 있음).
        stats.optimality_proven = (
            incumbent_path is not None and not stats.budget_exceeded
        )
        if incumbent_path is None or incumbent_edges is None:
            raise nx.NetworkXNoPath(
                f"{origin!r}에서 {destination!r}까지 양방향 Pulse 경로가 없습니다."
            )
        return PulseSearchResult(
            path=list(incumbent_path),
            edge_path=list(incumbent_edges),
            objective_cost=float(incumbent_objective),
            total_distance=float(incumbent_distance),
            stats=stats,
        )


# ---------------------------------------------------------------------------
# 공유 유틸리티 함수 (모듈 수준 — 단방향·양방향 모두 사용)
# ---------------------------------------------------------------------------

def _edge_value(data: dict, attribute: str) -> float:
    value = float(data.get(attribute, inf))
    if value < 0:
        raise ValueError(
            f"Pulse 알고리즘은 비음수 비용만 지원합니다: {attribute}={value}"
        )
    return value


def _is_better_solution(
    objective: float,
    distance: float,
    incumbent_objective: float,
    incumbent_distance: float,
) -> bool:
    return objective < incumbent_objective - 1e-12 or (
        abs(objective - incumbent_objective) <= 1e-12
        and distance < incumbent_distance - 1e-9
    )


def _is_dominated(
    labels: list[tuple[float, float]],
    distance: float,
    objective: float,
) -> bool:
    return any(
        known_distance <= distance + 1e-9 and known_objective <= objective + 1e-12
        for known_distance, known_objective in labels
    )


def _record_label(
    labels: list[tuple[float, float]],
    distance: float,
    objective: float,
) -> None:
    labels[:] = [
        (kd, ko)
        for kd, ko in labels
        if not (distance <= kd + 1e-9 and objective <= ko + 1e-12)
    ]
    labels.append((distance, objective))


def _outgoing_edges(
    node: object,
    graph: nx.Graph,
) -> Iterator[tuple[EdgeRef, object, dict]]:
    if graph.is_multigraph():
        if graph.is_directed():
            for _, v, key, data in graph.out_edges(node, keys=True, data=True):
                yield (node, v, key), v, data
        else:
            for u, v, key, data in graph.edges(node, keys=True, data=True):
                next_node = v if u == node else u
                yield (node, next_node, key), next_node, data
        return
    if graph.is_directed():
        for _, v, data in graph.out_edges(node, data=True):
            yield (node, v, 0), v, data
    else:
        for u, v, data in graph.edges(node, data=True):
            next_node = v if u == node else u
            yield (node, next_node, 0), next_node, data


def _all_edges(
    graph: nx.Graph,
) -> Iterator[tuple[object, object, object, dict]]:
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, 0, data


def _minimum_cost_ratio(
    graph: nx.Graph,
    objective_attribute: str,
    resource_attribute: str,
) -> float:
    ratios = []
    for _, _, _, data in _all_edges(graph):
        resource = _edge_value(data, resource_attribute)
        objective = _edge_value(data, objective_attribute)
        if resource > 0:
            ratios.append(objective / resource)
    return min(ratios, default=0.0)


def _coordinate_distance_scale(graph: nx.Graph) -> float:
    """좌표 직선거리를 실제 간선 거리의 안전한 하한으로 변환하는 배율."""
    ratios = []
    for u, v, _, data in _all_edges(graph):
        coord_dist = _coordinate_distance(graph, u, v)
        if coord_dist <= 0:
            continue
        length = float(data.get("length", 0.0))
        if length > 0:
            ratios.append(length / coord_dist)
    return min(ratios, default=0.0)


def _coordinate_distance(graph: nx.Graph, left: object, right: object) -> float:
    left_data = graph.nodes[left]
    right_data = graph.nodes[right]
    if not {"x", "y"}.issubset(left_data) or not {"x", "y"}.issubset(right_data):
        return 0.0
    dx = float(left_data["x"]) - float(right_data["x"])
    dy = float(left_data["y"]) - float(right_data["y"])
    return (dx * dx + dy * dy) ** 0.5


def _reconstruct_state(state: "_BiState") -> tuple[tuple, tuple]:
    """부모 포인터를 거슬러 (node_path, edge_path)를 재구성한다.

    반환 순서는 루트(시작점)→state 방향이다.
    전진 상태: (origin, ..., node) / 간선은 원본 그래프 방향.
    후진 상태: (destination, ..., node) / 간선은 역방향 그래프 방향.
    """
    nodes: list[object] = []
    edges: list[EdgeRef] = []
    cur: "_BiState | None" = state
    while cur is not None:
        nodes.append(cur.node)
        if cur.edge is not None:
            edges.append(cur.edge)
        cur = cur.parent
    nodes.reverse()
    edges.reverse()
    return tuple(nodes), tuple(edges)


def _combine_paths(
    graph: nx.Graph,
    fwd_path: tuple,
    fwd_edges: tuple,
    bwd_path: tuple,
    bwd_edges: tuple,
) -> tuple[tuple, tuple]:
    """전진 경로와 후진 경로를 만남점에서 결합한다.

    fwd_path: (origin, ..., m) — 원본 그래프 방향
    bwd_path: (destination, ..., m) — 역방향 그래프 탐색으로 저장된 순서
    bwd_edges: 역방향 그래프의 간선 refs [(destination, w1, k1), ..., (w_{n-1}, m, k_n)]

    결합 결과: (origin, ..., m, ..., destination)
    간선: fwd_edges + (역순·방향 변환된 bwd_edges)

    유향 그래프: 역방향 그래프 간선 (u_rev, v_rev, k) → 원본 간선 (v_rev, u_rev, k)
    무향 그래프: 간선 u↔v이므로 방향 변환 없이 순서만 역전
    """
    # 후진 경로를 역순으로 뒤집어 m → destination 순서로 만들기
    rev_bwd_path = tuple(reversed(bwd_path))  # (m, ..., destination)
    # meeting_node(m)은 fwd_path 마지막 원소이므로 중복 제거
    combined_path = fwd_path + rev_bwd_path[1:]

    # 후진 간선을 원본 그래프 방향으로 변환 후 역순 적용
    if graph.is_directed():
        # 역방향 그래프 간선 (u_rev, v_rev, key) → 원본 (v_rev, u_rev, key)
        original_bwd_edges: tuple[EdgeRef, ...] = tuple(
            (v, u, key) for (u, v, key) in reversed(bwd_edges)
        )
    else:
        # 무향 그래프: 역방향 그래프 = 원본, 방향 변환 불필요
        # 단, 경로 방향에 맞게 순서만 역전
        original_bwd_edges = tuple(
            (v, u, key) for (u, v, key) in reversed(bwd_edges)
        )

    combined_edges = fwd_edges + original_bwd_edges
    return combined_path, combined_edges
