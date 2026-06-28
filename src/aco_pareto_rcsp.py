from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from math import inf, isfinite
from random import Random
from statistics import mean, pstdev
from time import perf_counter
from typing import Iterable, Iterator, Sequence

import networkx as nx


EdgeRef = tuple[object, object, object]
EPS = 1e-9


@dataclass(frozen=True)
class AntColonyResult:
    path: list[object]
    edge_path: list[EdgeRef]
    objective_cost: float
    total_distance: float
    total_cmcs: float
    feasible: bool
    runtime_ms: float
    iterations_completed: int
    ants_total: int
    feasible_solutions: int
    pure_aco_feasible_solutions: int
    seeded_feasible_solutions: int
    fallback_feasible_solutions: int
    aco_found_feasible: bool
    fallback_used: bool
    best_iteration: int | None
    random_state: int | None
    run_history: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RCSPResult:
    path: list[object]
    edge_path: list[EdgeRef]
    objective_cost: float
    total_distance: float
    total_cmcs: float
    feasible: bool
    optimality_proven: bool
    timeout: bool
    runtime_ms: float
    labels_created: int
    labels_expanded: int
    dominance_prunes: int
    resource_prunes: int
    upper_bound_prunes: int
    feasible_solutions: int
    max_labels_at_node: int
    incumbent_updates: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HybridRouteResult:
    path: list[object]
    edge_path: list[EdgeRef]
    objective_cost: float
    total_distance: float
    total_cmcs: float
    feasible: bool
    selected_source: str
    optimality_proven: bool
    aco_objective: float | None
    aco_total_distance: float | None
    aco_total_cmcs: float | None
    aco_found_feasible: bool
    aco_feasible_solutions: int
    pure_aco_feasible_solutions: int
    seeded_feasible_solutions: int
    rcsp_objective: float | None
    rcsp_total_distance: float | None
    rcsp_total_cmcs: float | None
    rcsp_used_aco_upper_bound: bool
    initial_upper_bound_source: str
    optimality_claim_scope: str
    gap_pct: float | None
    aco_runtime_ms: float
    rcsp_runtime_ms: float
    total_runtime_ms: float
    detour_ratio: float
    detour_constraint_satisfied: bool
    risk_reduction_pct_against_shortest: float | None
    distance_increase_pct_against_shortest: float | None
    search_stats: dict[str, object]
    aco_stats: dict[str, object]
    rcsp_stats: dict[str, object]


@dataclass(frozen=True)
class _CandidatePath:
    path: tuple[object, ...]
    edge_path: tuple[EdgeRef, ...]
    objective: float
    distance: float
    cmcs: float
    feasible: bool = True
    source: str = "pure_aco"


@dataclass(frozen=True)
class _Label:
    node: object
    distance: float
    objective: float
    path: tuple[object, ...]
    edge_path: tuple[EdgeRef, ...]


class AntColonyRouter:
    """Ant Colony Optimization candidate search for constrained routes."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        n_ants: int = 32,
        n_iterations: int = 40,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.30,
        q: float = 100.0,
        random_state: int | None = 42,
        time_budget_s: float | None = None,
    ):
        self.graph = graph
        self.n_ants = int(max(1, n_ants))
        self.n_iterations = int(max(1, n_iterations))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation_rate = float(min(max(evaporation_rate, 0.0), 0.95))
        self.q = float(q)
        self.random_state = random_state
        self.random = Random(random_state)
        self.time_budget_s = time_budget_s

    def solve(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        *,
        resource_attribute: str = "length",
        max_resource: float | None = None,
        time_budget_s: float | None = None,
    ) -> AntColonyResult:
        _validate_nodes(self.graph, origin, destination)
        started = perf_counter()
        if origin == destination:
            return AntColonyResult(
                path=[origin],
                edge_path=[],
                objective_cost=0.0,
                total_distance=0.0,
                total_cmcs=0.0,
                feasible=True,
                runtime_ms=0.0,
                iterations_completed=0,
                ants_total=0,
                feasible_solutions=1,
                pure_aco_feasible_solutions=1,
                seeded_feasible_solutions=0,
                fallback_feasible_solutions=0,
                aco_found_feasible=True,
                fallback_used=False,
                best_iteration=0,
                random_state=self.random_state,
                run_history=[],
            )

        budget = self.time_budget_s if time_budget_s is None else time_budget_s
        deadline = started + budget if budget is not None else None
        pheromones = {
            _pheromone_key(self.graph, edge): 2.0
            for edge, _, _ in _all_directed_edge_choices(self.graph)
        }
        reachability = _length_to_destination(
            self.graph, destination, resource_attribute
        )
        connector_paths = _length_paths_to_destination(
            self.graph, destination, resource_attribute
        )

        best: _CandidatePath | None = None
        best_iteration: int | None = None
        pure_aco_feasible_solutions = 0
        seeded_feasible_solutions = 0
        fallback_feasible_solutions = 0
        ants_total = 0
        history: list[dict[str, object]] = []
        iterations_completed = 0
        fallback_used = False

        for iteration in range(1, self.n_iterations + 1):
            if _deadline_passed(deadline):
                break
            iteration_solutions: list[_CandidatePath] = []
            for _ in range(self.n_ants):
                if _deadline_passed(deadline):
                    break
                ants_total += 1
                candidate = self._construct_ant_path(
                    origin,
                    destination,
                    objective_attribute,
                    resource_attribute,
                    max_resource,
                    pheromones,
                    reachability,
                    connector_paths,
                )
                if candidate is not None and candidate.feasible:
                    iteration_solutions.append(candidate)

            iterations_completed = iteration
            iteration_pure = sum(
                1 for item in iteration_solutions if item.source == "pure_aco"
            )
            iteration_seeded = sum(
                1 for item in iteration_solutions if item.source != "pure_aco"
            )
            pure_aco_feasible_solutions += iteration_pure
            seeded_feasible_solutions += iteration_seeded
            if iteration_solutions:
                iteration_best = min(
                    iteration_solutions,
                    key=lambda item: (item.objective, item.distance),
                )
                if best is None or _better_candidate(iteration_best, best):
                    best = iteration_best
                    best_iteration = iteration

            self._evaporate(pheromones)
            self._deposit_pheromones(pheromones, iteration_solutions, best)

            objectives = [item.objective for item in iteration_solutions]
            history.append(
                {
                    "iteration": iteration,
                    "best_objective": _finite_or_none(
                        best.objective if best is not None else inf
                    ),
                    "best_distance": _finite_or_none(
                        best.distance if best is not None else inf
                    ),
                    "best_cmcs": _finite_or_none(
                        best.cmcs if best is not None else inf
                    ),
                    "feasible_ants": len(iteration_solutions),
                    "pure_aco_feasible_ants": iteration_pure,
                    "seeded_feasible_ants": iteration_seeded,
                    "best_source": best.source if best is not None else None,
                    "success_rate": round(
                        len(iteration_solutions) / max(1, self.n_ants), 6
                    ),
                    "mean_objective": round(mean(objectives), 6)
                    if objectives
                    else None,
                    "std_objective": round(pstdev(objectives), 6)
                    if len(objectives) > 1
                    else 0.0
                    if objectives
                    else None,
                }
            )

        if best is None:
            best = _dijkstra_candidate(
                self.graph,
                origin,
                destination,
                objective_attribute,
                resource_attribute,
                max_resource,
            )
            if best is not None:
                best = _CandidatePath(
                    best.path,
                    best.edge_path,
                    best.objective,
                    best.distance,
                    best.cmcs,
                    best.feasible,
                    "fallback_dijkstra",
                )
                fallback_used = True
                fallback_feasible_solutions += 1
                best_iteration = best_iteration or 0
                history.append(
                    {
                        "iteration": iterations_completed + 1,
                        "best_objective": _finite_or_none(best.objective),
                        "best_distance": _finite_or_none(best.distance),
                        "best_cmcs": _finite_or_none(best.cmcs),
                        "feasible_ants": 1,
                        "pure_aco_feasible_ants": 0,
                        "seeded_feasible_ants": 0,
                        "fallback_feasible_ants": 1,
                        "best_source": best.source,
                        "success_rate": 0.0,
                        "mean_objective": round(best.objective, 6),
                        "std_objective": 0.0,
                    }
                )

        runtime_ms = round((perf_counter() - started) * 1000.0, 4)
        feasible_solutions = (
            pure_aco_feasible_solutions
            + seeded_feasible_solutions
            + fallback_feasible_solutions
        )
        if best is None:
            return AntColonyResult(
                path=[],
                edge_path=[],
                objective_cost=inf,
                total_distance=inf,
                total_cmcs=inf,
                feasible=False,
                runtime_ms=runtime_ms,
                iterations_completed=iterations_completed,
                ants_total=ants_total,
                feasible_solutions=feasible_solutions,
                pure_aco_feasible_solutions=pure_aco_feasible_solutions,
                seeded_feasible_solutions=seeded_feasible_solutions,
                fallback_feasible_solutions=fallback_feasible_solutions,
                aco_found_feasible=False,
                fallback_used=fallback_used,
                best_iteration=best_iteration,
                random_state=self.random_state,
                run_history=history,
            )

        return AntColonyResult(
            path=list(best.path),
            edge_path=list(best.edge_path),
            objective_cost=float(best.objective),
            total_distance=float(best.distance),
            total_cmcs=float(best.cmcs),
            feasible=True,
            runtime_ms=runtime_ms,
            iterations_completed=iterations_completed,
            ants_total=ants_total,
            feasible_solutions=feasible_solutions,
            pure_aco_feasible_solutions=pure_aco_feasible_solutions,
            seeded_feasible_solutions=seeded_feasible_solutions,
            fallback_feasible_solutions=fallback_feasible_solutions,
            aco_found_feasible=(
                pure_aco_feasible_solutions + seeded_feasible_solutions > 0
            ),
            fallback_used=fallback_used,
            best_iteration=best_iteration,
            random_state=self.random_state,
            run_history=history,
        )

    def _construct_ant_path(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        resource_attribute: str,
        max_resource: float | None,
        pheromones: dict[object, float],
        reachability: dict[object, float],
        connector_paths: dict[object, list[object]],
    ) -> _CandidatePath | None:
        node = origin
        path = [origin]
        edge_path: list[EdgeRef] = []
        visited = {origin}
        objective = 0.0
        distance = 0.0
        cmcs = 0.0
        max_steps = max(4, min(self.graph.number_of_nodes(), 256))
        stack: list[tuple[object, list[object], list[EdgeRef], set, float, float, float]] = []

        for _ in range(max_steps):
            if node == destination:
                return _CandidatePath(
                    tuple(path),
                    tuple(edge_path),
                    objective,
                    distance,
                    cmcs,
                    feasible=max_resource is None or distance <= max_resource + EPS,
                    source="pure_aco",
                )
            if (
                max_resource is not None
                and distance + reachability.get(node, inf) > max_resource + EPS
            ):
                if stack:
                    node, path, edge_path, visited, objective, distance, cmcs = stack.pop()
                    continue
                return None

            if len(path) > 1 and self.random.random() < 0.35:
                connected = self._connector_candidate(
                    node,
                    destination,
                    path,
                    edge_path,
                    visited,
                    objective,
                    distance,
                    cmcs,
                    objective_attribute,
                    resource_attribute,
                    max_resource,
                    connector_paths,
                )
                if connected is not None:
                    return connected

            choices: list[tuple[float, EdgeRef, object, dict]] = []
            for edge, next_node, data in _outgoing_edges(node, self.graph):
                if next_node in visited:
                    continue
                edge_distance = _edge_value(data, resource_attribute)
                next_distance = distance + edge_distance
                if max_resource is not None and next_distance > max_resource + EPS:
                    continue
                if (
                    max_resource is not None
                    and next_distance + reachability.get(next_node, inf)
                    > max_resource + EPS
                ):
                    continue
                edge_objective = _edge_value(data, objective_attribute)
                reachability_bonus = 1.0 / (1.0 + reachability.get(next_node, 1e9))
                heuristic = 1.0 / (
                    edge_objective + 0.05 * edge_distance + EPS
                )
                heuristic *= 1.0 + 0.25 * reachability_bonus
                tau = pheromones.get(_pheromone_key(self.graph, edge), 1.0)
                probability_weight = (tau**self.alpha) * (heuristic**self.beta)
                if probability_weight > 0 and isfinite(probability_weight):
                    choices.append((probability_weight, edge, next_node, data))

            if not choices:
                connected = self._connector_candidate(
                    node,
                    destination,
                    path,
                    edge_path,
                    visited,
                    objective,
                    distance,
                    cmcs,
                    objective_attribute,
                    resource_attribute,
                    max_resource,
                    connector_paths,
                )
                if connected is not None:
                    return connected
                if stack:
                    node, path, edge_path, visited, objective, distance, cmcs = stack.pop()
                    continue
                return None

            _, edge, next_node, data = self._roulette(choices)
            alternatives = sorted(choices, key=lambda item: item[0], reverse=True)[1:4]
            for _, alt_edge, alt_next, alt_data in alternatives:
                alt_dist = distance + _edge_value(alt_data, resource_attribute)
                alt_obj = objective + _edge_value(alt_data, objective_attribute)
                stack.append(
                    (
                        alt_next,
                        path + [alt_next],
                        edge_path + [alt_edge],
                        visited | {alt_next},
                        alt_obj,
                        alt_dist,
                        cmcs + _edge_cmcs(alt_data),
                    )
                )
            edge_path.append(edge)
            path.append(next_node)
            visited.add(next_node)
            objective += _edge_value(data, objective_attribute)
            distance += _edge_value(data, resource_attribute)
            cmcs += _edge_cmcs(data)
            node = next_node

        return None

    def _connector_candidate(
        self,
        node: object,
        destination: object,
        path: list[object],
        edge_path: list[EdgeRef],
        visited: set,
        objective: float,
        distance: float,
        cmcs: float,
        objective_attribute: str,
        resource_attribute: str,
        max_resource: float | None,
        connector_paths: dict[object, list[object]],
    ) -> _CandidatePath | None:
        connector = connector_paths.get(node)
        if not connector or connector[-1] != destination:
            return None
        connector_tail = connector[1:]
        if any(next_node in visited for next_node in connector_tail):
            return None
        try:
            connector_edges = _select_edge_path(
                self.graph,
                connector,
                resource_attribute,
            )
        except nx.NetworkXNoPath:
            return None
        add_distance = _sum_edge_path(self.graph, connector_edges, resource_attribute)
        total_distance = distance + add_distance
        if max_resource is not None and total_distance > max_resource + EPS:
            return None
        total_objective = objective + _sum_edge_path(
            self.graph, connector_edges, objective_attribute
        )
        total_cmcs = cmcs + _sum_edge_path(self.graph, connector_edges, "cmcs")
        return _CandidatePath(
            tuple(path + connector_tail),
            tuple(edge_path + list(connector_edges)),
            total_objective,
            total_distance,
            total_cmcs,
            True,
            "seeded_connector",
        )

    def _roulette(
        self, choices: Sequence[tuple[float, EdgeRef, object, dict]]
    ) -> tuple[float, EdgeRef, object, dict]:
        total = sum(weight for weight, _, _, _ in choices)
        if total <= 0:
            return self.random.choice(list(choices))
        pick = self.random.random() * total
        cumulative = 0.0
        for choice in choices:
            cumulative += choice[0]
            if cumulative >= pick:
                return choice
        return choices[-1]

    def _evaporate(self, pheromones: dict[object, float]) -> None:
        retention = 1.0 - self.evaporation_rate
        for key in list(pheromones):
            pheromones[key] = max(1e-6, pheromones[key] * retention)

    def _deposit_pheromones(
        self,
        pheromones: dict[object, float],
        iteration_solutions: Sequence[_CandidatePath],
        global_best: _CandidatePath | None,
    ) -> None:
        ranked = sorted(
            iteration_solutions,
            key=lambda item: (item.objective, item.distance),
        )[:3]
        if global_best is not None:
            ranked.append(global_best)
        for solution in ranked:
            deposit = self.q / (solution.objective + EPS)
            for edge in solution.edge_path:
                key = _pheromone_key(self.graph, edge)
                pheromones[key] = pheromones.get(key, 1.0) + deposit


class ParetoLabelCorrectingRCSP:
    """Exact Pareto label-correcting solver for one-resource RCSP."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def solve(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        *,
        resource_attribute: str = "length",
        max_resource: float | None = None,
        upper_bound: float | None = None,
        initial_path: Sequence[object] | None = None,
        initial_edge_path: Sequence[EdgeRef] | None = None,
        time_budget_s: float | None = None,
    ) -> RCSPResult:
        _validate_nodes(self.graph, origin, destination)
        started = perf_counter()
        deadline = started + time_budget_s if time_budget_s is not None else None

        labels_created = 1
        labels_expanded = 0
        dominance_prunes = 0
        resource_prunes = 0
        upper_bound_prunes = 0
        feasible_solutions = 0
        incumbent_updates = 0
        timeout = False

        if origin == destination:
            return RCSPResult(
                path=[origin],
                edge_path=[],
                objective_cost=0.0,
                total_distance=0.0,
                total_cmcs=0.0,
                feasible=True,
                optimality_proven=True,
                timeout=False,
                runtime_ms=0.0,
                labels_created=1,
                labels_expanded=0,
                dominance_prunes=0,
                resource_prunes=0,
                upper_bound_prunes=0,
                feasible_solutions=1,
                max_labels_at_node=1,
                incumbent_updates=1,
            )

        incumbent = self._initial_incumbent(
            upper_bound,
            initial_path,
            initial_edge_path,
            objective_attribute,
            resource_attribute,
            max_resource,
        )
        incumbent_objective = incumbent.objective if incumbent else inf
        incumbent_candidate = incumbent
        if incumbent is not None:
            feasible_solutions = 1
            incumbent_updates = 1

        start_label = _Label(origin, 0.0, 0.0, (origin,), ())
        queue: deque[_Label] = deque([start_label])
        labels: dict[object, list[tuple[float, float]]] = {origin: [(0.0, 0.0)]}
        max_labels_at_node = 1

        while queue:
            if _deadline_passed(deadline):
                timeout = True
                break

            label = queue.popleft()
            if _strictly_dominated(labels.get(label.node, []), label.distance, label.objective):
                dominance_prunes += 1
                continue

            if label.node == destination:
                feasible_solutions += 1
                candidate = _CandidatePath(
                    label.path,
                    label.edge_path,
                    label.objective,
                    label.distance,
                    _sum_edge_path(self.graph, label.edge_path, "cmcs"),
                )
                if incumbent_candidate is None or _better_candidate(candidate, incumbent_candidate):
                    incumbent_candidate = candidate
                    incumbent_objective = candidate.objective
                    incumbent_updates += 1
                continue

            labels_expanded += 1
            for edge, next_node, data in _outgoing_edges(label.node, self.graph):
                if next_node in label.path:
                    continue

                edge_distance = _edge_value(data, resource_attribute)
                next_distance = label.distance + edge_distance
                if max_resource is not None and next_distance > max_resource + EPS:
                    resource_prunes += 1
                    continue

                next_objective = label.objective + _edge_value(
                    data, objective_attribute
                )
                if next_objective >= incumbent_objective - EPS:
                    upper_bound_prunes += 1
                    continue

                node_labels = labels.setdefault(next_node, [])
                if _is_dominated(node_labels, next_distance, next_objective):
                    dominance_prunes += 1
                    continue

                # Safe dominance pruning: with nonnegative distance and objective
                # increments, every extension of a dominated label remains no
                # better than the corresponding extension of its dominator.
                _record_label(node_labels, next_distance, next_objective)
                max_labels_at_node = max(max_labels_at_node, len(node_labels))
                queue.append(
                    _Label(
                        next_node,
                        next_distance,
                        next_objective,
                        label.path + (next_node,),
                        label.edge_path + (edge,),
                    )
                )
                labels_created += 1

        runtime_ms = round((perf_counter() - started) * 1000.0, 4)
        if incumbent_candidate is None:
            raise nx.NetworkXNoPath(
                f"{origin!r} to {destination!r} has no feasible RCSP route."
            )

        return RCSPResult(
            path=list(incumbent_candidate.path),
            edge_path=list(incumbent_candidate.edge_path),
            objective_cost=float(incumbent_candidate.objective),
            total_distance=float(incumbent_candidate.distance),
            total_cmcs=float(incumbent_candidate.cmcs),
            feasible=True,
            optimality_proven=not timeout,
            timeout=timeout,
            runtime_ms=runtime_ms,
            labels_created=labels_created,
            labels_expanded=labels_expanded,
            dominance_prunes=dominance_prunes,
            resource_prunes=resource_prunes,
            upper_bound_prunes=upper_bound_prunes,
            feasible_solutions=feasible_solutions,
            max_labels_at_node=max_labels_at_node,
            incumbent_updates=incumbent_updates,
        )

    def _initial_incumbent(
        self,
        upper_bound: float | None,
        initial_path: Sequence[object] | None,
        initial_edge_path: Sequence[EdgeRef] | None,
        objective_attribute: str,
        resource_attribute: str,
        max_resource: float | None,
    ) -> _CandidatePath | None:
        if not initial_path or not initial_edge_path:
            return None
        objective = _sum_edge_path(
            self.graph, initial_edge_path, objective_attribute
        )
        if upper_bound is not None:
            objective = min(objective, float(upper_bound))
        distance = _sum_edge_path(self.graph, initial_edge_path, resource_attribute)
        if max_resource is not None and distance > max_resource + EPS:
            return None
        cmcs = _sum_edge_path(self.graph, initial_edge_path, "cmcs")
        return _CandidatePath(
            tuple(initial_path),
            tuple(initial_edge_path),
            objective,
            distance,
            cmcs,
        )


class HybridACOParetoRCSP:
    """Run ACO first, then use Pareto RCSP to improve or certify it."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        n_ants: int = 32,
        n_iterations: int = 40,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.30,
        q: float = 100.0,
        random_state: int | None = 42,
    ):
        self.graph = graph
        self.aco_params = {
            "n_ants": n_ants,
            "n_iterations": n_iterations,
            "alpha": alpha,
            "beta": beta,
            "evaporation_rate": evaporation_rate,
            "q": q,
            "random_state": random_state,
        }

    def solve(
        self,
        origin: object,
        destination: object,
        objective_attribute: str,
        *,
        resource_attribute: str = "length",
        max_resource: float | None = None,
        time_budget_s: float | None = None,
    ) -> HybridRouteResult:
        started = perf_counter()
        aco_budget, rcsp_budget = _split_budget(time_budget_s)

        aco = AntColonyRouter(
            self.graph,
            **self.aco_params,
            time_budget_s=aco_budget,
        )
        aco_result = aco.solve(
            origin,
            destination,
            objective_attribute,
            resource_attribute=resource_attribute,
            max_resource=max_resource,
            time_budget_s=aco_budget,
        )

        rcsp_result: RCSPResult | None = None
        try:
            rcsp_result = ParetoLabelCorrectingRCSP(self.graph).solve(
                origin,
                destination,
                objective_attribute,
                resource_attribute=resource_attribute,
                max_resource=max_resource,
                upper_bound=aco_result.objective_cost
                if aco_result.feasible
                else None,
                initial_path=aco_result.path if aco_result.feasible else None,
                initial_edge_path=aco_result.edge_path
                if aco_result.feasible
                else None,
                time_budget_s=rcsp_budget,
            )
        except nx.NetworkXNoPath:
            rcsp_result = None

        selected_source: str
        selected: RCSPResult | AntColonyResult
        if rcsp_result is not None and rcsp_result.optimality_proven:
            selected = rcsp_result
            selected_source = "rcsp_certified"
        elif (
            rcsp_result is not None
            and rcsp_result.feasible
            and (
                not aco_result.feasible
                or rcsp_result.objective_cost < aco_result.objective_cost - EPS
                or rcsp_result.incumbent_updates > 1
            )
        ):
            selected = rcsp_result
            selected_source = "rcsp_incumbent"
        elif aco_result.feasible:
            selected = aco_result
            selected_source = "aco_approx"
        else:
            raise nx.NetworkXNoPath(
                f"{origin!r} to {destination!r} has no feasible hybrid route."
            )

        total_runtime_ms = round((perf_counter() - started) * 1000.0, 4)
        shortest = _dijkstra_candidate(
            self.graph,
            origin,
            destination,
            "length",
            resource_attribute,
            None,
        )
        shortest_distance = shortest.distance if shortest else selected.total_distance
        shortest_cmcs = shortest.cmcs if shortest else selected.total_cmcs
        detour_ratio = (
            selected.total_distance / shortest_distance
            if shortest_distance > EPS
            else 0.0
        )
        detour_constraint_satisfied = (
            max_resource is None or selected.total_distance <= max_resource + EPS
        )
        risk_reduction = None
        if shortest_cmcs > EPS:
            risk_reduction = (
                (shortest_cmcs - selected.total_cmcs) / shortest_cmcs * 100.0
            )
        distance_increase = None
        if shortest_distance > EPS:
            distance_increase = (
                (selected.total_distance - shortest_distance)
                / shortest_distance
                * 100.0
            )

        rcsp_objective = (
            rcsp_result.objective_cost if rcsp_result and rcsp_result.feasible else None
        )
        rcsp_total_distance = (
            rcsp_result.total_distance if rcsp_result and rcsp_result.feasible else None
        )
        rcsp_total_cmcs = (
            rcsp_result.total_cmcs if rcsp_result and rcsp_result.feasible else None
        )
        rcsp_used_aco_upper_bound = bool(aco_result.feasible and rcsp_result is not None)
        initial_upper_bound_source = (
            "aco"
            if aco_result.aco_found_feasible
            else "fallback_dijkstra"
            if aco_result.feasible
            else "none"
        )
        optimality_claim_scope = "full_graph"
        gap_pct = _gap_pct(aco_result.objective_cost, rcsp_objective)
        aco_stats = aco_result.to_dict()
        rcsp_stats = rcsp_result.to_dict() if rcsp_result is not None else {}
        search_stats = {
            "algorithm": "aco_pareto_rcsp",
            "selected_source": selected_source,
            "aco_found_feasible": aco_result.aco_found_feasible,
            "aco_total_distance": _finite_or_none(aco_result.total_distance),
            "aco_total_cmcs": _finite_or_none(aco_result.total_cmcs),
            "aco_feasible_solutions": aco_result.feasible_solutions,
            "pure_aco_feasible_solutions": aco_result.pure_aco_feasible_solutions,
            "seeded_feasible_solutions": aco_result.seeded_feasible_solutions,
            "fallback_feasible_solutions": aco_result.fallback_feasible_solutions,
            "rcsp_total_distance": rcsp_total_distance,
            "rcsp_total_cmcs": rcsp_total_cmcs,
            "rcsp_used_aco_upper_bound": rcsp_used_aco_upper_bound,
            "initial_upper_bound_source": initial_upper_bound_source,
            "optimality_claim_scope": optimality_claim_scope,
            "optimality_proven": bool(
                rcsp_result.optimality_proven if rcsp_result else False
            ),
            "timeout": bool(rcsp_result.timeout) if rcsp_result else True,
            "runtime_ms": total_runtime_ms,
            "aco_runtime_ms": aco_result.runtime_ms,
            "rcsp_runtime_ms": rcsp_result.runtime_ms
            if rcsp_result is not None
            else 0.0,
            "aco_objective": _finite_or_none(aco_result.objective_cost),
            "rcsp_objective": rcsp_objective,
            "gap_pct": gap_pct,
            "detour_ratio": round(detour_ratio, 6),
            "detour_constraint_satisfied": detour_constraint_satisfied,
            "risk_reduction_pct_against_shortest": _round_optional(
                risk_reduction, 6
            ),
            "distance_increase_pct_against_shortest": _round_optional(
                distance_increase, 6
            ),
            "ants_total": aco_result.ants_total,
            "rcsp_feasible_solutions": rcsp_result.feasible_solutions
            if rcsp_result
            else 0,
            "labels_created": rcsp_result.labels_created if rcsp_result else 0,
            "labels_expanded": rcsp_result.labels_expanded if rcsp_result else 0,
            "dominance_prunes": rcsp_result.dominance_prunes
            if rcsp_result
            else 0,
            "resource_prunes": rcsp_result.resource_prunes
            if rcsp_result
            else 0,
            "upper_bound_prunes": rcsp_result.upper_bound_prunes
            if rcsp_result
            else 0,
            # Backward-compatible aliases for older consumers.
            "states_expanded": rcsp_result.labels_expanded if rcsp_result else 0,
            "bound_prunes": rcsp_result.upper_bound_prunes if rcsp_result else 0,
            "cycle_prunes": 0,
        }

        return HybridRouteResult(
            path=list(selected.path),
            edge_path=list(selected.edge_path),
            objective_cost=float(selected.objective_cost),
            total_distance=float(selected.total_distance),
            total_cmcs=float(selected.total_cmcs),
            feasible=bool(selected.feasible),
            selected_source=selected_source,
            optimality_proven=search_stats["optimality_proven"],
            aco_objective=_finite_or_none(aco_result.objective_cost),
            aco_total_distance=_finite_or_none(aco_result.total_distance),
            aco_total_cmcs=_finite_or_none(aco_result.total_cmcs),
            aco_found_feasible=aco_result.aco_found_feasible,
            aco_feasible_solutions=aco_result.feasible_solutions,
            pure_aco_feasible_solutions=aco_result.pure_aco_feasible_solutions,
            seeded_feasible_solutions=aco_result.seeded_feasible_solutions,
            rcsp_objective=rcsp_objective,
            rcsp_total_distance=rcsp_total_distance,
            rcsp_total_cmcs=rcsp_total_cmcs,
            rcsp_used_aco_upper_bound=rcsp_used_aco_upper_bound,
            initial_upper_bound_source=initial_upper_bound_source,
            optimality_claim_scope=optimality_claim_scope,
            gap_pct=gap_pct,
            aco_runtime_ms=aco_result.runtime_ms,
            rcsp_runtime_ms=rcsp_result.runtime_ms if rcsp_result else 0.0,
            total_runtime_ms=total_runtime_ms,
            detour_ratio=round(detour_ratio, 6),
            detour_constraint_satisfied=detour_constraint_satisfied,
            risk_reduction_pct_against_shortest=_round_optional(
                risk_reduction, 6
            ),
            distance_increase_pct_against_shortest=_round_optional(
                distance_increase, 6
            ),
            search_stats=search_stats,
            aco_stats=aco_stats,
            rcsp_stats=rcsp_stats,
        )


def _validate_nodes(graph: nx.Graph, origin: object, destination: object) -> None:
    if origin not in graph:
        raise nx.NodeNotFound(f"Origin node {origin!r} is not in the graph.")
    if destination not in graph:
        raise nx.NodeNotFound(
            f"Destination node {destination!r} is not in the graph."
        )


def _deadline_passed(deadline: float | None) -> bool:
    return deadline is not None and perf_counter() > deadline


def _split_budget(time_budget_s: float | None) -> tuple[float | None, float | None]:
    if time_budget_s is None:
        return None, None
    if time_budget_s <= 0:
        return 0.0, 0.0
    aco_budget = min(max(time_budget_s * 0.35, 0.001), time_budget_s)
    return aco_budget, max(0.0, time_budget_s - aco_budget)


def _reverse_for_destination(graph: nx.Graph) -> nx.Graph:
    return graph.reverse(copy=False) if graph.is_directed() else graph


def _length_to_destination(
    graph: nx.Graph,
    destination: object,
    resource_attribute: str,
) -> dict[object, float]:
    try:
        return nx.single_source_dijkstra_path_length(
            _reverse_for_destination(graph),
            destination,
            weight=_networkx_weight(resource_attribute),
        )
    except (nx.NetworkXException, ValueError):
        return {}


def _length_paths_to_destination(
    graph: nx.Graph,
    destination: object,
    resource_attribute: str,
) -> dict[object, list[object]]:
    try:
        reverse_paths = nx.single_source_dijkstra_path(
            _reverse_for_destination(graph),
            destination,
            weight=_networkx_weight(resource_attribute),
        )
    except (nx.NetworkXException, ValueError):
        return {}
    return {node: list(reversed(path)) for node, path in reverse_paths.items()}


def _edge_value(data: dict, attribute: str) -> float:
    value = float(data.get(attribute, inf))
    if value < 0:
        raise ValueError(
            f"ACO-Pareto RCSP requires nonnegative edge weights: {attribute}={value}"
        )
    return value


def _edge_cmcs(data: dict) -> float:
    if "risk_exposure" in data:
        return float(data.get("risk_exposure", 0.0))
    length = float(data.get("length", 0.0))
    if "cmcs" in data:
        return length * float(data.get("cmcs", 0.0))
    if "safety_cost" in data:
        return float(data.get("safety_cost", 0.0))
    return 0.0


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


def _all_directed_edge_choices(
    graph: nx.Graph,
) -> Iterator[tuple[EdgeRef, object, dict]]:
    for node in graph.nodes:
        yield from _outgoing_edges(node, graph)


def _edge_data(graph: nx.Graph, edge: EdgeRef) -> dict:
    u, v, key = edge
    if graph.is_multigraph():
        return graph[u][v][key]
    return graph[u][v]


def _sum_edge_path(
    graph: nx.Graph, edge_path: Iterable[EdgeRef], attribute: str
) -> float:
    total = 0.0
    for edge in edge_path:
        data = _edge_data(graph, edge)
        total += _edge_cmcs(data) if attribute == "cmcs" else _edge_value(data, attribute)
    return float(total)


def _is_dominated(
    labels: Sequence[tuple[float, float]], distance: float, objective: float
) -> bool:
    return any(
        known_distance <= distance + EPS and known_objective <= objective + EPS
        for known_distance, known_objective in labels
    )


def _strictly_dominated(
    labels: Sequence[tuple[float, float]], distance: float, objective: float
) -> bool:
    return any(
        known_distance <= distance + EPS
        and known_objective <= objective + EPS
        and (
            known_distance < distance - EPS
            or known_objective < objective - EPS
        )
        for known_distance, known_objective in labels
    )


def _record_label(
    labels: list[tuple[float, float]], distance: float, objective: float
) -> None:
    labels[:] = [
        (known_distance, known_objective)
        for known_distance, known_objective in labels
        if not (
            distance <= known_distance + EPS
            and objective <= known_objective + EPS
            and (
                distance < known_distance - EPS
                or objective < known_objective - EPS
            )
        )
    ]
    labels.append((distance, objective))


def _better_candidate(left: _CandidatePath, right: _CandidatePath) -> bool:
    return left.objective < right.objective - EPS or (
        abs(left.objective - right.objective) <= EPS
        and left.distance < right.distance - EPS
    )


def _pheromone_key(graph: nx.Graph, edge: EdgeRef) -> object:
    if graph.is_directed():
        return edge
    u, v, key = edge
    return frozenset((u, v)), key


def _networkx_weight(attribute: str):
    def weight(_u: object, _v: object, data: dict) -> float:
        if data and all(isinstance(value, dict) for value in data.values()):
            return min(_edge_value(edge_data, attribute) for edge_data in data.values())
        return _edge_value(data, attribute)

    return weight


def _dijkstra_candidate(
    graph: nx.Graph,
    origin: object,
    destination: object,
    objective_attribute: str,
    resource_attribute: str,
    max_resource: float | None,
) -> _CandidatePath | None:
    try:
        path = nx.dijkstra_path(
            graph,
            origin,
            destination,
            weight=_networkx_weight(objective_attribute),
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    edge_path = tuple(
        _select_edge_path(graph, path, objective_attribute)
    )
    objective = _sum_edge_path(graph, edge_path, objective_attribute)
    distance = _sum_edge_path(graph, edge_path, resource_attribute)
    if max_resource is not None and distance > max_resource + EPS:
        return None
    return _CandidatePath(
        tuple(path),
        edge_path,
        objective,
        distance,
        _sum_edge_path(graph, edge_path, "cmcs"),
    )


def _select_edge_path(
    graph: nx.Graph, path: Sequence[object], cost_attribute: str
) -> list[EdgeRef]:
    selected: list[EdgeRef] = []
    for u, v in zip(path, path[1:]):
        if graph.is_multigraph():
            edge_options = graph.get_edge_data(u, v) or {}
            if not edge_options:
                raise nx.NetworkXNoPath(f"{u!r} -> {v!r} edge is missing.")
            key, _ = min(
                edge_options.items(),
                key=lambda item: _edge_value(item[1], cost_attribute),
            )
        else:
            if graph.get_edge_data(u, v) is None:
                raise nx.NetworkXNoPath(f"{u!r} -> {v!r} edge is missing.")
            key = 0
        selected.append((u, v, key))
    return selected


def _finite_or_none(value: float) -> float | None:
    return float(value) if isfinite(value) else None


def _round_optional(value: float | None, digits: int) -> float | None:
    return round(float(value), digits) if value is not None and isfinite(value) else None


def _gap_pct(
    aco_objective: float | None, rcsp_objective: float | None
) -> float | None:
    if (
        aco_objective is None
        or rcsp_objective is None
        or not isfinite(aco_objective)
        or not isfinite(rcsp_objective)
        or abs(rcsp_objective) <= EPS
    ):
        return None
    return round((aco_objective - rcsp_objective) / rcsp_objective * 100.0, 6)
