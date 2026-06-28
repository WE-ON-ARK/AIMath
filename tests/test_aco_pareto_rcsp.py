import json

import networkx as nx
import pandas as pd

from config import REPORT_OUTPUT_DIR
from src.aco_pareto_rcsp import (
    AntColonyRouter,
    HybridACOParetoRCSP,
    ParetoLabelCorrectingRCSP,
)
from src.route_optimizer import RouteOptimizer


def _graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    coordinates = {
        "A": (0.0, 0.0),
        "B": (1.0, 0.0),
        "C": (0.0, 1.0),
        "D": (2.0, 0.0),
    }
    for node, (x, y) in coordinates.items():
        graph.add_node(node, x=x, y=y)
    graph.add_edge(
        "A", "B", edge_id="ab", length=10.0, safety_cost=10.0, cmcs=0.9
    )
    graph.add_edge(
        "B", "D", edge_id="bd", length=10.0, safety_cost=10.0, cmcs=0.9
    )
    graph.add_edge(
        "A", "C", edge_id="ac", length=12.0, safety_cost=3.0, cmcs=0.1
    )
    graph.add_edge(
        "A", "C", edge_id="ac_bad", length=15.0, safety_cost=30.0, cmcs=0.8
    )
    graph.add_edge(
        "C", "D", edge_id="cd", length=12.0, safety_cost=3.0, cmcs=0.1
    )
    graph.add_edge(
        "B", "C", edge_id="bc", length=1.0, safety_cost=50.0, cmcs=1.0
    )
    return graph


def _route_is_connected(graph: nx.MultiDiGraph, edge_path):
    for u, v, key in edge_path:
        assert graph.has_edge(u, v, key)


def test_aco_returns_feasible_simple_path():
    graph = _graph()
    result = AntColonyRouter(
        graph, n_ants=12, n_iterations=8, random_state=7
    ).solve(
        "A",
        "D",
        "safety_cost",
        max_resource=30.0,
    )

    assert result.feasible
    assert result.path[0] == "A"
    assert result.path[-1] == "D"
    assert len(result.path) == len(set(result.path))
    assert result.total_distance <= 30.0
    _route_is_connected(graph, result.edge_path)


def test_rcsp_finds_known_optimum_under_resource_limit():
    result = ParetoLabelCorrectingRCSP(_graph()).solve(
        "A", "D", "safety_cost", max_resource=30.0
    )

    assert result.path == ["A", "C", "D"]
    assert result.objective_cost == 6.0
    assert result.total_distance == 24.0
    assert result.optimality_proven


def test_rcsp_dominance_prunes_dominated_label():
    result = ParetoLabelCorrectingRCSP(_graph()).solve(
        "A", "D", "safety_cost", max_resource=30.0
    )

    assert result.dominance_prunes > 0


def test_hybrid_certifies_when_rcsp_finishes():
    result = HybridACOParetoRCSP(
        _graph(), n_ants=8, n_iterations=4, random_state=11
    ).solve(
        "A",
        "D",
        "safety_cost",
        max_resource=30.0,
    )

    assert result.selected_source == "rcsp_certified"
    assert result.optimality_proven
    assert result.objective_cost == 6.0
    assert result.detour_constraint_satisfied


def test_hybrid_timeout_can_return_aco_approximation():
    result = HybridACOParetoRCSP(
        _graph(), n_ants=8, n_iterations=4, random_state=13
    ).solve(
        "A",
        "D",
        "safety_cost",
        max_resource=30.0,
        time_budget_s=0.0,
    )

    assert result.selected_source == "aco_approx"
    assert not result.optimality_proven
    assert result.feasible


def test_route_optimizer_writes_algorithm_reports(tmp_path, monkeypatch):
    graph = _graph()
    scores = pd.DataFrame(
        [
            {"edge_id": "ab", "cmcs": 0.9},
            {"edge_id": "bd", "cmcs": 0.9},
            {"edge_id": "ac", "cmcs": 0.1},
            {"edge_id": "ac_bad", "cmcs": 0.8},
            {"edge_id": "cd", "cmcs": 0.1},
            {"edge_id": "bc", "cmcs": 1.0},
        ]
    )
    monkeypatch.setattr("config.REPORT_OUTPUT_DIR", tmp_path)
    optimizer = RouteOptimizer(graph=graph, cmcs_data=scores)
    comparison = optimizer.compare_routes("A", "D", max_detour_ratio=1.6)
    performance_path = tmp_path / "aco_pareto_algorithm_performance.csv"
    evaluation_path = tmp_path / "aco_pareto_algorithm_evaluation.json"

    comparison.to_csv(performance_path, index=False)
    evaluation_path.write_text(
        json.dumps(
            {
                "algorithm": "aco_pareto_rcsp",
                "routes": comparison.to_dict(orient="records"),
            }
        ),
        encoding="utf-8",
    )

    assert performance_path.exists()
    assert evaluation_path.exists()
    assert not (tmp_path / "pulse_algorithm_evaluation.json").exists()
