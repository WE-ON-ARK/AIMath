from pathlib import Path

import networkx as nx
import pytest

from src.pulse_algorithm import PulseAlgorithm


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
    graph.add_edge("A", "B", edge_id="ab", length=10.0, cost=10.0)
    graph.add_edge("B", "D", edge_id="bd", length=10.0, cost=10.0)
    graph.add_edge("A", "C", edge_id="ac", length=12.0, cost=3.0)
    graph.add_edge("C", "D", edge_id="cd", length=12.0, cost=3.0)
    return graph


def test_pulse_finds_minimum_objective_path():
    result = PulseAlgorithm(_graph()).solve("A", "D", "cost")
    assert result.path == ["A", "C", "D"]
    assert result.objective_cost == 6.0
    assert result.stats.algorithm == "pulse"
    assert result.stats.optimality_proven


def test_pulse_honors_distance_resource_constraint():
    result = PulseAlgorithm(_graph()).solve(
        "A",
        "D",
        "cost",
        max_resource=20.0,
    )
    assert result.path == ["A", "B", "D"]
    assert result.total_distance == 20.0
    assert result.stats.resource_prunes > 0


def test_pulse_reports_no_path_when_resource_is_too_small():
    with pytest.raises(nx.NetworkXNoPath):
        PulseAlgorithm(_graph()).solve(
            "A",
            "D",
            "cost",
            max_resource=19.0,
        )


def test_pulse_supports_parallel_edges():
    graph = nx.MultiDiGraph()
    graph.add_node("A", x=0.0, y=0.0)
    graph.add_node("B", x=1.0, y=0.0)
    graph.add_edge("A", "B", edge_id="expensive", length=5.0, cost=9.0)
    graph.add_edge("A", "B", edge_id="cheap", length=7.0, cost=2.0)
    result = PulseAlgorithm(graph).solve("A", "B", "cost")
    assert result.edge_path[0][2] == 1
    assert result.objective_cost == 2.0


def test_route_code_contains_no_dijkstra_or_networkx_shortest_path():
    project_root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            project_root / "src" / "pulse_algorithm.py",
            project_root / "src" / "route_optimizer.py",
        )
    ).lower()
    assert "dijkstra" not in source
    assert "nx.shortest_path" not in source
    assert "nx.astar_path" not in source
