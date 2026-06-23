import networkx as nx
import pandas as pd

from src.route_optimizer import RouteOptimizer


def build_test_graph():
    graph = nx.MultiDiGraph()
    for index, node in enumerate(["A", "B", "C", "D"]):
        graph.add_node(node, x=float(index), y=0.0)
    edges = [
        ("A", "B", "direct-1", 100.0, 0.9),
        ("B", "D", "direct-2", 100.0, 0.9),
        ("A", "C", "safe-1", 140.0, 0.1),
        ("C", "D", "safe-2", 140.0, 0.1),
    ]
    for u, v, edge_id, length, _ in edges:
        graph.add_edge(u, v, edge_id=edge_id, length=length)
    scores = pd.DataFrame(
        [{"edge_id": edge_id, "cmcs": cmcs} for _, _, edge_id, _, cmcs in edges]
    )
    return graph, scores


def test_safest_route_can_trade_distance_for_lower_risk():
    graph, scores = build_test_graph()
    optimizer = RouteOptimizer(graph=graph, cmcs_data=scores)
    shortest = optimizer.shortest_route("A", "D")
    safest = optimizer.safest_route("A", "D")

    assert shortest["path"] == ["A", "B", "D"]
    assert safest["path"] == ["A", "C", "D"]
    assert safest["total_distance_m"] > shortest["total_distance_m"]
    assert safest["total_cmcs"] < shortest["total_cmcs"]
    assert shortest["algorithm"] == "pulse"
    assert safest["algorithm"] == "pulse"
    assert shortest["search_stats"]["optimality_proven"]
    assert safest["search_stats"]["optimality_proven"]


def test_parallel_edge_uses_the_edge_matching_route_cost():
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", edge_id="short-risky", length=10, cmcs=0.9)
    graph.add_edge("A", "B", edge_id="long-safe", length=15, cmcs=0.1)
    optimizer = RouteOptimizer(graph=graph)

    assert optimizer.path_edge_ids(optimizer.shortest_route("A", "B")) == [
        "short-risky"
    ]
    assert optimizer.path_edge_ids(optimizer.safest_route("A", "B")) == ["long-safe"]

