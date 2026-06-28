from pathlib import Path

from src.demo import run_demo


def test_demo_generates_safer_alternative(tmp_path, monkeypatch):
    import src.demo as demo

    monkeypatch.setattr(demo, "PROCESSED_DATA_DIR", tmp_path / "processed")
    monkeypatch.setattr(demo, "GRAPH_DATA_DIR", tmp_path / "graph")
    monkeypatch.setattr(demo, "REPORT_OUTPUT_DIR", tmp_path / "reports")
    for path in (
        demo.PROCESSED_DATA_DIR,
        demo.GRAPH_DATA_DIR,
        demo.REPORT_OUTPUT_DIR,
    ):
        path.mkdir(parents=True)

    result = run_demo(with_visuals=False)
    assert result["safest"]["total_cmcs"] < result["shortest"]["total_cmcs"]
    assert (tmp_path / "processed" / "edge_cmcs.csv").exists()
    assert (Path("outputs/debug") / "route_comparison.csv").exists()

