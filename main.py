from __future__ import annotations

import argparse

from config import GRAPH_DATA_DIR, SETTINGS, ensure_directories
from src.data_collector import DataCollector, build_pedestrian_network
from src.demo import run_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CMCS 어린이 안전 통학 경로 추천 시스템"
    )
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="오프라인 합성 데이터 데모")
    demo_parser.add_argument(
        "--no-visuals", action="store_true", help="HTML 시각화 생성을 생략"
    )

    network_parser = subparsers.add_parser(
        "download-network", help="OSM 대전 보행 도로망 다운로드"
    )
    network_parser.add_argument("--place", default=SETTINGS.place)

    collect_parser = subparsers.add_parser(
        "collect", help="등록된 공공데이터 API 수집"
    )
    collect_parser.add_argument("--year", type=int, default=2023)
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    command = args.command or "demo"

    if command == "demo":
        result = run_demo(with_visuals=not getattr(args, "no_visuals", False))
        print("\nCMCS 오프라인 데모 완료")
        print(
            result["comparison"][
                [
                    "mode",
                    "total_distance_m",
                    "total_cmcs",
                    "cmcs_reduction_pct",
                ]
            ].to_string(index=False)
        )
        print("\n산출물: data/processed, data/graph, outputs/")
    elif command == "download-network":
        graph = build_pedestrian_network(
            args.place, GRAPH_DATA_DIR / "daejeon_walk.graphml"
        )
        print(f"도로망 저장 완료: 노드 {graph.number_of_nodes():,}, 간선 {graph.number_of_edges():,}")
    elif command == "collect":
        collector = DataCollector(
            api_key=SETTINGS.api_key,
            base_url=SETTINGS.base_url,
            request_interval_seconds=SETTINGS.request_interval_seconds,
            timeout_seconds=SETTINGS.request_timeout_seconds,
        )
        school_zones = collector.fetch_school_zone()
        accidents = collector.fetch_traffic_accident(args.year)
        print(f"어린이보호구역 {len(school_zones):,}건, 교통사고 {len(accidents):,}건")


if __name__ == "__main__":
    main()

