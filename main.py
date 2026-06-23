from __future__ import annotations

import argparse

from config import (
    GRAPH_DATA_DIR,
    PROCESSED_DATA_DIR,
    SETTINGS,
    ensure_directories,
)
from src.data_collector import DataCollector, build_pedestrian_network
from src.demo import run_demo
from src.full_pipeline import (
    EDGE_FEATURE_PATH,
    run_full_pipeline,
    train_edge_risk_model,
    train_regional_boosting_model,
)
from src.final_model_evaluation import generate_final_model_evaluation
from src.real_data_pipeline import run_real_data_pipeline


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

    real_parser = subparsers.add_parser(
        "real-data", help="병합된 대전 실데이터로 학교 사고 다발지역 모델 학습"
    )
    real_parser.add_argument(
        "--accident-file",
        default="data/raw/daejeon_schoolzone_accident_hotspots.csv",
    )
    real_parser.add_argument("--radius", type=float, default=300.0)

    full_parser = subparsers.add_parser(
        "full-pipeline",
        help="대전 전체 실제 데이터 수집부터 CMCS 경로 추천까지 실행",
    )
    full_parser.add_argument("--refresh-data", action="store_true")
    full_parser.add_argument("--refresh-network", action="store_true")

    train_parser = subparsers.add_parser(
        "train-edge-models",
        help="기존 도로 피처로 RandomForest·XGBoost 공간 교차검증 및 최종 학습",
    )
    train_parser.add_argument(
        "--edge-features", default=str(EDGE_FEATURE_PATH)
    )
    subparsers.add_parser(
        "evaluate-safety",
        help="최종 모델 안전성·적용 가능성 표와 대시보드 생성",
    )
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
    elif command == "real-data":
        result = run_real_data_pipeline(
            accident_path=args.accident_file,
            label_radius_m=args.radius,
        )
        report = result["model_report"]
        verification = result["verification"]
        best = report["best_model"]
        metrics = report["models"][best]
        print("\n대전 실데이터 학습 완료")
        print(
            f"학교 {report['dataset']['school_count']}개 / "
            f"사고 다발지역 인접 {report['dataset']['positive_count']}개"
        )
        print(
            f"최적 모델: {best}, ROC-AUC={metrics['roc_auc']:.3f}, "
            f"AP={metrics['average_precision']:.3f}"
        )
        print(
            f"검증 지역: {verification['nearest_school']} 인근, "
            f"{verification['years']}, 사고 {verification['accident_count']}건, "
            f"사망 {verification['death_count']}명"
        )
        print(f"배포 가능 판정: {report['deployment_ready']}")
    elif command == "full-pipeline":
        report = run_full_pipeline(
            refresh_data=args.refresh_data,
            refresh_network=args.refresh_network,
        )
        route = report["route"]
        edge_model = report["edge_model"]
        best_model = edge_model["best_model"]
        edge_metrics = edge_model["models"][best_model]
        regional_model = report["regional_boosting_model"]
        regional_metrics = regional_model["metrics"]["optimized_threshold"]
        print("\nCMCS 전체 파이프라인 완료")
        print(
            f"도로망: 노드 {report['graph']['nodes']:,}, "
            f"방향 간선 {report['graph']['directed_edges']:,}, "
            f"고유 구간 {report['graph']['unique_segments']:,}"
        )
        print(
            f"도로 위험 모델: {best_model}, "
            f"ROC-AUC={edge_metrics['roc_auc']:.3f}, "
            f"AP={edge_metrics['average_precision']:.3f}, "
            f"연구검증={edge_model['research_validation_passed']}, "
            f"운영배포={edge_model['production_deployment_ready']}"
        )
        print(
            f"권역 부스팅: XGBoost, "
            f"F1={regional_metrics['f1']:.3f}, "
            f"Precision={regional_metrics['precision']:.3f}, "
            f"Recall={regional_metrics['recall']:.3f}, "
            f"목표달성={regional_model['f1_target_achieved']}"
        )
        print(
            f"실제 경로: {route['origin']} → {route['destination']}, "
            f"최단 {route['shortest_distance_m']:.0f}m / "
            f"안전 {route['safest_distance_m']:.0f}m / "
            f"위험노출 감소 {route['risk_reduction_pct']:.1f}%"
        )
        print(f"지도: {report['artifacts']['route_map']}")
    elif command == "train-edge-models":
        import pandas as pd

        edge_features = pd.read_csv(args.edge_features)
        segment_scores, report = train_edge_risk_model(edge_features)
        segment_scores, regional_report = train_regional_boosting_model(
            segment_scores
        )
        score_path = PROCESSED_DATA_DIR / "edge_model_segment_scores.csv"
        segment_scores.to_csv(
            score_path, index=False, encoding="utf-8-sig"
        )
        leaderboard = pd.DataFrame(
            [
                {
                    "model": name,
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "brier_score": metrics["brier_score"],
                    "optimized_threshold": metrics[
                        "optimized_threshold"
                    ]["threshold"],
                    "optimized_f1": metrics["optimized_threshold"]["f1"],
                }
                for name, metrics in report["models"].items()
            ]
        ).sort_values(
            ["average_precision", "roc_auc"], ascending=False
        )
        print("\n도로 위험 모델 비교 학습 완료")
        print(leaderboard.to_string(index=False))
        print(
            f"\n최종 트리 모델: {report['best_model']} / "
            f"임계값 {report['selected_decision_threshold']:.4f}"
        )
        regional_metrics = regional_report["metrics"]["optimized_threshold"]
        print(
            f"권역 XGBoost: F1={regional_metrics['f1']:.4f}, "
            f"Precision={regional_metrics['precision']:.4f}, "
            f"Recall={regional_metrics['recall']:.4f}, "
            f"목표달성={regional_report['f1_target_achieved']}"
        )
        print(f"모델: models/edge_accident_risk_model.pkl")
        print(f"권역 모델: models/regional_xgboost_risk_model.pkl")
        print(f"구간 점수: {score_path}")
    elif command == "evaluate-safety":
        report = generate_final_model_evaluation()
        metrics = report["regional_metrics"]
        print("\n최종 모델 안전성 평가 완료")
        print(f"종합 판정: {report['overall_verdict']}")
        print(
            f"권역 ROC-AUC={metrics['roc_auc']:.3f}, "
            f"F1={metrics['f1']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}"
        )
        print(f"대시보드: {report['artifacts']['dashboard_png']}")
        print(f"HTML 보고서: {report['artifacts']['html_report']}")


if __name__ == "__main__":
    main()
