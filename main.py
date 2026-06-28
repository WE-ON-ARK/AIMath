from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from config import (
    GRAPH_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORT_OUTPUT_DIR,
    SETTINGS,
    ensure_directories,
)
from src.data_collector import DataCollector, build_pedestrian_network
from src.api_preprocessing import preprocess_existing_api_files
from src.data_driven_cmcs import derive_data_driven_cmcs_weights
from src.demo import run_demo
from src.full_pipeline import (
    EDGE_FEATURE_PATH,
    run_full_pipeline,
    train_edge_risk_model,
    train_regional_boosting_model,
)
from src.final_model_evaluation import generate_final_model_evaluation
from src.real_data_pipeline import run_real_data_pipeline
from src.visualize_risk_zones import generate_risk_zone_maps_from_saved_artifacts


def _artifact_ref(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def clean_artifacts() -> dict[str, int]:
    removed = {"files": 0, "directories": 0}
    root = Path(".")
    directory_names = {"__pycache__", ".pytest_cache", "cmcs_safe_route.egg-info"}
    old_pulse_reports = {
        "outputs/reports/pulse_algorithm_evaluation.json",
        "outputs/reports/pulse_algorithm_performance.csv",
        "outputs/reports/route_comparison.csv",
        "outputs/reports/pareto_front.csv",
    }
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        relative = path.as_posix()
        if path.is_dir() and path.name in directory_names:
            shutil.rmtree(path, ignore_errors=True)
            removed["directories"] += 1
        elif path.is_file() and (
            path.suffix == ".pyc"
            or path.name in {".DS_Store", "outputs.zip"}
            or relative in old_pulse_reports
        ):
            path.unlink(missing_ok=True)
            removed["files"] += 1
    return removed


def run_algorithm_evaluation() -> dict[str, object]:
    result = run_demo(with_visuals=False)
    comparison = result["comparison"]
    history_rows = []
    rcsp_rows = []
    for _, row in comparison.iterrows():
        aco_stats = row.get("aco_stats") or {}
        rcsp_stats = row.get("rcsp_stats") or {}
        for history in aco_stats.get("run_history", []):
            history_rows.append({"mode": row["mode"], **history})
        rcsp_rows.append(
            {
                "mode": row["mode"],
                "selected_source": row["selected_source"],
                "optimality_proven": bool(row["optimality_proven"]),
                "optimality_claim_scope": row["optimality_claim_scope"],
                "timeout": bool(row["search_stats"]["timeout"]),
                "aco_found_feasible": bool(row["aco_found_feasible"]),
                "aco_objective": row["aco_objective"],
                "rcsp_objective": row["rcsp_objective"],
                "gap_pct": row["gap_pct"],
                "detour_ratio": row["detour_ratio"],
                "detour_constraint_satisfied": bool(
                    row["detour_constraint_satisfied"]
                ),
                "risk_reduction_pct_against_shortest": row[
                    "risk_reduction_pct_against_shortest"
                ],
                "distance_increase_pct_against_shortest": row[
                    "distance_increase_pct_against_shortest"
                ],
                "labels_created": rcsp_stats.get("labels_created", 0),
                "labels_expanded": rcsp_stats.get("labels_expanded", 0),
                "dominance_prunes": rcsp_stats.get("dominance_prunes", 0),
                "resource_prunes": rcsp_stats.get("resource_prunes", 0),
                "upper_bound_prunes": rcsp_stats.get("upper_bound_prunes", 0),
                "rcsp_used_aco_upper_bound": bool(
                    row["rcsp_used_aco_upper_bound"]
                ),
            }
        )

    import pandas as pd

    aco_history_path = REPORT_OUTPUT_DIR / "aco_run_history.csv"
    pd.DataFrame(history_rows).to_csv(
        aco_history_path, index=False, encoding="utf-8-sig"
    )
    rcsp_report_path = REPORT_OUTPUT_DIR / "rcsp_certification_report.json"
    rcsp_report = {
        "algorithm": "aco_pareto_rcsp",
        "routes": rcsp_rows,
        "optimality_proven_rate": float(
            comparison["optimality_proven"].mean()
        ),
    }
    rcsp_report_path.write_text(
        json.dumps(rcsp_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = REPORT_OUTPUT_DIR / "route_algorithm_summary.md"
    summary_path.write_text(
        """# ACO-Pareto RCSP Route Algorithm Summary

- Route distance: `L(P)=sum length_e`
- Route risk cost: `C(P)=sum safety_cost_e`
- Detour constraint: `L(P) <= L(P_shortest) * r_age`
- Balanced objective: `Score_lambda(P)=lambda L(P)+(1-lambda)C(P)`
- Pareto dominance removes labels with no better distance or objective.
- ACO는 확률적 후보 경로와 초기 상한을 제공하고, Pareto Label-Correcting RCSP가 지정된 탐색 범위에서 종료되는 경우 해당 범위 내 최적성을 인증한다.
""",
        encoding="utf-8",
    )
    return {
        "algorithm": "aco_pareto_rcsp",
        "routes": int(len(comparison)),
        "evaluation_json": _artifact_ref(
            REPORT_OUTPUT_DIR / "aco_pareto_algorithm_evaluation.json"
        ),
        "performance_csv": _artifact_ref(
            REPORT_OUTPUT_DIR / "aco_pareto_algorithm_performance.csv"
        ),
        "aco_run_history_csv": _artifact_ref(aco_history_path),
        "rcsp_certification_json": _artifact_ref(rcsp_report_path),
        "summary_md": _artifact_ref(summary_path),
    }


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

    subparsers.add_parser(
        "evaluate-algorithm",
        help="Run only the ACO-Pareto RCSP route algorithm evaluation",
    )

    subparsers.add_parser(
        "visualize-risk-zones",
        help="Generate CMCS risk-zone overview and pilot-route maps",
    )

    subparsers.add_parser(
        "clean-artifacts",
        help="Remove caches, bytecode, zip files, egg-info, and old Pulse reports",
    )

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
    preprocess_parser = subparsers.add_parser(
        "preprocess-api-data",
        help="캐시된 공공데이터 API 파일의 좌표·중복·수치·시점을 표준화",
    )
    preprocess_parser.add_argument("--raw-dir", default="data/raw")
    weight_parser = subparsers.add_parser(
        "derive-cmcs-weights",
        help="Spearman·Logistic·Poisson·Moran's I로 CMCS 가중치 산출",
    )
    weight_parser.add_argument(
        "--edge-features",
        default=str(EDGE_FEATURE_PATH),
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
        regional_metrics = regional_model["nested_validation_metrics"]
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
            f"권역 부스팅: XGBoost 앙상블(중첩 검증), "
            f"F1={regional_metrics['f1']:.3f}, "
            f"Precision={regional_metrics['precision']:.3f}, "
            f"Recall={regional_metrics['recall']:.3f}, "
            f"목표달성={regional_model['f1_target_achieved']}"
        )
        print(
            f"경로 알고리즘: {route['algorithm']}, "
            f"실제 경로: {route['origin']} → {route['destination']}, "
            f"최단 {route['shortest_distance_m']:.0f}m / "
            f"안전 {route['safest_distance_m']:.0f}m / "
            f"위험노출 감소 {route['risk_reduction_pct']:.1f}%"
        )
        stability = route["stability_validation"]
        print(
            f"다중 OD 검증: {stability['evaluated_pairs']}개, "
            f"위험감소 경로 {stability['positive_risk_reduction_ratio'] * 100:.1f}%, "
            f"중앙 위험감소 {stability['median_risk_reduction_pct']:.1f}%, "
            f"안정성={stability['route_selection_stability_passed']}"
        )
        print(f"지도: {report['artifacts']['route_map']}")
    elif command == "evaluate-algorithm":
        report = run_algorithm_evaluation()
        print("\nACO-Pareto RCSP algorithm evaluation complete")
        print(f"Routes evaluated: {report['routes']}")
        print(f"Evaluation JSON: {report['evaluation_json']}")
        print(f"Performance CSV: {report['performance_csv']}")
    elif command == "visualize-risk-zones":
        outputs = generate_risk_zone_maps_from_saved_artifacts()
        print("\nCMCS risk-zone maps generated")
        for name, path in outputs.items():
            print(f"{name}: {_artifact_ref(path)}")
    elif command == "clean-artifacts":
        removed = clean_artifacts()
        print(
            "Cleaned artifacts: "
            f"{removed['files']} files, {removed['directories']} directories"
        )
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
        regional_metrics = regional_report["nested_validation_metrics"]
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
    elif command == "preprocess-api-data":
        reports = preprocess_existing_api_files(args.raw_dir)
        print("\n공공데이터 API 전처리 완료")
        for name, report in reports.items():
            print(
                f"{name}: {report['input_rows']} → {report['output_rows']}행, "
                f"중복 {report['duplicate_rows_removed']}건 제거, "
                f"좌표 이상 {report['out_of_daejeon_rows']}건 제거"
            )
    elif command == "derive-cmcs-weights":
        import pandas as pd

        edge_features = pd.read_csv(args.edge_features)
        weights, report = derive_data_driven_cmcs_weights(edge_features)
        print("\n데이터 기반 CMCS 가중치 산출 완료")
        for dimension, weight in weights.dimensions.items():
            print(f"{dimension}: {weight:.4f}")
        print(
            f"Logistic 공간 ROC-AUC="
            f"{report['analyses']['logistic']['roc_auc']:.3f}"
        )
        print(f"보고서: outputs/reports/cmcs_weight_evidence_report.json")


if __name__ == "__main__":
    main()
