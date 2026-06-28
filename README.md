# ACO-Pareto RCSP 기반 우회율 제약 안전 경로 탐색

이 프로젝트의 최종 경로 탐색 알고리즘은 `aco_pareto_rcsp`입니다. ACO가 먼저 좋은 후보 경로를 찾고, Pareto Label-Correcting RCSP가 거리 자원 제약 안에서 해당 후보를 개선하거나 최적성을 인증합니다.

## 알고리즘 구조

- `AntColonyRouter`: pheromone과 heuristic을 이용해 확률적으로 feasible 후보 경로를 탐색합니다.
- `ParetoLabelCorrectingRCSP`: `min C(P)` subject to `L(P) <= B`를 푸는 정확 검증 알고리즘입니다.
- `HybridACOParetoRCSP`: ACO 후보의 objective를 upper bound로 사용하고, RCSP 결과에 따라 `rcsp_certified`, `rcsp_incumbent`, `aco_approx` 중 최종 경로 출처를 기록합니다.

수학적 정의:

- 경로 거리: `L(P)=sum(length_e)`
- 경로 위험비용: `C(P)=sum(safety_cost_e)`
- 우회율 제약: `L(P) <= L(P_shortest) * r_age`
- 균형 목적함수: `Score_lambda(P)=lambda L(P)+(1-lambda)C(P)`
- Pareto dominance: 같은 노드에서 `L_A <= L_B`, `C_A <= C_B`이고 하나 이상이 strict이면 B label을 제거합니다.

ACO는 전역 최적성을 보장하지 않습니다. 따라서 최종 평가는 RCSP 인증 여부(`optimality_proven`)와 ACO 대비 RCSP gap으로 안정성을 보고합니다.

## CMCS 및 보조 모델

LogisticRegression, RandomForest, XGBoost는 경로 탐색 모델이 아닙니다. 이들은 간선별 CMCS 위험도를 보정하고 설명하기 위한 지능형 보조 모델입니다.

모델 평가지표:

- CMCS 보정 모델: Average Precision, ROC-AUC, Brier score
- 경로 탐색 알고리즘: 거리 증가율, CMCS 감소율, 우회율 제약 만족률, 성공률, 실행시간, 반복 안정성, gap, `optimality_proven` 비율
- F1, precision, recall, confusion matrix는 임계값 기반 보조 해석에만 사용합니다.

## 주요 산출물

`outputs/reports/` 아래에 다음 파일을 생성합니다.

- `aco_pareto_algorithm_evaluation.json`
- `aco_pareto_algorithm_performance.csv`
- `aco_run_history.csv`
- `rcsp_certification_report.json`
- `route_algorithm_summary.md`
- `actual_route_comparison.csv`
- `actual_route_pareto.csv`
- `actual_route_avoided_segments.csv`
- `actual_route_endpoints.json`
- `route_stability_evaluation.json`
- `od_algorithm_evaluation.csv`
- `edge_model_report.json`
- `edge_model_leaderboard.csv`
- `edge_model_validation_predictions.csv`
- `regional_boosting_report.json`
- `regional_boosting_predictions.csv`
- `full_pipeline_report.json`
- `final_model_safety_evaluation.json`
- `final_model_safety_evaluation.md`
- `final_outputs_manifest.json`

## 실행

```bash
pip install -e ".[dev,viz,geo,ml]"
python main.py full-pipeline
python main.py evaluate-algorithm
python main.py clean-artifacts
```

`clean-artifacts`는 `__pycache__/`, `.pytest_cache/`, `*.pyc`, `cmcs_safe_route.egg-info/`, 루트 `outputs.zip`, 이전 Pulse 평가 파일만 제거합니다. 데이터, 모델, 최종 리포트, 지도, 차트는 보존합니다.
