# ACO-Pareto RCSP 기반 우회율 제약 안전 경로 탐색

이 프로젝트의 최종 경로 탐색 알고리즘 이름은 `aco_pareto_rcsp`입니다. 다만 이 이름은 ACO 단독 최적화기를 뜻하지 않습니다. 본 연구의 경로 탐색 알고리즘은 seed 기반 ACO 후보 생성과 Pareto Label-Correcting RCSP 인증을 결합한 하이브리드 구조입니다.

## 알고리즘 해석

ACO는 seed 기반 확률적 후보 경로와 초기 상한을 제공하고, Pareto Label-Correcting RCSP가 전체 그래프 탐색을 완료하는 경우 우회율 제약 하 최적성을 인증합니다.

ACO는 확률적 탐색과 seed connector를 통해 초기 feasible path 및 upper bound를 제공합니다. RCSP는 dominance pruning과 resource constraint를 이용해 전체 그래프에서 최적성 인증을 수행합니다. 따라서 최종 경로의 최적성 주장은 ACO가 아니라 RCSP 인증 결과에 근거합니다. ACO의 역할은 창의적 후보 생성 및 탐색 상한 제공이며, 순수 ACO 성공률과 seed 포함 성공률은 별도로 해석합니다.

현재 실제 도로 그래프 파일럿에서는 pure ACO feasible solution이 제한적이었고, seed connector 기반 후보 생성이 안정적으로 사용되었습니다. 따라서 `aco_run_history.csv`에서 `best_source=seeded_connector`로 기록된 경우, 이는 ACO 단독 최적 경로 발견이 아니라 RCSP 인증을 위한 초기 후보와 upper bound 제공으로 해석해야 합니다.

## 구성 요소

- `AntColonyRouter`: seed connector와 pheromone heuristic을 이용해 후보 경로를 생성합니다.
- `ParetoLabelCorrectingRCSP`: `min C(P)` subject to `L(P) <= B`를 푸는 정확 검증 알고리즘입니다.
- `HybridACOParetoRCSP`: ACO 후보 또는 fallback 후보의 objective를 upper bound로 사용하고, RCSP 결과에 따라 `rcsp_certified`, `rcsp_incumbent`, `aco_approx` 중 최종 경로 출처를 기록합니다.

## 수학적 정의

- 경로 거리: `L(P)=sum(length_e)`
- 경로 위험비용: `C(P)=sum(safety_cost_e)`
- 우회율 제약: `L(P) <= L(P_shortest) * r_age`
- 균형 목적함수: `Score_lambda(P)=lambda L(P)+(1-lambda)C(P)`
- Pareto dominance: 같은 노드에서 `L_A <= L_B`, `C_A <= C_B`이고 하나 이상이 strict이면 B label을 제거합니다.

## 평가 지표

- `pure_aco_success_rate`: seed connector 없이 pure ACO ant가 feasible path를 만든 비율
- `seeded_aco_success_rate`: seed connector 기반 후보 생성 성공률
- `combined_candidate_success_rate`: pure ACO와 seeded 후보를 합친 후보 생성 성공률
- `aco_success_rate`: 기존 호환 필드이며, seeded 포함 성공률입니다.
- `rcsp_certification_rate`: RCSP가 지정된 탐색 범위에서 종료되어 최적성을 인증한 비율
- `gap_pct`: ACO upper bound와 RCSP 결과의 objective 차이

## CMCS 및 보조 모델

LogisticRegression, RandomForest, XGBoost는 경로 탐색 모델이 아닙니다. 이들은 간선별 CMCS 위험도를 보정하고 설명하기 위한 지능형 보조 모델입니다.

- CMCS 보정 모델: Average Precision, ROC-AUC, Brier score
- 경로 탐색 알고리즘: 거리 증가율, CMCS 감소율, 우회율 제약 만족률, 실행시간, 반복 안정성, gap, `optimality_proven` 비율
- F1, precision, recall, confusion matrix는 임계값 기반 보조 해석에만 사용합니다.

## 주요 산출물

`outputs/reports/` 아래에 다음 공식 산출물을 생성합니다.

- `aco_pareto_algorithm_evaluation.json`
- `aco_pareto_algorithm_performance.csv`
- `aco_run_history.csv`
- `rcsp_certification_report.json`
- `route_algorithm_summary.md`
- `actual_route_comparison.csv`
- `actual_route_pareto.csv`
- `actual_route_avoided_segments.csv`
- `actual_route_endpoints.json`
- `risk_zone_map_summary.json`
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

CMCS risk-zone report assets are generated in both static and interactive
formats:

- `outputs/charts/daejeon_cmcs_risk_overview.png`
- `outputs/maps/daejeon_cmcs_risk_overview.html`
- `outputs/charts/pilot_cmcs_risk_zone_map.png`
- `outputs/maps/pilot_cmcs_risk_zone_map.html`

`outputs/debug/route_comparison.csv`와 `outputs/debug/pareto_front.csv`는 데모 그래프용 debug 산출물이며, 공식 manifest에 포함하지 않습니다.

## 실행

```bash
pip install -e ".[dev,viz,geo,ml]"
python main.py full-pipeline
python main.py visualize-risk-zones
python main.py evaluate-algorithm
python main.py evaluate-safety
python main.py clean-artifacts
```

`clean-artifacts`는 `__pycache__/`, `.pytest_cache/`, `*.pyc`, `cmcs_safe_route.egg-info/`, 루트 `outputs.zip`, 이전 Pulse 평가 파일, 공식 reports에 남은 데모 CSV만 제거합니다. 데이터, 모델, 최종 리포트, 지도, 차트는 보존합니다.
