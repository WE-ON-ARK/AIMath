# CMCS 안전 통학 경로 추천 AI

대전광역시 도로 구간별 어린이 이동 제약 점수(CMCS)를 계산하고, 거리와
위험 노출량을 함께 고려해 안전 통학 경로를 추천하는 프로젝트입니다.

현재 버전은 다음을 포함하는 실행 가능한 MVP입니다.

- AHP 기반 CMCS 점수 및 세부 차원 산출
- 공공데이터포털 JSON 페이지네이션 수집기
- GeoPandas/OSMnx 기반 포인트–도로 간선 전처리
- 회귀·위험 등급 분류 모델 비교와 평가 리포트
- 최단거리·최저위험·균형 경로와 파레토 프론트
- Folium 경로/CMCS 지도와 Plotly 파레토 차트
- API 키 없이 실행되는 합성 대전 도로망 데모

## 빠른 시작

Python 3.10 이상을 사용합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"
python main.py demo
pytest  # 테스트 75개 실행
```

데모 산출물은 다음 위치에 생성됩니다.

```text
data/processed/edge_cmcs.csv
data/graph/demo_walk.graphml
outputs/maps/cmcs_heatmap.html
outputs/maps/route_comparison.html
outputs/charts/pareto_front.html
outputs/reports/route_comparison.csv
outputs/reports/pareto_front.csv
```

## 병합된 실데이터 학습

`data` 브랜치의 대전 학교·횡단보도·신호등·학원·불법주정차 자료와
공공데이터 API에서 받은 어린이보호구역 사고 다발지역 CSV를 사용합니다.

```bash
python main.py real-data
```

산출물:

```text
data/processed/daejeon_school_risk_features.csv
models/real_school_hotspot_model.pkl
outputs/reports/real_data_model_report.json
outputs/reports/school_hotspot_predictions.csv
outputs/reports/verified_accident_locations.csv
outputs/reports/verified_accident_summary.json
outputs/maps/verified_accident_hotspots.html
outputs/charts/real_model_feature_importance.png
```

라벨은 개별 사고 전체가 아니라 한국도로교통공단 API가 지정한
`어린이보호구역 내 어린이 교통사고 다발지역`입니다. 따라서 음성 라벨은
무사고를 뜻하지 않으며, 모델 평가는 구 단위 Leave-One-Group-Out 방식으로
공간 누수를 줄여 수행합니다.

## 전체 실제 경로 파이프라인

다음 명령은 공식 API·OSM·병합 데이터로 아래 과정을 순차 실행합니다.

```bash
python main.py full-pipeline
```

1. 과속방지턱과 어린이보호구역 공식 API 수집
2. 대전 전체 OSM 보행 도로망 구축 또는 캐시 로드
3. 횡단보도·신호등·과속방지턱·버스정류장·보호구역·사고를 도로 구간에 결합
4. OSM 도로 등급, 제한속도, 차로, 보도, 조명 태그 기반 프록시 생성
5. AHP CMCS 계산
6. 2km 공간 그룹 교차검증을 적용한 도로 사고 위험 모델 학습
7. LogisticRegression 기준선과 RandomForest·XGBoost를 동일 Fold로 비교
8. OOF Average Precision 우선, ROC-AUC 차순으로 RF/XGBoost 최종 모델 선정
9. OOF F1 최적 임계값과 고정 0.5 임계값 성능을 함께 저장
10. 자치구 경계로 분리한 1.5km 통학 권역 XGBoost를 구별 홀드아웃으로 검증
11. 권역 OOF F1 0.5 이상일 때 권역 위험 확률을 도로 CMCS에 혼합
12. 대전문정초등학교에서 실제 둔산 지역 학원까지 최단·안전·균형 경로 산출
13. 경로 비교 CSV, 파레토 데이터, HTML 지도, 모델·종합 리포트 저장

F1 0.5 목표는 희소한 사고 다발지역 라벨을 그대로 복제하거나 음성 표본을
임의 축소하지 않고, 1.5km 통학 위험 권역 분류 문제로 계층화해 평가합니다.
보고서에는 도로 단위 모델과 권역 단위 XGBoost 지표를 분리해 저장합니다.

주요 결과:

```text
data/processed/daejeon_edge_features.csv
data/processed/daejeon_edge_cmcs.csv
data/graph/daejeon_walk_cmcs.graphml
models/edge_accident_risk_model.pkl
models/regional_xgboost_risk_model.pkl
outputs/maps/actual_safe_route.html
outputs/maps/daejeon_cmcs_risk_map.html
outputs/reports/actual_route_comparison.csv
outputs/reports/actual_route_pareto.csv
outputs/reports/actual_route_avoided_segments.csv
outputs/reports/edge_model_report.json
outputs/reports/edge_model_leaderboard.csv
outputs/reports/edge_model_validation_predictions.csv
outputs/reports/regional_boosting_report.json
outputs/reports/regional_boosting_predictions.csv
outputs/reports/full_pipeline_report.json
outputs/charts/edge_model_roc_pr.png
outputs/charts/edge_model_explainability.png
outputs/charts/regional_boosting_roc_pr.png
outputs/charts/regional_boosting_shap.png
outputs/charts/actual_route_pareto.html
outputs/charts/district_safety_radar.html
```

데이터와 도로망을 다시 내려받으려면 다음 플래그를 사용합니다.

```bash
python main.py full-pipeline --refresh-data --refresh-network
```

도로 피처가 이미 생성된 상태에서 모델 비교·학습만 다시 실행할 수도 있습니다.

```bash
python main.py train-edge-models
```

## 실제 데이터 연동

전체 지리·ML 의존성을 설치하고 환경 변수를 설정합니다.

```bash
pip install -r requirements.txt
cp .env.example .env
export DATA_GO_KR_API_KEY="발급받은_서비스키"
export KAKAO_API_KEY="발급받은_카카오_REST_API_키"
python main.py collect --year 2023
python main.py download-network
```

공공데이터포털 API는 활용신청한 데이터셋마다 엔드포인트와 응답 컬럼이
다를 수 있습니다. `src/data_collector.py`의 `APIEndpoint`를 이용해 승인된
API 명세를 등록한 뒤, 좌표 컬럼명을 확인하여 `Preprocessor.unify_crs`에
전달합니다.

`KAKAO_API_KEY`는 `src/geocoder.py`의 `KakaoGeocoder`가 사용합니다.
지오코딩 결과는 파일 캐시에 저장되어 재요청을 줄입니다.

## API 서비스

FastAPI 기반 경로 추천 서비스입니다.

```bash
uvicorn api.main:app --reload
```

브라우저에서 `static/index.html`을 열면 Leaflet.js 지도 기반 웹 UI를
사용할 수 있습니다.

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/route/recommend` | 출발지·목적지 기반 안전 경로 추천 |
| POST | `/route/compare` | 복수 경로 비교 (최단·최저위험·균형) |
| GET | `/search/schools` | 학교 목록 검색 |
| GET | `/search/academies` | 학원 목록 검색 |
| GET | `/health` | 서비스 상태 확인 |
| GET | `/metrics` | 운영 지표 조회 |

## Docker 배포

Docker Compose로 전체 서비스를 실행합니다.

```bash
# 기본 서비스 실행
docker compose up

# 데이터 갱신 스케줄러 포함 실행
docker compose --profile scheduler up
```

주요 환경 변수:

| 변수 | 설명 |
|------|------|
| `DATA_GO_KR_API_KEY` | 공공데이터포털 서비스키 |
| `KAKAO_API_KEY` | 카카오 REST API 키 (지오코딩) |
| `CMCS_GRAPH_PATH` | 사전 구축된 GraphML 경로 (선택) |

## 핵심 설계

CMCS는 위험도, 보행불편도, 혼잡도, 시야방해도, 횡단위험도를 합산하고
과속방지턱·CCTV·어린이보호구역을 안전 보너스로 차감합니다. 모든 점수는
0~1 범위로 제한됩니다.

안전 경로 비용은 단순한 구간 CMCS 합이 아니라 다음 위험 노출량을
기본으로 사용합니다.

```text
risk_exposure = length_m × CMCS
safety_cost = length_m × (risk_floor + CMCS)
```

`risk_floor`는 CMCS가 0인 구간을 무제한으로 우회하는 현상을 막습니다.
멀티그래프의 병렬 간선도 실제 최적화에 선택된 간선 키를 유지해 거리와
위험 합계를 계산합니다.

## 프로젝트 구조

```text
src/
  data_collector.py       # 공공데이터포털 JSON 페이지네이션 수집
  preprocessor.py         # GeoPandas/OSMnx 포인트–도로 간선 전처리
  cmcs_calculator.py      # AHP 기반 CMCS 점수 계산
  model_trainer.py        # 회귀·분류 모델 학습 및 비교
  route_optimizer.py      # 최단·최저위험·균형 경로와 파레토 프론트
  visualizer.py           # Folium 지도 및 Plotly 차트 생성
  demo.py                 # 합성 대전 도로망 데모
  data_quality.py         # 좌표 bbox·결측·중복 검사, DATA_VINTAGE 정의
  geocoder.py             # KakaoGeocoder (파일 캐시, KAKAO_API_KEY 필요)
  model_validation.py     # CMCS 민감도, Ablation, 보정, 임계값, 시계열 분할, 구별 홀드아웃
  route_validator.py      # AgeProfile(low/mid/high/middle), TimeWeights, batch_od_evaluation
api/
  main.py                 # FastAPI 앱 및 라우터
static/
  index.html              # Leaflet.js 기반 웹 UI
tests/                    # 테스트 75개
main.py
config.py
```

## 모델 검증

`src/model_validation.py`는 다음 6종의 검증을 자동 실행합니다.

| 검증 항목 | 내용 |
|-----------|------|
| CMCS 민감도 | AHP 가중치 ±20% 교란 → Spearman ρ, Tier 변동율 |
| Ablation Test | LeaveOneGroupOut AUC 기반 특성별 기여도 측정 |
| 보정 분석 | Reliability Diagram + Brier Score |
| 임계값 최적화 | F1 최대화 기준 0.05–0.95 구간 탐색 |
| 시계열 분할 | 연도별 Train/Test 분리 검증 |
| 구별 홀드아웃 | 5개 구 Leave-One-Out 요약 |

```bash
python -c "from src.model_validation import run_full_model_validation; run_full_model_validation()"
```

최종 모델의 성능, 확률 보정, 공간·시간 일반화, 데이터 한계와 운영 적용
범위를 한 번에 평가하려면 다음 명령을 실행합니다.

```bash
python main.py evaluate-safety
```

주요 산출물:

```text
outputs/reports/final_model_safety_evaluation.html
outputs/reports/final_model_safety_evaluation.md
outputs/reports/final_model_performance_metrics.csv
outputs/reports/final_model_safety_matrix.csv
outputs/charts/final_model_safety_dashboard.png
```

## 다음 개발 단계

- 공공데이터 API와 OSM 원천 데이터의 정기 갱신 자동화
- 가로등 실측 데이터 확보 (현재 OSM 희박 — 7개 포인트)
- 사고 이력 데이터 보강 (현재 22건 → 통계 신뢰도 제한)
- 클라우드(GCP/AWS) 배포 및 모바일 UI 연동
