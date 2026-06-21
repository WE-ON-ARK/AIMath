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
pytest
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
7. 검증 성능에 따라 AHP와 ML 위험 확률을 자동 혼합
8. 대전문정초등학교에서 실제 둔산 지역 학원까지 최단·안전·균형 경로 산출
9. 경로 비교 CSV, 파레토 데이터, HTML 지도, 모델·종합 리포트 저장

주요 결과:

```text
data/processed/daejeon_edge_features.csv
data/processed/daejeon_edge_cmcs.csv
data/graph/daejeon_walk_cmcs.graphml
models/edge_accident_risk_model.pkl
outputs/maps/actual_safe_route.html
outputs/maps/daejeon_cmcs_risk_map.html
outputs/reports/actual_route_comparison.csv
outputs/reports/actual_route_pareto.csv
outputs/reports/actual_route_avoided_segments.csv
outputs/reports/edge_model_report.json
outputs/reports/full_pipeline_report.json
outputs/charts/edge_model_roc_pr.png
outputs/charts/edge_model_explainability.png
outputs/charts/actual_route_pareto.html
outputs/charts/district_safety_radar.html
```

데이터와 도로망을 다시 내려받으려면 다음 플래그를 사용합니다.

```bash
python main.py full-pipeline --refresh-data --refresh-network
```

## 실제 데이터 연동

전체 지리·ML 의존성을 설치하고 환경 변수를 설정합니다.

```bash
pip install -r requirements.txt
cp .env.example .env
export DATA_GO_KR_API_KEY="발급받은_서비스키"
python main.py collect --year 2023
python main.py download-network
```

공공데이터포털 API는 활용신청한 데이터셋마다 엔드포인트와 응답 컬럼이
다를 수 있습니다. `src/data_collector.py`의 `APIEndpoint`를 이용해 승인된
API 명세를 등록한 뒤, 좌표 컬럼명을 확인하여 `Preprocessor.unify_crs`에
전달합니다.

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
  data_collector.py
  preprocessor.py
  cmcs_calculator.py
  model_trainer.py
  route_optimizer.py
  visualizer.py
  demo.py
tests/
main.py
config.py
```

## 다음 개발 단계

1. 승인받은 공공데이터 API별 엔드포인트와 좌표 컬럼 매핑
2. 대전 OSM 보행망과 실데이터 공간 조인
3. 학습/검증 데이터 누수 점검 및 공간 교차검증
4. 학교–학원 실제 OD 쌍의 경로 평가
5. 가중치 민감도·Ablation·구별 안전 격차 리포트
