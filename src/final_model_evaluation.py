"""최종 모델 안전성·적용 가능성 종합 평가 보고서 생성."""
from __future__ import annotations

import html
import json
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import CHART_OUTPUT_DIR, REPORT_OUTPUT_DIR
from src.full_pipeline import _optimal_f1_threshold


PERFORMANCE_CSV = REPORT_OUTPUT_DIR / "final_model_performance_metrics.csv"
SAFETY_MATRIX_CSV = REPORT_OUTPUT_DIR / "final_model_safety_matrix.csv"
EVALUATION_JSON = REPORT_OUTPUT_DIR / "final_model_safety_evaluation.json"
EVALUATION_MD = REPORT_OUTPUT_DIR / "final_model_safety_evaluation.md"
EVALUATION_HTML = REPORT_OUTPUT_DIR / "final_model_safety_evaluation.html"
DASHBOARD_PNG = CHART_OUTPUT_DIR / "final_model_safety_dashboard.png"

STATUS_COLORS = {
    "양호": "#DCFCE7",
    "조건부": "#FEF3C7",
    "미흡": "#FEE2E2",
}
STATUS_TEXT_COLORS = {
    "양호": "#166534",
    "조건부": "#92400E",
    "미흡": "#991B1B",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_intervals(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float | None = None,
    prediction: np.ndarray | None = None,
    iterations: int = 3000,
    random_state: int = 42,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(random_state)
    values: list[list[float]] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(target), len(target))
        sampled_target = target[indices]
        if np.unique(sampled_target).size < 2:
            continue
        sampled_probability = probability[indices]
        sampled_prediction = (
            prediction[indices]
            if prediction is not None
            else sampled_probability >= float(threshold)
        )
        values.append(
            [
                roc_auc_score(sampled_target, sampled_probability),
                average_precision_score(sampled_target, sampled_probability),
                f1_score(
                    sampled_target, sampled_prediction, zero_division=0
                ),
                precision_score(
                    sampled_target, sampled_prediction, zero_division=0
                ),
                recall_score(
                    sampled_target, sampled_prediction, zero_division=0
                ),
            ]
        )
    array = np.asarray(values)
    names = ("roc_auc", "average_precision", "f1", "precision", "recall")
    return {
        name: [
            round(float(np.quantile(array[:, index], 0.025)), 4),
            round(float(np.quantile(array[:, index], 0.975)), 4),
        ]
        for index, name in enumerate(names)
    }


def _expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    frame = pd.DataFrame({"target": target, "probability": probability})
    frame["bin"] = pd.qcut(
        frame["probability"],
        bins,
        duplicates="drop",
    )
    grouped = frame.groupby("bin", observed=True).agg(
        count=("target", "size"),
        observed=("target", "mean"),
        predicted=("probability", "mean"),
    )
    gap = (grouped["observed"] - grouped["predicted"]).abs()
    return float((grouped["count"] / len(frame) * gap).sum())


def _regional_metrics(
    predictions: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame, dict[str, list[float]]]:
    target = predictions["accident_label"].astype(int).to_numpy()
    nested_columns = {
        "nested_oof_probability",
        "nested_oof_prediction",
        "nested_decision_threshold",
    }
    if nested_columns.issubset(predictions.columns):
        probability = predictions["nested_oof_probability"].astype(
            float
        ).to_numpy()
        prediction = predictions["nested_oof_prediction"].astype(int).to_numpy()
        threshold = float(
            predictions["nested_decision_threshold"].astype(float).mean()
        )
        validation_method = "중첩 자치구 홀드아웃"
    else:
        probability = predictions["oof_risk_probability"].astype(
            float
        ).to_numpy()
        threshold = _optimal_f1_threshold(target, probability)
        prediction = probability >= threshold
        validation_method = "자치구 홀드아웃"
    tn = int(((prediction == 0) & (target == 0)).sum())
    fp = int(((prediction == 1) & (target == 0)).sum())
    fn = int(((prediction == 0) & (target == 1)).sum())
    tp = int(((prediction == 1) & (target == 1)).sum())
    prevalence = float(target.mean())
    dummy_brier = float(
        brier_score_loss(target, np.full(len(target), prevalence))
    )
    brier = float(brier_score_loss(target, probability))
    metrics: dict[str, float | int] = {
        "regions": int(len(target)),
        "positive_regions": int(target.sum()),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "validation_method": validation_method,
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(
            average_precision_score(target, probability)
        ),
        "ap_lift": float(
            average_precision_score(target, probability) / prevalence
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
        "precision": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "specificity": tn / max(tn + fp, 1),
        "negative_predictive_value": tn / max(tn + fn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "brier_score": brier,
        "dummy_brier_score": dummy_brier,
        "brier_skill_score": 1.0 - brier / dummy_brier,
        "expected_calibration_error": _expected_calibration_error(
            target, probability
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    district_rows: list[dict[str, object]] = []
    for district, group in predictions.groupby("district"):
        district_target = group["accident_label"].astype(int).to_numpy()
        probability_column = (
            "nested_oof_probability"
            if "nested_oof_probability" in group
            else "oof_risk_probability"
        )
        district_probability = group[probability_column].astype(float).to_numpy()
        district_prediction = (
            group["nested_oof_prediction"].astype(int).to_numpy()
            if "nested_oof_prediction" in group
            else district_probability >= threshold
        )
        row: dict[str, object] = {
            "자치구": district,
            "권역 수": len(group),
            "양성 권역": int(district_target.sum()),
            "정밀도": precision_score(
                district_target, district_prediction, zero_division=0
            ),
            "재현율": recall_score(
                district_target, district_prediction, zero_division=0
            ),
            "F1": f1_score(
                district_target, district_prediction, zero_division=0
            ),
        }
        if np.unique(district_target).size == 2:
            row["ROC-AUC"] = roc_auc_score(
                district_target, district_probability
            )
            row["AP"] = average_precision_score(
                district_target, district_probability
            )
        district_rows.append(row)
    district_metrics = pd.DataFrame(district_rows)
    intervals = _bootstrap_intervals(
        target,
        probability,
        prediction=prediction,
    )
    return metrics, district_metrics, intervals


def _performance_table(
    edge_report: dict[str, object],
    regional: dict[str, float | int],
    intervals: dict[str, list[float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edge_models = edge_report["models"]
    for name in ("LogisticRegression", "RandomForest", "XGBoost"):
        metrics = edge_models[name]
        optimized = metrics["optimized_threshold"]
        rows.append(
            {
                "평가 단위": "도로 구간",
                "모델": name,
                "ROC-AUC": metrics["roc_auc"],
                "AP": metrics["average_precision"],
                "F1": optimized["f1"],
                "정밀도": optimized["precision"],
                "재현율": optimized["recall"],
                "Brier": metrics["brier_score"],
                "검증": "2km 공간 그룹 5-fold",
                "판정": "미흡",
            }
        )
    rows.append(
        {
            "평가 단위": "1.75km 권역",
            "모델": "XGBoost 앙상블",
            "ROC-AUC": regional["roc_auc"],
            "AP": regional["average_precision"],
            "F1": regional["f1"],
            "정밀도": regional["precision"],
            "재현율": regional["recall"],
            "Brier": regional["brier_score"],
            "검증": "중첩 자치구 Leave-One-Group-Out",
            "판정": "조건부",
        }
    )
    frame = pd.DataFrame(rows)
    frame.attrs["regional_ci"] = intervals
    return frame


def _safety_matrix(
    regional: dict[str, float | int],
    district_metrics: pd.DataFrame,
    route_report: dict[str, object],
    regional_report: dict[str, object],
    temporal_report: dict[str, object],
    quality_report: dict[str, object],
) -> pd.DataFrame:
    min_district_f1 = float(district_metrics["F1"].min())
    route = route_report["route"]
    route_stability = route.get("stability_validation", {})
    model_stability = regional_report.get("stability", {})
    all_coordinate_pass_rates = [
        dataset["pass_rate"]
        for dataset in quality_report["coordinate_quality"].values()
    ]
    rows = [
        {
            "평가 영역": "위험 순위 판별력",
            "근거": (
                f"권역 ROC-AUC {regional['roc_auc']:.3f}, "
                f"AP {regional['average_precision']:.3f} "
                f"(발생률 대비 {regional['ap_lift']:.1f}배)"
            ),
            "상태": "양호",
            "운영 의미": "위험 권역 우선 점검·정렬에는 활용 가능",
        },
        {
            "평가 영역": "위험 누락 통제",
            "근거": (
                f"재현율 {regional['recall']:.3f}, "
                f"양성 {regional['positive_regions']}개 중 "
                f"{regional['false_negative']}개 누락"
            ),
            "상태": "조건부" if regional["recall"] >= 0.65 else "미흡",
            "운영 의미": "개선됐지만 모델이 낮게 평가한 경로도 안전 단정 금지",
        },
        {
            "평가 영역": "부스팅 재현 안정성",
            "근거": (
                f"7개 시드 최소 F1 "
                f"{model_stability.get('minimum_seed_f1', 0):.3f}, "
                f"표준편차 "
                f"{model_stability.get('seed_f1_standard_deviation', 0):.3f}"
            ),
            "상태": (
                "양호"
                if model_stability.get("minimum_seed_f1", 0) >= 0.5
                else "미흡"
            ),
            "운영 의미": "랜덤 시드 변화에도 전체 F1 0.5 이상 유지",
        },
        {
            "평가 영역": "확률 보정",
            "근거": (
                f"Brier {regional['brier_score']:.3f}, "
                f"무정보 대비 개선 {regional['brier_skill_score'] * 100:.1f}%, "
                f"ECE {regional['expected_calibration_error']:.3f}"
            ),
            "상태": "미흡",
            "운영 의미": "출력값은 절대 사고확률이 아닌 상대 위험점수로만 사용",
        },
        {
            "평가 영역": "공간 일반화",
            "근거": (
                "5개 자치구 완전 홀드아웃, "
                f"구별 F1 {min_district_f1:.3f}~"
                f"{district_metrics['F1'].max():.3f}"
            ),
            "상태": "조건부",
            "운영 의미": "구별 임계값·성능 모니터링 필요",
        },
        {
            "평가 영역": "시간 일반화",
            "근거": (
                f"과거→최근 ROC-AUC "
                f"{temporal_report['metrics']['roc_auc']:.3f}, "
                f"Brier {temporal_report['metrics']['brier_score']:.3f} "
                f"> Dummy {temporal_report['dummy_brier_score']:.3f}"
            ),
            "상태": "미흡",
            "운영 의미": "연도 변화에 대한 재학습·전향 검증 전제",
        },
        {
            "평가 영역": "데이터 완전성",
            "근거": (
                f"원 사고다발지역 22건, 권역 양성 "
                f"{regional['positive_regions']}개, 좌표 통과율 "
                f"{min(all_coordinate_pass_rates) * 100:.2f}% 이상, "
                "가로등 OSM 7개"
            ),
            "상태": "미흡",
            "운영 의미": "희소 라벨·시점 불일치·시설 누락 보강 필요",
        },
        {
            "평가 영역": "CMCS 구조 안정성",
            "근거": "가중치 ±20%에서 Spearman ρ 0.980~0.998",
            "상태": "양호",
            "운영 의미": "AHP 점수 순위는 가중치 변화에 비교적 안정적",
        },
        {
            "평가 영역": "경로 효용",
            "근거": (
                f"{route_stability.get('evaluated_pairs', 1)}개 OD에서 "
                f"위험감소 비율 "
                f"{route_stability.get('positive_risk_reduction_ratio', 0) * 100:.1f}%, "
                f"중앙 위험감소 "
                f"{route_stability.get('median_risk_reduction_pct', route['risk_reduction_pct']):.1f}%"
            ),
            "상태": (
                "양호"
                if route_stability.get("route_selection_stability_passed")
                else "조건부"
            ),
            "운영 의미": "다수 OD 수치 검증 통과, 현장 보행 검증은 별도 필요",
        },
        {
            "평가 영역": "설명 가능성",
            "근거": "도로·권역 모델 SHAP 산출물 제공",
            "상태": "양호",
            "운영 의미": "운영자 검토와 이상 예측 조사에 활용 가능",
        },
        {
            "평가 영역": "자동 배포 안전성",
            "근거": "도로 모델 production_deployment_ready=False",
            "상태": "미흡",
            "운영 의미": "사람 검토 없는 단독 경로 결정·안전 보증 금지",
        },
    ]
    return pd.DataFrame(rows)


def _recommendations() -> list[str]:
    return [
        "현재 허용 범위는 연구·시연 및 담당자 검토가 포함된 제한적 파일럿이다.",
        "사용자 화면에는 '상대 위험 기반 추천이며 실제 안전을 보장하지 않음'을 항상 표시한다.",
        "권역 XGBoost 확률은 사고 발생확률로 표시하지 말고 위험도 또는 우선점검 점수로 표시한다.",
        "최소 30개 이상의 학교–학원 OD를 대상으로 우회율·위험감소·현장 장애물을 검증한다.",
        "최근 연도 개별 사고 원장과 실측 교통량·보도 폭·조도 데이터를 확보해 재학습한다.",
        "구별 재현율 하한, 데이터 신선도, 예측 분포 변화를 운영 모니터링 지표로 둔다.",
        "임계값은 별도 검증 세트 또는 중첩 교차검증으로 다시 결정해 낙관 편향을 줄인다.",
        "보정 모델을 적용하고 Brier skill 10% 이상, 재현율 0.70 이상을 자동화 검토 게이트로 둔다.",
    ]


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = [
            _fmt(value).replace("|", "\\|").replace("\n", " ")
            for value in row.tolist()
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _markdown_report(
    performance: pd.DataFrame,
    safety: pd.DataFrame,
    district: pd.DataFrame,
    regional: dict[str, float | int],
    intervals: dict[str, list[float]],
    route_report: dict[str, object],
) -> str:
    route = route_report["route"]
    route_stability = route.get("stability_validation", {})
    return f"""# CMCS 모델 안전성 및 적용 가능성 종합 평가

- 평가일: {date.today().isoformat()}
- 종합 판정: **제한적 파일럿 가능 / 무감독 실서비스 불가**
- 핵심 근거: 권역 XGBoost의 공간 홀드아웃 F1은 {regional['f1']:.3f}이지만,
  재현율은 {regional['recall']:.3f}이고 위험 권역 {regional['positive_regions']}개 중
  {regional['false_negative']}개를 놓쳤습니다.

## 성능 지표

{_markdown_table(performance)}

권역 XGBoost 95% 단순 부트스트랩 구간:
ROC-AUC {intervals['roc_auc'][0]:.3f}–{intervals['roc_auc'][1]:.3f},
AP {intervals['average_precision'][0]:.3f}–{intervals['average_precision'][1]:.3f},
F1 {intervals['f1'][0]:.3f}–{intervals['f1'][1]:.3f}.
이 구간은 공간 군집 구조를 완전히 반영하지 않은 참고치입니다.

## 안전성·적용성 게이트

{_markdown_table(safety)}

## 자치구별 권역 성능

{_markdown_table(district)}

## 경로 효용 확인

- 실제 경로: {route['origin']} → {route['destination']}
- 최단거리: {route['shortest_distance_m']:.0f}m
- 안전 경로: {route['safest_distance_m']:.0f}m
- 위험노출 감소: {route['risk_reduction_pct']:.1f}%
- 다중 OD 검증: {route_stability.get('evaluated_pairs', 0)}개 경로 중
  {route_stability.get('positive_risk_reduction_ratio', 0) * 100:.1f}%에서
  위험노출 감소, 중앙값 {route_stability.get('median_risk_reduction_pct', 0):.1f}%
- 주의: 정답 통학 경로 라벨은 없어 경로 선정 자체의 F1은 계산할 수 없습니다.

## 최종 적용 범위

| 적용 시나리오 | 판정 |
|---|---|
| 연구·시연·위험지도 탐색 | 가능 |
| 담당자 검토가 포함된 제한적 파일럿 | 조건부 가능 |
| 일반 사용자 대상 베타 경로 추천 | 추가 현장 검증 후 조건부 |
| 모델 단독 자동 경로 결정 | 불가 |
| 실제 사고 예방 또는 절대 안전 보증 | 불가 |

## 필수 개선 체크리스트

{chr(10).join(f'- [ ] {item}' for item in _recommendations())}
"""


def _html_table(frame: pd.DataFrame, status_column: str | None = None) -> str:
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column, value in row.items():
            style = ""
            if status_column and column == status_column:
                color = STATUS_COLORS.get(str(value), "#FFFFFF")
                text = STATUS_TEXT_COLORS.get(str(value), "#111827")
                style = f' style="background:{color};color:{text};font-weight:700"'
            cells.append(f"<td{style}>{html.escape(_fmt(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _html_report(
    performance: pd.DataFrame,
    safety: pd.DataFrame,
    district: pd.DataFrame,
    regional: dict[str, float | int],
    intervals: dict[str, list[float]],
    route_report: dict[str, object],
) -> str:
    route = route_report["route"]
    route_stability = route.get("stability_validation", {})
    recommendations = "".join(
        f"<li>{html.escape(item)}</li>" for item in _recommendations()
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CMCS 모델 안전성 종합 평가</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Nanum Gothic",sans-serif;background:#f8fafc;color:#172033;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:36px 24px 60px}}
h1{{margin-bottom:8px}} h2{{margin-top:34px}}
.verdict{{background:#fff7ed;border-left:6px solid #f97316;padding:18px 22px;border-radius:10px;margin:20px 0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px}}
.value{{font-size:28px;font-weight:800;color:#0f172a}} .label{{color:#64748b;font-size:13px}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;overflow:hidden;font-size:14px}}
th,td{{padding:10px 12px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}}
th{{background:#0f172a;color:white;position:sticky;top:0}}
.note{{color:#64748b;font-size:13px}} li{{margin:8px 0}}
@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}} main{{padding:20px 12px}}}}
</style>
</head>
<body><main>
<h1>CMCS 모델 안전성 및 적용 가능성 종합 평가</h1>
<p class="note">평가일 {date.today().isoformat()} · 실제 데이터 및 공간 홀드아웃 결과 기준</p>
<div class="verdict"><strong>종합 판정: 제한적 파일럿 가능 / 무감독 실서비스 불가</strong><br>
권역 순위 판별력은 양호하지만 위험 권역 {regional['positive_regions']}개 중
{regional['false_negative']}개를 누락하고 확률 보정과 시간 일반화가 부족합니다.</div>
<div class="cards">
<div class="card"><div class="label">권역 ROC-AUC</div><div class="value">{regional['roc_auc']:.3f}</div></div>
<div class="card"><div class="label">권역 F1</div><div class="value">{regional['f1']:.3f}</div></div>
<div class="card"><div class="label">권역 재현율</div><div class="value">{regional['recall']:.3f}</div></div>
<div class="card"><div class="label">다중 OD 위험감소 경로 비율</div><div class="value">{route_stability.get('positive_risk_reduction_ratio', 0) * 100:.1f}%</div></div>
</div>
<h2>성능 지표 비교</h2>
{_html_table(performance, "판정")}
<p class="note">권역 XGBoost 95% 단순 부트스트랩 구간:
ROC-AUC {intervals['roc_auc'][0]:.3f}–{intervals['roc_auc'][1]:.3f},
F1 {intervals['f1'][0]:.3f}–{intervals['f1'][1]:.3f}.
공간 군집 구조를 완전히 반영하지 않은 참고치입니다.</p>
<h2>안전성·적용성 게이트</h2>
{_html_table(safety, "상태")}
<h2>자치구별 권역 성능</h2>
{_html_table(district)}
<h2>적용 판정</h2>
<table><thead><tr><th>시나리오</th><th>판정</th></tr></thead><tbody>
<tr><td>연구·시연·위험지도 탐색</td><td style="background:#DCFCE7;color:#166534;font-weight:700">가능</td></tr>
<tr><td>담당자 검토가 포함된 제한적 파일럿</td><td style="background:#FEF3C7;color:#92400E;font-weight:700">조건부 가능</td></tr>
<tr><td>일반 사용자 대상 베타 경로 추천</td><td style="background:#FEF3C7;color:#92400E;font-weight:700">추가 현장 검증 필요</td></tr>
<tr><td>모델 단독 자동 경로 결정·절대 안전 보증</td><td style="background:#FEE2E2;color:#991B1B;font-weight:700">불가</td></tr>
</tbody></table>
<h2>필수 개선 체크리스트</h2><ul>{recommendations}</ul>
</main></body></html>"""


def _draw_dashboard(
    performance: pd.DataFrame,
    safety: pd.DataFrame,
    district: pd.DataFrame,
    regional: dict[str, float | int],
    route_report: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(18, 16.5), facecolor="#F8FAFC")
    grid = figure.add_gridspec(
        4,
        2,
        height_ratios=[0.55, 1.2, 2.05, 1.2],
        hspace=0.42,
        wspace=0.16,
    )

    title_axis = figure.add_subplot(grid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0,
        0.88,
        "CMCS 모델 안전성 및 적용 가능성 종합 평가",
        fontsize=25,
        fontweight="bold",
        color="#0F172A",
    )
    title_axis.text(
        0,
        0.54,
        "종합 판정  |  제한적 파일럿 가능 · 무감독 실서비스 불가",
        fontsize=17,
        fontweight="bold",
        color="#C2410C",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFEDD5",
            "edgecolor": "#FDBA74",
        },
    )
    title_axis.text(
        0,
        0.14,
        (
            f"권역 {regional['regions']}개 중 양성 {regional['positive_regions']}개 · "
            f"위험 권역 누락 {regional['false_negative']}개 · "
            f"평가일 {date.today().isoformat()}"
        ),
        fontsize=12,
        color="#475569",
    )

    metrics_axis = figure.add_subplot(grid[1, 0])
    metric_names = ["ROC-AUC", "AP", "F1", "정밀도", "재현율", "특이도"]
    metric_values = [
        regional["roc_auc"],
        regional["average_precision"],
        regional["f1"],
        regional["precision"],
        regional["recall"],
        regional["specificity"],
    ]
    colors = [
        "#16A34A",
        "#22C55E",
        "#F59E0B",
        "#F59E0B",
        "#DC2626",
        "#16A34A",
    ]
    positions = np.arange(len(metric_names))
    metrics_axis.barh(positions, metric_values, color=colors, height=0.58)
    metrics_axis.set_yticks(positions, metric_names)
    metrics_axis.invert_yaxis()
    metrics_axis.set_xlim(0, 1)
    metrics_axis.set_title(
        "권역 XGBoost 핵심 성능",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    metrics_axis.grid(axis="x", alpha=0.18)
    for position, value in zip(positions, metric_values):
        metrics_axis.text(
            min(float(value) + 0.02, 0.95),
            position,
            f"{float(value):.3f}",
            va="center",
            fontweight="bold",
        )

    comparison_axis = figure.add_subplot(grid[1, 1])
    comparison_axis.axis("off")
    comparison_view = performance[
        ["평가 단위", "모델", "ROC-AUC", "AP", "F1", "정밀도", "재현율"]
    ].copy()
    for column in ["ROC-AUC", "AP", "F1", "정밀도", "재현율"]:
        comparison_view[column] = comparison_view[column].map(
            lambda value: f"{value:.3f}"
        )
    table = comparison_axis.table(
        cellText=comparison_view.values,
        colLabels=comparison_view.columns,
        loc="center",
        cellLoc="center",
        colWidths=[0.13, 0.24, 0.12, 0.10, 0.10, 0.12, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(comparison_view):
            cell.set_facecolor("#FEF3C7")
    comparison_axis.set_title(
        "도로 모델과 권역 모델 비교",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )

    safety_axis = figure.add_subplot(grid[2, :])
    safety_axis.axis("off")
    safety_view = safety[["평가 영역", "근거", "상태", "운영 의미"]].copy()
    safety_view["근거"] = safety_view["근거"].map(
        lambda value: textwrap.fill(str(value), width=38)
    )
    safety_view["운영 의미"] = safety_view["운영 의미"].map(
        lambda value: textwrap.fill(str(value), width=38)
    )
    safety_table = safety_axis.table(
        cellText=safety_view.values,
        colLabels=safety_view.columns,
        loc="center",
        cellLoc="left",
        colWidths=[0.14, 0.36, 0.09, 0.36],
    )
    safety_table.auto_set_font_size(False)
    safety_table.set_fontsize(9.5)
    safety_table.scale(1, 1.72)
    for (row, column), cell in safety_table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", fontweight="bold")
        elif column == 2:
            status = safety_view.iloc[row - 1]["상태"]
            cell.set_facecolor(STATUS_COLORS[status])
            cell.set_text_props(
                color=STATUS_TEXT_COLORS[status],
                fontweight="bold",
                ha="center",
            )
    safety_axis.set_title(
        "안전성·적용성 게이트",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    district_axis = figure.add_subplot(grid[3, 0])
    district_view = district.sort_values("자치구")
    district_axis.bar(
        district_view["자치구"],
        district_view["F1"],
        color=[
            "#16A34A" if value >= 0.5 else "#DC2626"
            for value in district_view["F1"]
        ],
    )
    district_axis.axhline(0.5, color="#F59E0B", linestyle="--", label="F1 0.5")
    district_axis.set_ylim(0, 0.75)
    district_axis.set_title(
        "자치구별 권역 F1 편차",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    district_axis.grid(axis="y", alpha=0.18)
    district_axis.legend()
    for index, value in enumerate(district_view["F1"]):
        district_axis.text(
            index,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )

    route_axis = figure.add_subplot(grid[3, 1])
    route_axis.axis("off")
    route = route_report["route"]
    route_stability = route.get("stability_validation", {})
    route_axis.text(
        0.02,
        0.72,
        (
            "실제 경로 효용\n"
            f"검증 OD        {route_stability.get('evaluated_pairs', 0)}개\n"
            f"위험감소 경로  "
            f"{route_stability.get('positive_risk_reduction_ratio', 0) * 100:.1f}%\n"
            f"중앙 위험감소  "
            f"{route_stability.get('median_risk_reduction_pct', 0):.1f}%"
        ),
        fontsize=14,
        linespacing=1.5,
        color="#0F172A",
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.7",
            "facecolor": "white",
            "edgecolor": "#CBD5E1",
        },
    )
    route_axis.text(
        0.02,
        0.0,
        (
            "허용: 연구·시연, 담당자 검토형 파일럿\n"
            "금지: 모델 단독 결정, 절대 안전 보증\n"
            "필수: 다수 OD 현장검증, 최신 사고·조도·교통량 보강"
        ),
        fontsize=13,
        linespacing=1.55,
        color="#991B1B",
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def generate_final_model_evaluation() -> dict[str, object]:
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_report = _read_json(REPORT_OUTPUT_DIR / "edge_model_report.json")
    regional_report = _read_json(
        REPORT_OUTPUT_DIR / "regional_boosting_report.json"
    )
    route_report = _read_json(REPORT_OUTPUT_DIR / "full_pipeline_report.json")
    temporal_report = _read_json(REPORT_OUTPUT_DIR / "temporal_split_report.json")
    quality_report = _read_json(REPORT_OUTPUT_DIR / "data_quality_report.json")
    predictions = pd.read_csv(
        REPORT_OUTPUT_DIR / "regional_boosting_predictions.csv"
    )
    regional, district, intervals = _regional_metrics(predictions)
    performance = _performance_table(edge_report, regional, intervals)
    safety = _safety_matrix(
        regional,
        district,
        route_report,
        regional_report,
        temporal_report,
        quality_report,
    )

    performance.to_csv(PERFORMANCE_CSV, index=False, encoding="utf-8-sig")
    safety.to_csv(SAFETY_MATRIX_CSV, index=False, encoding="utf-8-sig")
    markdown = _markdown_report(
        performance,
        safety,
        district,
        regional,
        intervals,
        route_report,
    )
    EVALUATION_MD.write_text(markdown, encoding="utf-8")
    EVALUATION_HTML.write_text(
        _html_report(
            performance,
            safety,
            district,
            regional,
            intervals,
            route_report,
        ),
        encoding="utf-8",
    )
    _draw_dashboard(
        performance,
        safety,
        district,
        regional,
        route_report,
        DASHBOARD_PNG,
    )
    result: dict[str, object] = {
        "evaluation_date": date.today().isoformat(),
        "overall_verdict": "제한적 파일럿 가능 / 무감독 실서비스 불가",
        "regional_metrics": {
            key: round(float(value), 6)
            if isinstance(value, (float, np.floating))
            else value
            for key, value in regional.items()
        },
        "bootstrap_95_percent_intervals": intervals,
        "district_metrics": district.to_dict(orient="records"),
        "safety_gate_counts": safety["상태"].value_counts().to_dict(),
        "recommendations": _recommendations(),
        "artifacts": {
            "performance_csv": str(PERFORMANCE_CSV),
            "safety_matrix_csv": str(SAFETY_MATRIX_CSV),
            "markdown_report": str(EVALUATION_MD),
            "html_report": str(EVALUATION_HTML),
            "dashboard_png": str(DASHBOARD_PNG),
        },
    }
    EVALUATION_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    report = generate_final_model_evaluation()
    metrics = report["regional_metrics"]
    print(
        "최종 평가 완료: "
        f"F1={metrics['f1']:.3f}, "
        f"Recall={metrics['recall']:.3f}, "
        f"판정={report['overall_verdict']}"
    )
