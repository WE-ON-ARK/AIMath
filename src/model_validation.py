"""모델 검증 모듈 — CMCS 민감도, Ablation, Calibration, 시간 분리, 구 홀드아웃."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CHART_OUTPUT_DIR, REPORT_OUTPUT_DIR
from src.cmcs_calculator import CMCSCalculator, CMCSWeights


# ---------------------------------------------------------------------------
# CMCS 가중치 ±20% 민감도 분석
# ---------------------------------------------------------------------------

def _perturb_dimension_weights(
    base: dict[str, float],
    target_dim: str,
    delta: float,
) -> dict[str, float]:
    """target_dim 가중치를 delta만큼 변경하고 나머지를 비례 보정한다."""
    original = base[target_dim]
    new_val = float(np.clip(original + delta, 0.02, 0.98))
    change = new_val - original
    others = {k: v for k, v in base.items() if k != target_dim}
    total_others = sum(others.values())
    adjusted = {
        k: v - change * (v / total_others) for k, v in others.items()
    }
    adjusted[target_dim] = new_val
    return adjusted


def cmcs_sensitivity_analysis(
    features: pd.DataFrame,
    delta_fraction: float = 0.20,
    output_path: str | Path = REPORT_OUTPUT_DIR / "cmcs_sensitivity_report.json",
) -> dict[str, object]:
    """각 CMCS 차원 가중치를 ±delta_fraction 변경했을 때 점수 변화를 측정한다."""
    base_weights = dict(CMCSWeights().dimensions)
    base_calc = CMCSCalculator()
    base_scores = base_calc.calculate_cmcs_ahp(features)

    results: dict[str, object] = {"base_weights": base_weights, "delta_fraction": delta_fraction, "dimensions": {}}

    for dim in base_weights:
        dim_results: dict[str, object] = {}
        for sign, label in [(+1, "plus"), (-1, "minus")]:
            delta = sign * delta_fraction * base_weights[dim]
            new_dim_weights = _perturb_dimension_weights(base_weights, dim, delta)
            new_weights = CMCSWeights(
                dimensions=new_dim_weights,
                sub_weights=CMCSWeights().sub_weights,
                safety_bonus=CMCSWeights().safety_bonus,
            )
            perturbed_scores = CMCSCalculator(new_weights).calculate_cmcs_ahp(features)

            diff = perturbed_scores - base_scores
            pct_tier_change = float(
                ((base_scores // 0.25) != (perturbed_scores // 0.25)).mean()
            )
            dim_results[label] = {
                "new_weight": round(new_dim_weights[dim], 4),
                "mean_abs_diff": round(float(diff.abs().mean()), 6),
                "max_abs_diff": round(float(diff.abs().max()), 6),
                "spearman_r": round(float(base_scores.corr(perturbed_scores, method="spearman")), 6),
                "pct_tier_change": round(pct_tier_change, 4),
            }
        results["dimensions"][dim] = dim_results  # type: ignore[index]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# 변수 제거 Ablation 테스트
# ---------------------------------------------------------------------------

def _quick_auc(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> float:
    """LeaveOneGroupOut CV로 ROC-AUC를 빠르게 계산한다."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)),
    ])
    try:
        proba = cross_val_predict(
            pipe, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba"
        )[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return float("nan")


def ablation_test(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "accident_hotspot_within_radius",
    group_col: str = "district",
    output_path: str | Path = REPORT_OUTPUT_DIR / "ablation_report.json",
) -> pd.DataFrame:
    """피처를 하나씩 제거했을 때 AUC 변화를 측정한다."""
    X_full = dataset[feature_columns]
    y = dataset[label_col].astype(int)
    groups = dataset[group_col]

    baseline_auc = _quick_auc(X_full, y, groups)
    rows = []
    for col in feature_columns:
        reduced = [c for c in feature_columns if c != col]
        auc = _quick_auc(dataset[reduced], y, groups)
        rows.append({
            "removed_feature": col,
            "auc_without": round(auc, 6),
            "auc_drop": round(baseline_auc - auc, 6),
        })

    result = pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)
    report = {
        "baseline_auc": round(baseline_auc, 6),
        "feature_count": len(feature_columns),
        "ablation": result.to_dict(orient="records"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# 예측 확률 Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(
    y: pd.Series,
    proba: np.ndarray,
    model_name: str = "model",
    n_bins: int = 10,
    output_path: str | Path = REPORT_OUTPUT_DIR / "calibration_report.json",
    chart_path: str | Path = CHART_OUTPUT_DIR / "calibration_curve.png",
) -> dict[str, object]:
    """신뢰도 다이어그램과 Brier 점수로 캘리브레이션을 평가한다."""
    fraction_positive, mean_predicted = calibration_curve(
        y, proba, n_bins=n_bins, strategy="quantile"
    )
    brier = float(brier_score_loss(y, proba))

    report: dict[str, object] = {
        "model": model_name,
        "brier_score": round(brier, 6),
        "brier_perfect": 0.0,
        "brier_dummy": round(float(brier_score_loss(y, np.full_like(proba, y.mean()))), 6),
        "calibration_bins": [
            {
                "mean_predicted": round(float(mp), 4),
                "fraction_positive": round(float(fp), 4),
            }
            for mp, fp in zip(mean_predicted, fraction_positive)
        ],
    }
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_predicted, fraction_positive, "s-", label=model_name, color="#2563eb")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration curve (reliability diagram)")
        ax.legend()
        fig.tight_layout()
        p = Path(chart_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        report["chart_path"] = str(p)
    except Exception:
        pass

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 위험 임계값 최적화
# ---------------------------------------------------------------------------

def optimize_threshold(
    y: pd.Series,
    proba: np.ndarray,
    metric: str = "f1",
    output_path: str | Path = REPORT_OUTPUT_DIR / "threshold_report.json",
) -> dict[str, object]:
    """precision-recall 트레이드오프에서 최적 임계값을 찾는다."""
    thresholds = np.linspace(0.05, 0.95, 181)
    records = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        records.append({
            "threshold": round(float(t), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    df = pd.DataFrame(records)
    if metric == "f1":
        best_idx = int(df["f1"].idxmax())
    else:
        best_idx = int(df["recall"].idxmax())

    best = df.iloc[best_idx].to_dict()
    prevalence = float(y.mean())

    report: dict[str, object] = {
        "metric": metric,
        "prevalence": round(prevalence, 4),
        "default_threshold_0.5": df[df["threshold"].between(0.499, 0.501)].to_dict(orient="records"),
        "optimal_threshold": best,
        "sweep": df.to_dict(orient="records"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 시간 분리 검증
# ---------------------------------------------------------------------------

def temporal_split_validation(
    features: pd.DataFrame,
    hotspots: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "accident_hotspot_within_radius",
    group_col: str = "district",
    train_years: Sequence[int] | None = None,
    test_years: Sequence[int] | None = None,
    label_radius_m: float = 300.0,
    output_path: str | Path = REPORT_OUTPUT_DIR / "temporal_split_report.json",
) -> dict[str, object]:
    """과거 사고 데이터로 학습 → 최근 사고 지점 예측 성능을 검증한다."""
    from src.real_data_pipeline import haversine_matrix

    if train_years is None:
        all_years = sorted(hotspots["search_year"].astype(int).unique())
        mid = all_years[len(all_years) // 2]
        train_years = [y for y in all_years if y <= mid]
        test_years = [y for y in all_years if y > mid]

    train_years_set = set(train_years)
    test_years_set = set(test_years)

    def _label_from_subset(year_filter: set[int]) -> pd.Series:
        subset = hotspots[hotspots["search_year"].astype(int).isin(year_filter)]
        if subset.empty:
            return pd.Series(0, index=features.index)
        dist = haversine_matrix(
            features["latitude"], features["longitude"],
            subset["la_crd"], subset["lo_crd"],
        )
        return pd.Series(
            (dist <= label_radius_m).any(axis=1).astype(int),
            index=features.index,
        )

    y_train = _label_from_subset(train_years_set)
    y_test = _label_from_subset(test_years_set)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)),
    ])

    X = features[feature_columns]
    if y_train.nunique() < 2:
        return {
            "error": "학습 레이블에 양성/음성 샘플 모두 필요합니다.",
            "train_years": [int(y) for y in train_years],
            "test_years": [int(y) for y in test_years],
            "train_positive": int(y_train.sum()),
            "train_total": int(len(y_train)),
        }

    pipe.fit(X, y_train)
    proba_test = pipe.predict_proba(X)[:, 1]

    metrics: dict[str, object] = {}
    if y_test.nunique() >= 2:
        metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, proba_test)), 6),
            "average_precision": round(float(average_precision_score(y_test, proba_test)), 6),
            "brier_score": round(float(brier_score_loss(y_test, proba_test)), 6),
        }
    else:
        metrics = {"warning": "검증 레이블에 한 가지 클래스만 있어 AUC를 계산할 수 없습니다."}

    dummy = DummyClassifier(strategy="prior").fit(X, y_train)
    dummy_proba = dummy.predict_proba(X)[:, 1]
    dummy_brier = (
        round(float(brier_score_loss(y_test, dummy_proba)), 6)
        if y_test.nunique() >= 2 else None
    )

    report: dict[str, object] = {
        "train_years": [int(y) for y in train_years],
        "test_years": [int(y) for y in test_years],
        "train_positive": int(y_train.sum()),
        "train_total": int(len(y_train)),
        "test_positive": int(y_test.sum()),
        "test_total": int(len(y_test)),
        "metrics": metrics,
        "dummy_brier_score": dummy_brier,
        "interpretation": (
            "train_years의 사고 다발지역으로 학습해 test_years의 사고 발생 여부를 예측. "
            "데이터가 22건으로 매우 적어 통계적 한계가 크다."
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# 구 단위 홀드아웃 상세 요약
# ---------------------------------------------------------------------------

def district_holdout_summary(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "accident_hotspot_within_radius",
    group_col: str = "district",
    output_path: str | Path = REPORT_OUTPUT_DIR / "district_holdout_summary.json",
) -> pd.DataFrame:
    """구 단위 LeaveOneOut에서 각 구별 성능을 측정한다."""
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)),
    ])
    X = dataset[feature_columns]
    y = dataset[label_col].astype(int)
    groups = dataset[group_col]

    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = dataset[group_col].iloc[test_idx[0]]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        if y_tr.nunique() < 2:
            continue
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pos_rate = round(float(y_te.mean()), 4)
        n_pos = int(y_te.sum())
        metrics_row: dict[str, object] = {
            "held_out_district": held_out,
            "test_count": len(y_te),
            "positive_count": n_pos,
            "positive_rate": pos_rate,
        }
        if y_te.nunique() >= 2:
            metrics_row["roc_auc"] = round(float(roc_auc_score(y_te, proba)), 6)
            metrics_row["average_precision"] = round(float(average_precision_score(y_te, proba)), 6)
        else:
            metrics_row["roc_auc"] = None
            metrics_row["average_precision"] = None
            metrics_row["note"] = "단일 클래스 — AUC 계산 불가"
        rows.append(metrics_row)

    result = pd.DataFrame(rows)
    report: dict[str, object] = {
        "validation_method": "LeaveOneGroupOut by district",
        "per_district": result.to_dict(orient="records"),
        "mean_roc_auc": round(float(result["roc_auc"].dropna().mean()), 6) if "roc_auc" in result else None,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# 통합 실행
# ---------------------------------------------------------------------------

def run_full_model_validation(
    dataset: pd.DataFrame,
    hotspots: pd.DataFrame,
    feature_columns: list[str],
    label_col: str = "accident_hotspot_within_radius",
    group_col: str = "district",
) -> dict[str, object]:
    """모든 검증을 순서대로 실행하고 통합 리포트를 반환한다."""

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=0.5, random_state=42)),
    ])
    X = dataset[feature_columns]
    y = dataset[label_col].astype(int)
    groups = dataset[group_col]

    cv_proba = cross_val_predict(
        pipe, X, y, groups=groups, cv=LeaveOneGroupOut(), method="predict_proba"
    )[:, 1]
    pipe.fit(X, y)

    print("[1/6] CMCS 가중치 민감도 분석...")
    sensitivity = cmcs_sensitivity_analysis(dataset)

    print("[2/6] Ablation 테스트...")
    ablation = ablation_test(dataset, feature_columns, label_col, group_col)

    print("[3/6] Calibration 분석...")
    cal = calibration_analysis(y, cv_proba)

    print("[4/6] 임계값 최적화...")
    threshold = optimize_threshold(y, cv_proba)

    print("[5/6] 시간 분리 검증...")
    temporal = temporal_split_validation(dataset, hotspots, feature_columns, label_col, group_col)

    print("[6/6] 구 단위 홀드아웃 요약...")
    holdout = district_holdout_summary(dataset, feature_columns, label_col, group_col)

    summary_path = REPORT_OUTPUT_DIR / "model_validation_summary.json"
    summary = {
        "cmcs_sensitivity": {"output": str(REPORT_OUTPUT_DIR / "cmcs_sensitivity_report.json")},
        "ablation": {
            "baseline_auc": ablation["baseline_auc"] if isinstance(ablation, dict) else None,  # type: ignore[index]
            "top3_important": ablation.head(3)["removed_feature"].tolist() if isinstance(ablation, pd.DataFrame) else [],
        },
        "calibration": {
            "brier_score": cal.get("brier_score"),
            "brier_dummy": cal.get("brier_dummy"),
        },
        "threshold": {
            "optimal": threshold.get("optimal_threshold"),
        },
        "temporal": temporal,
        "district_holdout": holdout.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
