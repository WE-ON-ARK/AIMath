"""통계·공간 증거를 결합해 CMCS 가중치를 재현 가능하게 산출한다.

수학적 도출 흐름
================
기존 "AHP"라 명명했던 직관적 가중치를 폐기하고, 관측 데이터에서 출발하는
4-채널 증거 합성(evidence synthesis) 방법을 사용한다.

  Stage 1 — Spearman 순위 상관
    각 피처 xⱼ와 사고 건수 y 사이의 비모수 단조 관계 ρⱼ를 측정.
    방향 부호를 검증(expected_sign_j × ρⱼ > 0)하고, p-value로
    유의성 가중치를 부여한다.
      evidence_spearman_j = max(0, sign_j × ρⱼ) × max(0, 1 − p_j/0.10)

  Stage 2 — Poisson 회귀 (노출량 보정)
    사고 발생률 = 사고 건수 / 도로 연장(km)을 종속변수로, 표준화 피처를
    투입해 도로 연장을 sample_weight로 보정한 Poisson GLM을 학습.
    LeaveOneGroupOut(자치구)으로 fold별 표준화 회귀계수 중앙값을 산출.
      evidence_poisson_j = max(0, sign_j × β̃_j) × consistency_j × R²_pseudo

  Stage 3 — Logistic 회귀 (이진 분류)
    사고 유무를 종속변수로 표준화 Logistic 회귀 수행. 동일하게
    LeaveOneGroupOut 교차검증으로 fold별 계수 중앙값과 부호 일관성,
    모델 신뢰도(ROC-AUC 기반)를 곱해 증거 점수를 산출.
      evidence_logistic_j = max(0, sign_j × β̃_j) × consistency_j × reliability

  Stage 4 — Bivariate Moran's I (공간 자기상관)
    KNN(k=8) 공간 가중치 행렬로 피처-사고 간 이변량 Moran's I를 계산.
    순열 검정(499회)으로 의사 p-value를 산출하고, 방향 부호 검증 후
    공간 통계 증거로 사용.
      evidence_moran_j = max(0, sign_j × I_xy) × (1 − p_perm)

  합성 공식
  ---------
  각 채널 m의 피처별 증거 점수를 L1 정규화한 뒤, 채널 간 산술평균으로
  통합한다. 이를 다시 L1 정규화하면 피처별 최종 증거 비중 w*_j가 된다.

    w̃_j^m = evidence_j^m / Σ_k evidence_k^m     (채널 내 L1 정규화)
    w̄_j   = (1/M) Σ_m w̃_j^m                     (채널 간 평균)
    w*_j   = w̄_j / Σ_k w̄_k                       (최종 L1 정규화)

  차원 가중치 도출
  ----------------
  피처 j가 속한 차원을 dim(j)라 하면:
    W_dim = Σ_{j: dim(j)=dim} w*_j / Σ_{j: dim(j) ∈ hazard} w*_j
  하위 가중치:
    w_sub(j|dim) = w*_j / Σ_{k: dim(k)=dim} w*_k
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CHART_OUTPUT_DIR, MODEL_DIR, REPORT_OUTPUT_DIR
from src.cmcs_calculator import CMCSCalculator, CMCSWeights


DATA_DRIVEN_WEIGHTS_PATH = MODEL_DIR / "cmcs_data_driven_weights.json"
DATA_DRIVEN_REPORT_PATH = REPORT_OUTPUT_DIR / "cmcs_weight_evidence_report.json"
DATA_DRIVEN_CHART_PATH = CHART_OUTPUT_DIR / "cmcs_weight_evidence.png"
WEIGHT_CELL_SIZE_M = 1750


@dataclass(frozen=True)
class FeatureEvidenceSpec:
    feature: str
    dimension: str
    expected_sign: int
    source: str


FEATURE_SPECS = (
    FeatureEvidenceSpec("traffic_volume_norm", "risk", 1, "traffic"),
    FeatureEvidenceSpec("avg_speed_norm", "risk", 1, "speed"),
    FeatureEvidenceSpec(
        "narrow_sidewalk_norm", "discomfort", 1, "sidewalk"
    ),
    FeatureEvidenceSpec("slope_norm", "discomfort", 1, "slope"),
    FeatureEvidenceSpec("is_alley", "discomfort", 1, "road_type"),
    FeatureEvidenceSpec(
        "pedestrian_flow_norm", "congestion", 1, "pedestrian_flow"
    ),
    FeatureEvidenceSpec(
        "academy_density_norm", "congestion", 1, "academy"
    ),
    FeatureEvidenceSpec(
        "bus_stop_nearby_norm", "congestion", 1, "bus_stop"
    ),
    FeatureEvidenceSpec(
        "illegal_parking_norm", "obstruction", 1, "illegal_parking"
    ),
    FeatureEvidenceSpec("light_deficit", "obstruction", 1, "lighting"),
    FeatureEvidenceSpec("crosswalk_deficit", "crossing", 1, "crosswalk"),
    FeatureEvidenceSpec("signal_deficit", "crossing", 1, "signal"),
    FeatureEvidenceSpec("lane_count_norm", "crossing", 1, "lane"),
    FeatureEvidenceSpec(
        "has_speed_bump", "safety_bonus", -1, "speed_bump"
    ),
    FeatureEvidenceSpec("has_cctv", "safety_bonus", -1, "cctv"),
    FeatureEvidenceSpec(
        "is_school_zone", "safety_bonus", -1, "school_zone"
    ),
)

HAZARD_DIMENSIONS = (
    "risk",
    "discomfort",
    "congestion",
    "obstruction",
    "crossing",
)


# ---------------------------------------------------------------------------
# Stage 0: 전처리된 edge feature → 공간 셀 집계
# ---------------------------------------------------------------------------

def _feature_frame(segments: pd.DataFrame) -> pd.DataFrame:
    calculator = CMCSCalculator()
    frame = pd.DataFrame(index=segments.index)
    for spec in FEATURE_SPECS:
        if spec.feature == "light_deficit":
            frame[spec.feature] = 1.0 - calculator._series(
                segments, "light_density_norm", 0.5
            )
        elif spec.feature == "crosswalk_deficit":
            frame[spec.feature] = 1.0 - calculator._series(
                segments, "has_crosswalk"
            )
        elif spec.feature == "signal_deficit":
            frame[spec.feature] = 1.0 - calculator._series(
                segments, "has_signal"
            )
        else:
            frame[spec.feature] = calculator._series(
                segments, spec.feature
            )
    return frame


def prepare_weight_learning_table(
    edge_features: pd.DataFrame,
    cell_size_m: int = WEIGHT_CELL_SIZE_M,
) -> pd.DataFrame:
    """방향 중복을 제거하고 자치구 경계로 분리된 공간 셀로 집계한다.

    전처리 연결: ``api_preprocessing.preprocess_for_cmcs_analysis``를 통과한
    edge_features를 입력으로 받아, 정규화·이상치 처리가 완료된 피처를 사용한다.
    """
    from src.api_preprocessing import preprocess_for_cmcs_analysis

    clean_features, preprocessing_report = preprocess_for_cmcs_analysis(
        edge_features
    )

    segments = clean_features.drop_duplicates("segment_id").copy()
    segments["district"] = segments["district"].fillna("미분류").astype(str)
    segments["center_x"] = pd.to_numeric(
        segments["center_x"], errors="coerce"
    )
    segments["center_y"] = pd.to_numeric(
        segments["center_y"], errors="coerce"
    )
    segments = segments.dropna(subset=["center_x", "center_y"])
    segments["region_x"] = (segments["center_x"] // cell_size_m).astype(int)
    segments["region_y"] = (segments["center_y"] // cell_size_m).astype(int)
    feature_values = _feature_frame(segments)
    working = pd.concat(
        [
            segments[
                [
                    "district",
                    "region_x",
                    "region_y",
                    "center_x",
                    "center_y",
                    "segment_id",
                    "length_m",
                    "accident_count",
                ]
            ].reset_index(drop=True),
            feature_values.reset_index(drop=True),
        ],
        axis=1,
    )
    working["accident_count"] = pd.to_numeric(
        working["accident_count"], errors="coerce"
    ).fillna(0.0)
    working["accident_label"] = (working["accident_count"] > 0).astype(int)
    keys = ["district", "region_x", "region_y"]
    aggregation: dict[str, object] = {
        spec.feature: "mean" for spec in FEATURE_SPECS
    }
    aggregation.update(
        {
            "center_x": "mean",
            "center_y": "mean",
            "segment_id": "size",
            "length_m": "sum",
            "accident_count": "max",
            "accident_label": "max",
        }
    )
    table = (
        working.groupby(keys, as_index=False)
        .agg(aggregation)
        .rename(
            columns={
                "segment_id": "segment_count",
                "length_m": "exposure_length_m",
            }
        )
    )
    table.attrs["preprocessing_report"] = preprocessing_report
    return table


def _spatial_groups(table: pd.DataFrame) -> pd.Series:
    return table["district"].astype(str)


# ---------------------------------------------------------------------------
# Stage 1: Spearman 순위 상관
# ---------------------------------------------------------------------------

def _spearman_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """각 피처와 사고 건수 간 Spearman ρ 산출.

    evidence_j = max(0, sign_j × ρ_j) × max(0, 1 − p_j / α)
    여기서 α = 0.10 (유의수준 경계).
    """
    scores: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    target = table["accident_count"].astype(float)
    for spec in FEATURE_SPECS:
        feature_values = table[spec.feature].astype(float)
        if feature_values.nunique(dropna=True) < 2:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = spearmanr(
                feature_values,
                target,
                nan_policy="omit",
            )
        correlation = 0.0 if not math.isfinite(correlation) else float(correlation)
        p_value = 1.0 if not math.isfinite(p_value) else float(p_value)
        signed_effect = spec.expected_sign * correlation
        significance = max(0.0, 1.0 - min(p_value / 0.10, 1.0))
        score = max(0.0, signed_effect) * significance
        scores[spec.feature] = score
        details[spec.feature] = {
            "rho": round(correlation, 6),
            "p_value": round(p_value, 6),
            "expected_sign": spec.expected_sign,
            "signed_effect": round(signed_effect, 6),
            "significance_weight": round(significance, 6),
            "evidence_score": round(score, 6),
        }
    return scores, details


# ---------------------------------------------------------------------------
# Stage 2: Poisson 회귀 (노출량 보정 사고율)
# ---------------------------------------------------------------------------

def _poisson_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object]]:
    """도로 연장을 노출량으로 보정한 Poisson GLM 회귀.

    종속변수: 사고 건수 / 도로 연장(km)
    sample_weight: 도로 연장(km)
    교차검증: LeaveOneGroupOut (자치구 단위)
    evidence_j = max(0, sign_j × β̃_j) × consistency_j × R²_pseudo
    """
    columns = [spec.feature for spec in FEATURE_SPECS]
    X = table[columns]
    exposure_km = (
        pd.to_numeric(table["exposure_length_m"], errors="coerce")
        .fillna(1.0)
        .clip(lower=50.0)
        / 1000.0
    )
    count = pd.to_numeric(
        table["accident_count"], errors="coerce"
    ).fillna(0.0)
    rate = count / exposure_km
    groups = _spatial_groups(table)
    coefficients = []
    heldout_predictions = np.zeros(len(table), dtype=float)
    splitter = LeaveOneGroupOut()
    for train_index, test_index in splitter.split(X, count, groups):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_values = imputer.fit_transform(X.iloc[train_index])
        train_scaled = scaler.fit_transform(train_values)
        model = PoissonRegressor(alpha=1.0, max_iter=2000)
        model.fit(
            train_scaled,
            rate.iloc[train_index],
            sample_weight=exposure_km.iloc[train_index],
        )
        coefficients.append(model.coef_.astype(float))
        test_scaled = scaler.transform(imputer.transform(X.iloc[test_index]))
        heldout_predictions[test_index] = model.predict(test_scaled)

    coefficients_array = np.asarray(coefficients)
    median_coefficient = np.median(coefficients_array, axis=0)
    observed_rate = rate.to_numpy()
    denominator = float(
        np.sum(
            exposure_km
            * (observed_rate - np.average(observed_rate, weights=exposure_km))
            ** 2
        )
    )
    numerator = float(
        np.sum(exposure_km * (observed_rate - heldout_predictions) ** 2)
    )
    pseudo_r2 = max(0.0, min(1.0, 1.0 - numerator / denominator)) if denominator else 0.0
    scores: dict[str, float] = {}
    feature_details: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(FEATURE_SPECS):
        signed = spec.expected_sign * median_coefficient[index]
        sign_consistency = float(
            np.mean(
                spec.expected_sign * coefficients_array[:, index] > 0
            )
        )
        score = max(0.0, float(signed)) * sign_consistency * pseudo_r2
        scores[spec.feature] = score
        feature_details[spec.feature] = {
            "median_standardized_coefficient": round(
                float(median_coefficient[index]), 6
            ),
            "expected_direction_consistency": round(sign_consistency, 6),
            "signed_effect": round(float(signed), 6),
            "evidence_score": round(score, 6),
        }
    return scores, {
        "model": "PoissonRegressor(alpha=1.0) on accident_count/road_km",
        "exposure": "road length in km via sample_weight",
        "validation": "LeaveOneGroupOut by district",
        "heldout_weighted_pseudo_r2": round(pseudo_r2, 6),
        "formula": (
            "evidence_j = max(0, sign_j × β̃_j) "
            "× direction_consistency_j × R²_pseudo"
        ),
        "features": feature_details,
    }


# ---------------------------------------------------------------------------
# Stage 3: Logistic 회귀 (이진 분류)
# ---------------------------------------------------------------------------

def _logistic_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object]]:
    """표준화 Logistic 회귀로 이진 사고 유무 분류.

    교차검증: LeaveOneGroupOut (자치구 단위)
    모델 신뢰도: reliability = max(0, (AUC − 0.5) / 0.5)
    evidence_j = max(0, sign_j × β̃_j) × consistency_j × reliability
    """
    columns = [spec.feature for spec in FEATURE_SPECS]
    X = table[columns]
    y = table["accident_label"].astype(int)
    groups = _spatial_groups(table)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.5,
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    probability = cross_val_predict(
        pipeline,
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    auc = float(roc_auc_score(y, probability))
    ap = float(average_precision_score(y, probability))
    reliability = max(0.0, min(1.0, (auc - 0.5) / 0.5))

    fold_coefficients = []
    splitter = LeaveOneGroupOut()
    for train_index, _ in splitter.split(X, y, groups):
        fold_model = Pipeline(pipeline.steps)
        fold_model.fit(X.iloc[train_index], y.iloc[train_index])
        fold_coefficients.append(
            fold_model.named_steps["model"].coef_[0].astype(float)
        )
    coefficients = np.asarray(fold_coefficients)
    median_coefficient = np.median(coefficients, axis=0)
    scores: dict[str, float] = {}
    feature_details: dict[str, dict[str, float]] = {}
    for index, spec in enumerate(FEATURE_SPECS):
        signed = spec.expected_sign * median_coefficient[index]
        sign_consistency = float(
            np.mean(spec.expected_sign * coefficients[:, index] > 0)
        )
        score = max(0.0, float(signed)) * sign_consistency * reliability
        scores[spec.feature] = score
        feature_details[spec.feature] = {
            "median_standardized_coefficient": round(
                float(median_coefficient[index]), 6
            ),
            "expected_direction_consistency": round(sign_consistency, 6),
            "signed_effect": round(float(signed), 6),
            "evidence_score": round(score, 6),
        }
    return scores, {
        "model": "LogisticRegression(C=0.5, balanced)",
        "validation": "LeaveOneGroupOut by district",
        "roc_auc": round(auc, 6),
        "average_precision": round(ap, 6),
        "reliability_multiplier": round(reliability, 6),
        "formula": (
            "evidence_j = max(0, sign_j × β̃_j) "
            "× direction_consistency_j × reliability"
        ),
        "features": feature_details,
    }


# ---------------------------------------------------------------------------
# Stage 4: Bivariate Moran's I (공간 자기상관)
# ---------------------------------------------------------------------------

def _knn_neighbors(
    coordinates: np.ndarray,
    neighbors: int = 8,
) -> np.ndarray:
    k = min(neighbors + 1, len(coordinates))
    _, indices = cKDTree(coordinates).query(coordinates, k=k)
    if indices.ndim == 1:
        indices = indices[:, None]
    return indices[:, 1:]


def _bivariate_moran(
    feature: np.ndarray,
    target: np.ndarray,
    neighbor_indices: np.ndarray,
) -> float:
    """이변량 Moran's I = Σ(z_x · W·z_y) / Σ(z_x²)."""
    feature_z = feature - np.nanmean(feature)
    target_z = target - np.nanmean(target)
    feature_std = np.nanstd(feature_z)
    target_std = np.nanstd(target_z)
    if np.isclose(feature_std, 0) or np.isclose(target_std, 0):
        return 0.0
    feature_z = feature_z / feature_std
    target_z = target_z / target_std
    spatial_lag = target_z[neighbor_indices].mean(axis=1)
    denominator = float(np.dot(feature_z, feature_z))
    return float(np.dot(feature_z, spatial_lag) / denominator) if denominator else 0.0


def _morans_evidence(
    table: pd.DataFrame,
    permutations: int = 499,
    random_state: int = 42,
) -> tuple[dict[str, float], dict[str, object]]:
    """순열 검정 기반 이변량 Moran's I.

    evidence_j = max(0, sign_j × I_xy) × (1 − p_perm)
    """
    coordinates = table[["center_x", "center_y"]].to_numpy(dtype=float)
    neighbors = _knn_neighbors(coordinates)
    target = table["accident_count"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    scores: dict[str, float] = {}
    feature_details: dict[str, dict[str, float]] = {}
    for spec in FEATURE_SPECS:
        feature = table[spec.feature].to_numpy(dtype=float)
        observed = _bivariate_moran(feature, target, neighbors)
        permutation_values = np.asarray(
            [
                _bivariate_moran(feature, rng.permutation(target), neighbors)
                for _ in range(permutations)
            ]
        )
        p_value = float(
            (1 + np.sum(np.abs(permutation_values) >= abs(observed)))
            / (permutations + 1)
        )
        signed = spec.expected_sign * observed
        score = max(0.0, signed) * (1.0 - p_value)
        scores[spec.feature] = score
        feature_details[spec.feature] = {
            "bivariate_morans_i": round(observed, 6),
            "permutation_p_value": round(p_value, 6),
            "expected_sign": spec.expected_sign,
            "signed_effect": round(signed, 6),
            "evidence_score": round(score, 6),
        }
    target_i = _bivariate_moran(target, target, neighbors)
    return scores, {
        "model": "Bivariate Moran's I with KNN spatial weights",
        "neighbors": int(neighbors.shape[1]),
        "permutations": permutations,
        "target_univariate_morans_i": round(target_i, 6),
        "formula": "evidence_j = max(0, sign_j × I_xy) × (1 − p_perm)",
        "features": feature_details,
    }


# ---------------------------------------------------------------------------
# 증거 합성 (Evidence Synthesis)
# ---------------------------------------------------------------------------

def _normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """L1 정규화: w̃_j = max(0, s_j) / Σ_k max(0, s_k)."""
    positive = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {key: 0.0 for key in positive}
    return {key: value / total for key, value in positive.items()}


def _combine_evidence(
    method_scores: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """4-채널 증거를 합성하여 피처별 최종 가중치를 산출한다.

    합성 공식:
      1. 각 채널 m의 증거 점수를 L1 정규화:
           w̃_j^m = evidence_j^m / Σ_k evidence_k^m
      2. 채널 간 산술 평균:
           w̄_j = (1/M) Σ_m w̃_j^m
      3. 최종 L1 정규화:
           w*_j = w̄_j / Σ_k w̄_k
    """
    normalized_methods = {
        method: _normalize_scores(scores)
        for method, scores in method_scores.items()
        if sum(max(0.0, float(value)) for value in scores.values()) > 0
    }
    if not normalized_methods:
        raise ValueError("양의 방향으로 일치하는 통계 증거가 없습니다.")
    features = [spec.feature for spec in FEATURE_SPECS]
    combined = {
        feature: float(
            np.mean(
                [
                    normalized.get(feature, 0.0)
                    for normalized in normalized_methods.values()
                ]
            )
        )
        for feature in features
    }
    return _normalize_scores(combined), normalized_methods


# ---------------------------------------------------------------------------
# 차원 가중치 도출
# ---------------------------------------------------------------------------

def _weights_from_combined_evidence(
    combined: Mapping[str, float],
) -> tuple[CMCSWeights, dict[str, object]]:
    """합성된 피처 증거를 차원·하위 가중치로 변환한다.

    차원 가중치:
      W_dim = Σ_{j: dim(j)=dim} w*_j / Σ_{j: dim(j) ∈ hazard} w*_j
    하위 가중치:
      w_sub(j|dim) = w*_j / Σ_{k: dim(k)=dim} w*_k
    """
    feature_dimension = {
        spec.feature: spec.dimension for spec in FEATURE_SPECS
    }
    dimension_mass = {
        dimension: sum(
            combined[feature]
            for feature, assigned_dimension in feature_dimension.items()
            if assigned_dimension == dimension
        )
        for dimension in (*HAZARD_DIMENSIONS, "safety_bonus")
    }
    hazard_total = sum(dimension_mass[dimension] for dimension in HAZARD_DIMENSIONS)
    if hazard_total <= 0:
        raise ValueError("위험 차원에 대한 통계 증거가 없습니다.")
    dimensions = {
        dimension: dimension_mass[dimension] / hazard_total
        for dimension in HAZARD_DIMENSIONS
    }
    sub_weights: dict[str, dict[str, float]] = {}
    for dimension in HAZARD_DIMENSIONS:
        members = [
            feature
            for feature, assigned_dimension in feature_dimension.items()
            if assigned_dimension == dimension
        ]
        total = sum(combined[feature] for feature in members)
        sub_weights[dimension] = (
            {feature: combined[feature] / total for feature in members}
            if total > 0
            else {feature: 1.0 / len(members) for feature in members}
        )

    protective_features = [
        spec.feature
        for spec in FEATURE_SPECS
        if spec.dimension == "safety_bonus"
    ]
    protective_mass = dimension_mass["safety_bonus"]
    bonus_total = min(0.15, protective_mass)
    safety_bonus = (
        {
            feature: bonus_total * combined[feature] / protective_mass
            for feature in protective_features
        }
        if protective_mass > 0
        else {feature: 0.0 for feature in protective_features}
    )
    weights = CMCSWeights(
        dimensions=dimensions,
        sub_weights=sub_weights,
        safety_bonus=safety_bonus,
        source="data_driven_statistical",
    )
    return weights, {
        "derivation_formula": {
            "dimension_weight": (
                "W_dim = Σ_{j ∈ dim} w*_j / Σ_{j ∈ hazard_dims} w*_j"
            ),
            "sub_weight": (
                "w_sub(j|dim) = w*_j / Σ_{k ∈ dim} w*_k"
            ),
            "safety_bonus_cap": 0.15,
        },
        "dimension_evidence_mass": {
            key: round(value, 6) for key, value in dimension_mass.items()
        },
        "dimension_weights": {
            key: round(value, 6) for key, value in dimensions.items()
        },
        "sub_weights": {
            dimension: {
                feature: round(value, 6)
                for feature, value in values.items()
            }
            for dimension, values in sub_weights.items()
        },
        "safety_bonus_weights": {
            key: round(value, 6) for key, value in safety_bonus.items()
        },
        "safety_bonus_total": round(bonus_total, 6),
    }


# ---------------------------------------------------------------------------
# 직렬화 / 역직렬화
# ---------------------------------------------------------------------------

def _serialize_weights(weights: CMCSWeights) -> dict[str, object]:
    return {
        "source": weights.source,
        "dimensions": dict(weights.dimensions),
        "sub_weights": {
            key: dict(value) for key, value in weights.sub_weights.items()
        },
        "safety_bonus": dict(weights.safety_bonus),
    }


def load_data_driven_weights(
    path: str | Path = DATA_DRIVEN_WEIGHTS_PATH,
) -> CMCSWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CMCSWeights(
        dimensions=payload["dimensions"],
        sub_weights=payload["sub_weights"],
        safety_bonus=payload["safety_bonus"],
        source=payload.get("source", "data_driven_statistical"),
    )


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def _plot_weight_evidence(
    combined: Mapping[str, float],
    weights: CMCSWeights,
    method_details: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))

    dimension_series = pd.Series(weights.dimensions).sort_values()
    dimension_series.plot.barh(ax=axes[0, 0], color="#2563eb")
    axes[0, 0].set_title("데이터 기반 CMCS 차원 가중치")
    axes[0, 0].set_xlabel("가중치")

    feature_series = pd.Series(combined).sort_values().tail(12)
    feature_series.plot.barh(ax=axes[0, 1], color="#16a34a")
    axes[0, 1].set_title("통합 통계 증거 상위 변수")
    axes[0, 1].set_xlabel("정규화 증거 비중")

    channel_names = {
        "spearman": "Spearman ρ",
        "poisson": "Poisson β",
        "logistic": "Logistic β",
        "morans_i": "Moran's I",
    }
    if "normalized_method_shares" in method_details:
        shares = method_details["normalized_method_shares"]
        channel_df = pd.DataFrame(shares).fillna(0)
        top_features = pd.Series(combined).nlargest(8).index.tolist()
        if top_features:
            subset = channel_df.loc[
                channel_df.index.intersection(top_features)
            ]
            if not subset.empty:
                subset.rename(columns=channel_names).plot.barh(
                    ax=axes[1, 0], stacked=True
                )
                axes[1, 0].set_title("채널별 증거 기여 (상위 8개 변수)")
                axes[1, 0].set_xlabel("정규화 비중")
                axes[1, 0].legend(fontsize=8)

    channel_totals = {}
    for method, scores in (method_details.get("normalized_method_shares") or {}).items():
        channel_totals[channel_names.get(method, method)] = sum(
            max(0, v) for v in scores.values()
        )
    if channel_totals:
        pd.Series(channel_totals).plot.bar(ax=axes[1, 1], color="#8b5cf6")
        axes[1, 1].set_title("채널별 총 증거량")
        axes[1, 1].set_ylabel("Σ 정규화 점수")
        axes[1, 1].tick_params(axis="x", rotation=0)

    figure.suptitle(
        "CMCS 가중치 통계적 도출 증거\n"
        "(Spearman → Poisson → Logistic → Moran's I 4-채널 합성)",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 메인 파이프라인: 전처리 → 4-채널 분석 → 합성 → 가중치 도출
# ---------------------------------------------------------------------------

def derive_data_driven_cmcs_weights(
    edge_features: pd.DataFrame,
    weights_path: str | Path = DATA_DRIVEN_WEIGHTS_PATH,
    report_path: str | Path = DATA_DRIVEN_REPORT_PATH,
    chart_path: str | Path = DATA_DRIVEN_CHART_PATH,
) -> tuple[CMCSWeights, dict[str, object]]:
    """전처리된 edge feature에서 4-채널 증거 합성으로 CMCS 가중치를 산출한다.

    파이프라인:
      Stage 0: API 전처리 + 공간 셀 집계 (prepare_weight_learning_table)
      Stage 1: Spearman 순위 상관 (단변량 비모수 관계)
      Stage 2: Poisson 회귀 (노출량 보정 효과 크기)
      Stage 3: Logistic 회귀 (이진 분류 표준화 계수)
      Stage 4: Bivariate Moran's I (공간 자기상관 검증)
      합성:   4-채널 L1 정규화 평균 → 차원/하위 가중치
    """
    table = prepare_weight_learning_table(edge_features)
    preprocessing_report = table.attrs.get("preprocessing_report", {})

    spearman_scores, spearman_details = _spearman_evidence(table)
    poisson_scores, poisson_details = _poisson_evidence(table)
    logistic_scores, logistic_details = _logistic_evidence(table)
    moran_scores, moran_details = _morans_evidence(table)

    method_scores = {
        "spearman": spearman_scores,
        "poisson": poisson_scores,
        "logistic": logistic_scores,
        "morans_i": moran_scores,
    }
    combined, normalized_methods = _combine_evidence(method_scores)
    weights, weight_details = _weights_from_combined_evidence(combined)

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(
        json.dumps(_serialize_weights(weights), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    method_viz_data = {"normalized_method_shares": {
        method: {key: round(value, 6) for key, value in scores.items()}
        for method, scores in normalized_methods.items()
    }}
    _plot_weight_evidence(combined, weights, method_viz_data, Path(chart_path))

    report: dict[str, object] = {
        "methodology": {
            "name": "4-channel normalized evidence synthesis",
            "not_ahp": True,
            "pipeline_stages": [
                "Stage 0: API preprocessing + spatial cell aggregation",
                "Stage 1: Spearman rank correlation (univariate)",
                "Stage 2: Poisson regression (exposure-adjusted rate)",
                "Stage 3: Logistic regression (binary classification)",
                "Stage 4: Bivariate Moran's I (spatial autocorrelation)",
                "Synthesis: equal-weight channel averaging + L1 normalization",
            ],
            "synthesis_formula": {
                "step_1_channel_normalization": (
                    "w̃_j^m = max(0, evidence_j^m) / "
                    "Σ_k max(0, evidence_k^m)"
                ),
                "step_2_cross_channel_average": (
                    "w̄_j = (1/M) Σ_m w̃_j^m  "
                    "(M = number of channels with positive evidence)"
                ),
                "step_3_final_normalization": (
                    "w*_j = w̄_j / Σ_k w̄_k"
                ),
                "dimension_weight": (
                    "W_dim = Σ_{j ∈ dim} w*_j / "
                    "Σ_{j ∈ hazard_dims} w*_j"
                ),
                "sub_weight": (
                    "w_sub(j|dim) = w*_j / Σ_{k ∈ dim} w*_k"
                ),
            },
            "channel_evidence_formulas": {
                "spearman": (
                    "evidence_j = max(0, sign_j × ρ_j) "
                    "× max(0, 1 − p_j/0.10)"
                ),
                "poisson": (
                    "evidence_j = max(0, sign_j × β̃_j) "
                    "× consistency_j × R²_pseudo"
                ),
                "logistic": (
                    "evidence_j = max(0, sign_j × β̃_j) "
                    "× consistency_j × reliability"
                ),
                "morans_i": (
                    "evidence_j = max(0, sign_j × I_xy) "
                    "× (1 − p_perm)"
                ),
            },
            "target_leakage_control": (
                "accident_count_norm and every accident-derived CMCS value "
                "are excluded from predictors"
            ),
        },
        "preprocessing": preprocessing_report,
        "dataset": {
            "spatial_unit": f"{WEIGHT_CELL_SIZE_M}m grid clipped by district",
            "region_count": int(len(table)),
            "positive_region_count": int(table["accident_label"].sum()),
            "districts": sorted(table["district"].unique().tolist()),
            "exposure": "summed road length per region",
        },
        "analyses": {
            "stage_1_spearman": {"features": spearman_details},
            "stage_2_poisson": poisson_details,
            "stage_3_logistic": logistic_details,
            "stage_4_morans_i": moran_details,
        },
        "normalized_method_shares": {
            method: {
                key: round(value, 6) for key, value in scores.items()
            }
            for method, scores in normalized_methods.items()
        },
        "evidence_channel_status": {
            method: {
                "included_in_synthesis": method in normalized_methods,
                "positive_evidence_sum": round(
                    sum(max(0.0, value) for value in scores.values()),
                    6,
                ),
            }
            for method, scores in method_scores.items()
        },
        "combined_feature_evidence": {
            key: round(value, 6) for key, value in combined.items()
        },
        "weights": weight_details,
        "artifacts": {
            "weights": str(weights_path),
            "chart": str(chart_path),
        },
        "limitations": [
            "관찰 자료의 상관관계는 인과효과를 의미하지 않는다.",
            "안전시설은 위험지역에 우선 설치되는 역인과성의 영향을 받을 수 있다.",
            "사고 다발지역 라벨이 희소하므로 정기 재학습이 필요하다.",
            "Poisson 노출량은 실측 보행량이 아니라 권역 내 도로 길이를 사용한다.",
            "4-채널 합성에서 채널 간 가중치를 동일(1/M)로 설정했으며, "
            "채널 간 차등 가중은 메타 분석 문헌이 축적된 후 적용 예정이다.",
        ],
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return weights, report
