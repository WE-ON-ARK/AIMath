"""통계·공간 증거를 결합해 CMCS 가중치를 재현 가능하게 산출한다."""
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
    """방향 중복을 제거하고 자치구 경계로 분리된 공간 셀로 집계한다."""
    segments = edge_features.drop_duplicates("segment_id").copy()
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
            # 도로 수에 따라 사고가 중복 합산되지 않도록 셀의 최대 이력값 사용.
            "accident_count": "max",
            "accident_label": "max",
        }
    )
    return (
        working.groupby(keys, as_index=False)
        .agg(aggregation)
        .rename(
            columns={
                "segment_id": "segment_count",
                "length_m": "exposure_length_m",
            }
        )
    )


def _spatial_groups(table: pd.DataFrame) -> pd.Series:
    return table["district"].astype(str)


def _spearman_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
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
            "evidence_score": round(score, 6),
        }
    return scores, details


def _logistic_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object]]:
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
        "validation": "LeaveOneGroupOut by district",
        "roc_auc": round(auc, 6),
        "average_precision": round(ap, 6),
        "reliability_multiplier": round(reliability, 6),
        "features": feature_details,
    }


def _poisson_evidence(
    table: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, object]]:
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
        "model": "PoissonRegressor on accident rate",
        "exposure": "road length in km via sample_weight",
        "validation": "LeaveOneGroupOut by district",
        "heldout_weighted_pseudo_r2": round(pseudo_r2, 6),
        "features": feature_details,
    }


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
        "neighbors": int(neighbors.shape[1]),
        "permutations": permutations,
        "target_univariate_morans_i": round(target_i, 6),
        "features": feature_details,
    }


def _normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    positive = {key: max(0.0, float(value)) for key, value in scores.items()}
    total = sum(positive.values())
    if total <= 0:
        return {key: 0.0 for key in positive}
    return {key: value / total for key, value in positive.items()}


def _combine_evidence(
    method_scores: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
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


def _weights_from_combined_evidence(
    combined: Mapping[str, float],
) -> tuple[CMCSWeights, dict[str, object]]:
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
    # 차감항이 전체 위험 점수를 역전하지 않도록 0.15를 수학적 상한으로 둔다.
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
        "safety_bonus_cap": 0.15,
    }


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


def _plot_weight_evidence(
    combined: Mapping[str, float],
    weights: CMCSWeights,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    dimension_series = pd.Series(weights.dimensions).sort_values()
    dimension_series.plot.barh(ax=axes[0], color="#2563eb")
    axes[0].set_title("데이터 기반 CMCS 차원 가중치")
    axes[0].set_xlabel("가중치")
    feature_series = pd.Series(combined).sort_values().tail(12)
    feature_series.plot.barh(ax=axes[1], color="#16a34a")
    axes[1].set_title("통합 통계 증거 상위 변수")
    axes[1].set_xlabel("정규화 증거 비중")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def derive_data_driven_cmcs_weights(
    edge_features: pd.DataFrame,
    weights_path: str | Path = DATA_DRIVEN_WEIGHTS_PATH,
    report_path: str | Path = DATA_DRIVEN_REPORT_PATH,
    chart_path: str | Path = DATA_DRIVEN_CHART_PATH,
) -> tuple[CMCSWeights, dict[str, object]]:
    table = prepare_weight_learning_table(edge_features)
    spearman_scores, spearman_details = _spearman_evidence(table)
    logistic_scores, logistic_details = _logistic_evidence(table)
    poisson_scores, poisson_details = _poisson_evidence(table)
    moran_scores, moran_details = _morans_evidence(table)
    method_scores = {
        "spearman": spearman_scores,
        "logistic": logistic_scores,
        "poisson": poisson_scores,
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
    _plot_weight_evidence(combined, weights, Path(chart_path))
    report: dict[str, object] = {
        "methodology": {
            "name": "equal-channel normalized evidence synthesis",
            "not_ahp": True,
            "formula": {
                "signed_effect": (
                    "expected_direction_j × estimated_effect_method,j"
                ),
                "method_share": (
                    "max(0, signed_effect × reliability) / "
                    "Σ_k max(0, signed_effect_k × reliability_k)"
                ),
                "feature_weight": (
                    "arithmetic mean of available method shares, "
                    "renormalized to sum 1"
                ),
                "dimension_weight": (
                    "Σ feature_weight in dimension / "
                    "Σ hazard feature_weight"
                ),
                "sub_weight": (
                    "feature_weight / Σ feature_weight within dimension"
                ),
            },
            "channels": [
                "Spearman rank association with accident count",
                "district-holdout standardized Logistic coefficient",
                "road-length exposure adjusted Poisson coefficient",
                "permutation-tested bivariate Moran's I",
            ],
            "target_leakage_control": (
                "accident_count_norm and every accident-derived CMCS value "
                "are excluded from predictors"
            ),
        },
        "dataset": {
            "spatial_unit": f"{WEIGHT_CELL_SIZE_M}m grid clipped by district",
            "region_count": int(len(table)),
            "positive_region_count": int(table["accident_label"].sum()),
            "districts": sorted(table["district"].unique().tolist()),
            "exposure": "summed road length per region",
        },
        "analyses": {
            "spearman": {"features": spearman_details},
            "logistic": logistic_details,
            "poisson": poisson_details,
            "morans_i": moran_details,
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
            "사고 다발지역 22건에서 생성된 희소 라벨이므로 정기 재학습이 필요하다.",
            "Poisson 노출량은 실측 보행량이 아니라 권역 내 도로 길이를 사용한다.",
        ],
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return weights, report
