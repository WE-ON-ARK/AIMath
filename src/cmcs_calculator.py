from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CMCSWeights:
    """CMCS 선형 결합 가중치.

    기본값은 과거의 휴리스틱 설계와 호환하기 위한 값이며 실제 AHP 결과가
    아니다. 통계 기반 운영 가중치는 ``src.data_driven_cmcs``에서 산출한다.
    """

    dimensions: Mapping[str, float] = field(
        default_factory=lambda: {
            "risk": 0.35,
            "discomfort": 0.15,
            "congestion": 0.15,
            "obstruction": 0.15,
            "crossing": 0.20,
        }
    )
    sub_weights: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "risk": {
                "accident_count_norm": 0.50,
                "traffic_volume_norm": 0.30,
                "avg_speed_norm": 0.20,
            },
            "discomfort": {
                "narrow_sidewalk_norm": 0.50,
                "slope_norm": 0.30,
                "is_alley": 0.20,
            },
            "congestion": {
                "pedestrian_flow_norm": 0.40,
                "academy_density_norm": 0.40,
                "bus_stop_nearby_norm": 0.20,
            },
            "obstruction": {
                "illegal_parking_norm": 0.60,
                "light_deficit": 0.40,
            },
            "crossing": {
                "crosswalk_deficit": 0.40,
                "signal_deficit": 0.40,
                "lane_count_norm": 0.20,
            },
        }
    )
    safety_bonus: Mapping[str, float] = field(
        default_factory=lambda: {
            "has_speed_bump": 0.05,
            "has_cctv": 0.03,
            "is_school_zone": 0.04,
        }
    )
    source: str = "manual_heuristic_legacy"


class CMCSCalculator:
    """명시적으로 제공된 가중치로 CMCS 점수와 구성 차원을 계산한다."""

    def __init__(self, weights: CMCSWeights | None = None):
        self.weights = weights or CMCSWeights()
        self._validate_weights()

    def _validate_weights(self) -> None:
        if not np.isclose(sum(self.weights.dimensions.values()), 1.0):
            raise ValueError("CMCS 차원 가중치의 합은 1이어야 합니다.")
        for dimension, weights in self.weights.sub_weights.items():
            if not np.isclose(sum(weights.values()), 1.0):
                raise ValueError(f"{dimension} 하위 가중치의 합은 1이어야 합니다.")

    @staticmethod
    def _series(
        features: pd.DataFrame, column: str, default: float = 0.0
    ) -> pd.Series:
        if column in features:
            values = pd.to_numeric(features[column], errors="coerce").fillna(default)
            return values.clip(lower=0.0, upper=1.0)
        return pd.Series(default, index=features.index, dtype=float)

    def calculate_components(self, features: pd.DataFrame) -> pd.DataFrame:
        sw = self.weights.sub_weights

        light_deficit = 1.0 - self._series(features, "light_density_norm", 0.5)
        crosswalk_deficit = 1.0 - self._series(features, "has_crosswalk")
        signal_deficit = 1.0 - self._series(features, "has_signal")

        derived = {
            "light_deficit": light_deficit,
            "crosswalk_deficit": crosswalk_deficit,
            "signal_deficit": signal_deficit,
        }
        components: dict[str, pd.Series] = {}
        for dimension, variable_weights in sw.items():
            value = pd.Series(0.0, index=features.index)
            for variable, weight in variable_weights.items():
                series = derived.get(variable)
                if series is None:
                    series = self._series(features, variable)
                value = value + weight * series
            components[dimension] = value

        safety_bonus = pd.Series(0.0, index=features.index)
        for variable, weight in self.weights.safety_bonus.items():
            safety_bonus = safety_bonus + weight * self._series(features, variable)
        components["safety_bonus"] = safety_bonus
        return pd.DataFrame(components, index=features.index)

    def calculate_cmcs(self, features: pd.DataFrame) -> pd.Series:
        components = self.calculate_components(features)
        cmcs = pd.Series(0.0, index=features.index)
        for dimension, weight in self.weights.dimensions.items():
            cmcs = cmcs + weight * components[dimension]
        cmcs = cmcs - components["safety_bonus"]
        return cmcs.clip(lower=0.0, upper=1.0).rename("cmcs")

    def calculate_cmcs_ahp(self, features: pd.DataFrame) -> pd.Series:
        """하위 호환 별칭. 이 함수명은 실제 AHP 검증을 의미하지 않는다."""
        return self.calculate_cmcs(features)

    def score(self, features: pd.DataFrame, include_components: bool = True) -> pd.DataFrame:
        result = features.copy()
        if include_components:
            components = self.calculate_components(features).add_prefix("cmcs_")
            result = pd.concat([result, components], axis=1)
        result["cmcs"] = self.calculate_cmcs(features)
        result["cmcs_weight_source"] = self.weights.source
        return result

    def fit_data_driven_weights(
        self,
        features: pd.DataFrame,
        accident_label: pd.Series,
        feature_columns: list[str] | None = None,
    ) -> dict[str, float]:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise RuntimeError(
                "데이터 기반 가중치 학습에는 scikit-learn이 필요합니다."
            ) from exc

        columns = feature_columns or [
            "accident_count_norm",
            "traffic_volume_norm",
            "avg_speed_norm",
            "narrow_sidewalk_norm",
            "slope_norm",
            "is_alley",
            "pedestrian_flow_norm",
            "academy_density_norm",
            "bus_stop_nearby_norm",
            "illegal_parking_norm",
            "light_density_norm",
            "has_crosswalk",
            "has_signal",
            "lane_count_norm",
            "has_speed_bump",
            "has_cctv",
            "is_school_zone",
        ]
        columns = list(dict.fromkeys(columns))
        X = pd.DataFrame(
            {column: self._series(features, column) for column in columns},
            index=features.index,
        )
        y = (pd.to_numeric(accident_label, errors="coerce").fillna(0) > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("사고 레이블에는 최소 두 개의 클래스가 필요합니다.")

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X, y)
        return dict(zip(columns, model.coef_[0].astype(float)))
