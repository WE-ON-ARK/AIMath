"""경로 추천 검증 — OD 일괄 평가, 우회 제한, 시간대·연령별 프로파일."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from config import REPORT_OUTPUT_DIR


# ---------------------------------------------------------------------------
# 연령대별 경로 프로파일
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgeProfile:
    label: str
    lam: float          # balanced_route의 λ (0=안전 최우선, 1=거리 최우선)
    max_detour_ratio: float   # 최단 거리 대비 허용 우회 배율
    description: str

AGE_PROFILES: dict[str, AgeProfile] = {
    "low":    AgeProfile("초등 1-2학년", lam=0.20, max_detour_ratio=2.0,
                         description="안전 최우선 — 위험 회피를 강하게 반영"),
    "mid":    AgeProfile("초등 3-4학년", lam=0.40, max_detour_ratio=1.6,
                         description="안전·거리 균형"),
    "high":   AgeProfile("초등 5-6학년", lam=0.55, max_detour_ratio=1.4,
                         description="거리 비중 소폭 상승"),
    "middle": AgeProfile("중학생",       lam=0.70, max_detour_ratio=1.25,
                         description="합리적 경로 선택"),
}


def age_profile(age_group: str) -> AgeProfile:
    key = age_group.lower().strip()
    if key not in AGE_PROFILES:
        raise ValueError(f"age_group은 {list(AGE_PROFILES)} 중 하나여야 합니다.")
    return AGE_PROFILES[key]


# ---------------------------------------------------------------------------
# 시간대별 CMCS 가중치 조정
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeWeights:
    hour: int
    light_multiplier: float   # 야간은 조명 부재 패널티 강화
    congestion_multiplier: float  # 등하교 시간대 혼잡 강화

    @property
    def label(self) -> str:
        if 7 <= self.hour < 9:
            return "등교 러시아워"
        if 9 <= self.hour < 16:
            return "낮 시간"
        if 16 <= self.hour < 19:
            return "하교 러시아워"
        return "야간"


def time_weights(hour: int) -> TimeWeights:
    """시간대(0-23)에 따라 조명·혼잡 가중치 배율을 반환한다."""
    if 0 <= hour < 6 or hour >= 21:        # 야간
        return TimeWeights(hour, light_multiplier=1.8, congestion_multiplier=0.6)
    if 7 <= hour < 9 or 16 <= hour < 19:  # 러시아워
        return TimeWeights(hour, light_multiplier=1.0, congestion_multiplier=1.5)
    return TimeWeights(hour, light_multiplier=1.0, congestion_multiplier=1.0)


def apply_time_weights_to_cmcs(
    cmcs_series: pd.Series,
    tw: TimeWeights,
    light_density_col: pd.Series | None = None,
) -> pd.Series:
    """시간대 가중치로 CMCS 값을 조정한다.

    야간에는 조명 부재(light deficit)를 강화하고,
    러시아워에는 혼잡도를 강화한다.
    """
    adjusted = cmcs_series.copy().clip(0, 1)
    if light_density_col is not None and tw.light_multiplier != 1.0:
        light_deficit = 1.0 - light_density_col.clip(0, 1)
        delta = (tw.light_multiplier - 1.0) * light_deficit * 0.15
        adjusted = (adjusted + delta).clip(0, 1)
    if tw.congestion_multiplier != 1.0:
        congestion_factor = (tw.congestion_multiplier - 1.0) * 0.05
        adjusted = (adjusted + congestion_factor).clip(0, 1)
    return adjusted


# ---------------------------------------------------------------------------
# 우회율 제한
# ---------------------------------------------------------------------------

def check_detour_ratio(
    shortest_m: float,
    candidate_m: float,
    max_ratio: float = 2.0,
) -> tuple[float, bool]:
    """실제 우회율과 제한 초과 여부를 반환한다."""
    if shortest_m <= 0:
        return 0.0, False
    ratio = candidate_m / shortest_m
    return round(ratio, 4), ratio > max_ratio


# ---------------------------------------------------------------------------
# OD 쌍 일괄 평가
# ---------------------------------------------------------------------------

@dataclass
class ODResult:
    origin_label: str
    destination_label: str
    shortest_m: float
    safest_m: float
    balanced_m: float
    detour_ratio: float
    detour_exceeded: bool
    risk_reduction_pct: float
    shortest_cmcs: float
    safest_cmcs: float


def batch_od_evaluation(
    optimizer: object,
    od_pairs: Sequence[tuple[object, object, str, str]],
    age_group: str = "mid",
    hour: int = 8,
    max_detour_ratio: float | None = None,
    output_path: str | Path = REPORT_OUTPUT_DIR / "od_evaluation_report.json",
) -> pd.DataFrame:
    """학교-학원 OD 쌍을 일괄 평가한다.

    od_pairs: (origin, destination, origin_label, destination_label) 목록.
    origin/destination은 (lat, lon) 튜플 또는 그래프 노드 ID.
    """
    from src.route_optimizer import RouteOptimizer
    import networkx as nx

    profile = age_profile(age_group)
    if max_detour_ratio is None:
        max_detour_ratio = profile.max_detour_ratio

    rows: list[dict] = []
    skipped = 0
    for origin, destination, origin_label, dest_label in od_pairs:
        try:
            shortest = optimizer.shortest_route(origin, destination)
            distance_limit = (
                shortest["total_distance_m"] * max_detour_ratio
            )
            safest = optimizer.safest_route(
                origin,
                destination,
                max_distance_m=distance_limit,
            )
            balanced = optimizer.balanced_route(
                origin,
                destination,
                profile.lam,
                max_distance_m=distance_limit,
            )
        except Exception:
            skipped += 1
            continue

        s_m = shortest["total_distance_m"]
        f_m = safest["total_distance_m"]
        b_m = balanced["total_distance_m"]
        ratio, exceeded = check_detour_ratio(s_m, f_m, max_detour_ratio)
        risk_reduction = 0.0
        if shortest["total_cmcs"] > 0:
            risk_reduction = (
                (shortest["total_cmcs"] - safest["total_cmcs"])
                / shortest["total_cmcs"]
            ) * 100.0

        rows.append({
            "origin": origin_label,
            "destination": dest_label,
            "shortest_m": round(s_m, 1),
            "safest_m": round(f_m, 1),
            "balanced_m": round(b_m, 1),
            "detour_ratio": ratio,
            "detour_exceeded": exceeded,
            "risk_reduction_pct": round(risk_reduction, 2),
            "shortest_cmcs": round(shortest["total_cmcs"], 4),
            "safest_cmcs": round(safest["total_cmcs"], 4),
            "age_group": age_group,
            "hour": hour,
            "algorithm": shortest["algorithm"],
            "shortest_runtime_ms": shortest["search_stats"]["runtime_ms"],
            "safest_runtime_ms": safest["search_stats"]["runtime_ms"],
            "balanced_runtime_ms": balanced["search_stats"]["runtime_ms"],
            "shortest_ants_total": shortest["search_stats"]["ants_total"],
            "safest_ants_total": safest["search_stats"]["ants_total"],
            "balanced_ants_total": balanced["search_stats"]["ants_total"],
            "shortest_labels_expanded": shortest["search_stats"][
                "labels_expanded"
            ],
            "safest_labels_expanded": safest["search_stats"]["labels_expanded"],
            "balanced_labels_expanded": balanced["search_stats"][
                "labels_expanded"
            ],
            "pure_aco_success_rate": float(
                np.mean(
                    [
                        route["aco_stats"]["pure_aco_feasible_solutions"]
                        / max(1, route["aco_stats"]["ants_total"])
                        for route in (shortest, safest, balanced)
                    ]
                )
            ),
            "seeded_aco_success_rate": float(
                np.mean(
                    [
                        route["aco_stats"]["seeded_feasible_solutions"]
                        / max(1, route["aco_stats"]["ants_total"])
                        for route in (shortest, safest, balanced)
                    ]
                )
            ),
            "combined_candidate_success_rate": float(
                np.mean(
                    [
                        (
                            route["aco_stats"]["pure_aco_feasible_solutions"]
                            + route["aco_stats"]["seeded_feasible_solutions"]
                        )
                        / max(1, route["aco_stats"]["ants_total"])
                        for route in (shortest, safest, balanced)
                    ]
                )
            ),
            "aco_success_rate": float(
                np.mean(
                    [
                        (
                            route["aco_stats"]["pure_aco_feasible_solutions"]
                            + route["aco_stats"]["seeded_feasible_solutions"]
                        )
                        / max(1, route["aco_stats"]["ants_total"])
                        for route in (shortest, safest, balanced)
                    ]
                )
            ),
            "pure_aco_feasible_solutions": int(
                sum(
                    route["aco_stats"]["pure_aco_feasible_solutions"]
                    for route in (shortest, safest, balanced)
                )
            ),
            "seeded_feasible_solutions": int(
                sum(
                    route["aco_stats"]["seeded_feasible_solutions"]
                    for route in (shortest, safest, balanced)
                )
            ),
            "aco_found_feasible": any(
                route["aco_stats"]["aco_found_feasible"]
                for route in (shortest, safest, balanced)
            ),
            "rcsp_certified": all(
                route["selected_source"] == "rcsp_certified"
                and route["optimality_proven"]
                for route in (shortest, safest, balanced)
            ),
            "optimality_proven": all(
                route["optimality_proven"] for route in (shortest, safest, balanced)
            ),
            "mean_gap_pct": float(
                np.mean(
                    [
                        route["gap_pct"]
                        for route in (shortest, safest, balanced)
                        if route["gap_pct"] is not None
                    ]
                    or [0.0]
                )
            ),
        })

    df = pd.DataFrame(rows)
    report: dict = {
        "total_pairs": len(od_pairs),
        "evaluated": len(df),
        "skipped": skipped,
        "age_group": age_group,
        "age_profile": {
            "label": profile.label,
            "lam": profile.lam,
            "max_detour_ratio": profile.max_detour_ratio,
        },
        "hour": hour,
        "time_label": time_weights(hour).label,
        "max_detour_ratio": max_detour_ratio,
        "detour_exceeded_count": int(df["detour_exceeded"].sum()) if len(df) else 0,
        "mean_detour_ratio": round(float(df["detour_ratio"].mean()), 4) if len(df) else 0,
        "mean_risk_reduction_pct": round(float(df["risk_reduction_pct"].mean()), 2) if len(df) else 0,
        "algorithm": df["algorithm"].iloc[0] if len(df) else "aco_pareto_rcsp",
        "pure_aco_success_rate": round(
            float(df["pure_aco_success_rate"].mean()), 6
        )
        if len(df)
        else 0.0,
        "seeded_aco_success_rate": round(
            float(df["seeded_aco_success_rate"].mean()), 6
        )
        if len(df)
        else 0.0,
        "combined_candidate_success_rate": round(
            float(df["combined_candidate_success_rate"].mean()), 6
        )
        if len(df)
        else 0.0,
        "aco_success_rate": round(float(df["aco_success_rate"].mean()), 6)
        if len(df)
        else 0.0,
        "aco_success_rate_note": "seeded included legacy compatibility field",
        "mean_total_runtime_ms": round(
            float(
                df[
                    [
                        "shortest_runtime_ms",
                        "safest_runtime_ms",
                        "balanced_runtime_ms",
                    ]
                ].sum(axis=1).mean()
            ),
            4,
        )
        if len(df)
        else 0,
        "p95_total_runtime_ms": round(
            float(
                df[
                    [
                        "shortest_runtime_ms",
                        "safest_runtime_ms",
                        "balanced_runtime_ms",
                    ]
                ].sum(axis=1).quantile(0.95)
            ),
            4,
        )
        if len(df)
        else 0,
        "all_optimality_proven": bool(
            df["optimality_proven"].all()
        )
        if len(df)
        else False,
        "pairs": df.to_dict(orient="records"),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return df
