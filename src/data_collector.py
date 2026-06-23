from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import pandas as pd
import requests

from src.api_preprocessing import (
    preprocess_api_frame,
    update_api_preprocessing_report,
)


@dataclass(frozen=True)
class APIEndpoint:
    path: str
    params: Mapping[str, Any] = field(default_factory=dict)
    page_parameter: str = "pageNo"
    page_size_parameter: str = "numOfRows"
    page_size: int = 1000


DEFAULT_ENDPOINTS: dict[str, APIEndpoint] = {
    "school_zone": APIEndpoint(
        path="/B552061/schoolzoneService/getSchoolzoneList",
        params={"type": "json", "sidoNm": "대전"},
    ),
    "traffic_accident": APIEndpoint(
        path="/B552061/AccidentDeath/getRestTrafficAccident",
        params={"type": "json", "searchYear": 2023, "siDo": "대전"},
    ),
    "schoolzone_child_hotspot": APIEndpoint(
        path="/B552061/schoolzoneChild/getRestSchoolzoneChild",
        params={"type": "json", "siDo": "30", "guGun": ""},
        page_size=100,
    ),
}


class DataCollector:
    """공공데이터포털 JSON API 수집기.

    데이터셋별 세부 엔드포인트는 활용신청한 API 명세에 따라 등록한다.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://apis.data.go.kr",
        raw_dir: str | Path = "data/raw",
        request_interval_seconds: float = 0.5,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.request_interval_seconds = request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @staticmethod
    def _extract_body(payload: dict) -> dict:
        response = payload.get("response", payload)
        header = response.get("header", {}) if isinstance(response, dict) else {}
        result_code = str(header.get("resultCode", "00"))
        if result_code not in {"00", "0", "NORMAL_SERVICE"}:
            message = header.get("resultMsg", "공공데이터 API 오류")
            raise RuntimeError(f"{result_code}: {message}")
        body = response.get("body", response) if isinstance(response, dict) else {}
        return body if isinstance(body, dict) else {}

    @staticmethod
    def _extract_items(body: dict) -> list[dict]:
        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", items)
        if not items:
            return []
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return [items] if isinstance(items, dict) else []

    def fetch_dataset(
        self,
        name: str,
        endpoint: APIEndpoint,
        parameter_overrides: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError(
                "DATA_GO_KR_API_KEY가 비어 있습니다. .env.example을 참고해 설정하세요."
            )

        page = 1
        records: list[dict] = []
        while True:
            params = dict(endpoint.params)
            params.update(parameter_overrides or {})
            params.update(
                {
                    "serviceKey": self.api_key,
                    endpoint.page_parameter: page,
                    endpoint.page_size_parameter: endpoint.page_size,
                }
            )
            response = self.session.get(
                f"{self.base_url}{endpoint.path}",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except requests.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{name} API가 JSON이 아닌 응답을 반환했습니다."
                ) from exc

            body = self._extract_body(payload)
            items = self._extract_items(body)
            records.extend(items)

            total_count = int(body.get("totalCount", len(records)) or 0)
            if not items or len(records) >= total_count:
                break
            page += 1
            time.sleep(self.request_interval_seconds)

        result, preprocessing_report = preprocess_api_frame(
            pd.DataFrame(records),
            name,
        )
        result.to_csv(self.raw_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        update_api_preprocessing_report(preprocessing_report)
        return result

    def fetch_school_zone(self) -> pd.DataFrame:
        return self.fetch_dataset("school_zone", DEFAULT_ENDPOINTS["school_zone"])

    def fetch_traffic_accident(self, year: int = 2023) -> pd.DataFrame:
        return self.fetch_dataset(
            "traffic_accident",
            DEFAULT_ENDPOINTS["traffic_accident"],
            {"searchYear": year},
        )

    def fetch_schoolzone_child_hotspots(
        self,
        years: Iterable[int] = range(2012, 2025),
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        endpoint = DEFAULT_ENDPOINTS["schoolzone_child_hotspot"]
        records: list[dict[str, Any]] = []
        for year in years:
            page = 1
            while True:
                params = {
                    **endpoint.params,
                    "serviceKey": self.api_key,
                    "searchYearCd": int(year),
                    endpoint.page_parameter: page,
                    endpoint.page_size_parameter: endpoint.page_size,
                }
                response = self.session.get(
                    f"{self.base_url}{endpoint.path}",
                    params=params,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                body = self._extract_body(payload)
                items = self._extract_items(body)
                for item in items:
                    item["search_year"] = int(year)
                records.extend(items)
                total_count = int(body.get("totalCount", len(records)) or 0)
                if not items or page * endpoint.page_size >= total_count:
                    break
                page += 1
                time.sleep(self.request_interval_seconds)
        clean, report = preprocess_api_frame(
            pd.DataFrame(records),
            "traffic_accident",
        )
        path = Path(output_path) if output_path else (
            self.raw_dir / "daejeon_schoolzone_accident_hotspots.csv"
        )
        clean.to_csv(path, index=False, encoding="utf-8-sig")
        update_api_preprocessing_report(report)
        return clean

    def fetch_all(
        self, endpoints: Mapping[str, APIEndpoint] | None = None
    ) -> dict[str, pd.DataFrame]:
        datasets: dict[str, pd.DataFrame] = {}
        for name, endpoint in (endpoints or DEFAULT_ENDPOINTS).items():
            try:
                datasets[name] = self.fetch_dataset(name, endpoint)
            except Exception as exc:
                print(f"[수집 실패] {name}: {exc}")
        return datasets


def build_pedestrian_network(
    place: str = "Daejeon, South Korea",
    save_path: str | Path = "data/graph/daejeon_walk.graphml",
) -> nx.MultiDiGraph:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("도로망 다운로드에는 osmnx가 필요합니다.") from exc

    graph = ox.graph_from_place(place, network_type="walk")
    for index, (_, _, _, data) in enumerate(graph.edges(keys=True, data=True)):
        data.setdefault("edge_id", f"E{index:07d}")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=path)
    return graph
