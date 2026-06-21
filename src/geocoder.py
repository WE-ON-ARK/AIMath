"""학원·교습소 주소 지오코딩 (Kakao REST API, 파일 캐시)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


KAKAO_API_URL = "https://dapi.kakao.com/v2/local/search/address.json"
_DEFAULT_CACHE = Path("cache/geocoder_academy.json")


class KakaoGeocoder:
    """주소 → (위도, 경도) 변환. API 키 없으면 캐시만 사용."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | Path = _DEFAULT_CACHE,
    ) -> None:
        self._key = api_key or os.getenv("KAKAO_API_KEY") or ""
        self._cache_path = Path(cache_path)
        self._cache: dict[str, tuple[float, float] | None] = self._load_cache()

    def _load_cache(self) -> dict[str, tuple[float, float] | None]:
        if self._cache_path.exists():
            try:
                raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                return {k: tuple(v) if v else None for k, v in raw.items()}  # type: ignore[misc]
            except Exception:
                pass
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def geocode(self, address: str) -> tuple[float, float] | None:
        """주소 1건 → (위도, 경도) 또는 None."""
        if address in self._cache:
            return self._cache[address]
        if not self._key:
            return None
        try:
            response = requests.get(
                KAKAO_API_URL,
                headers={"Authorization": f"KakaoAK {self._key}"},
                params={"query": address, "analyze_type": "similar", "size": 1},
                timeout=10,
            )
            response.raise_for_status()
            documents = response.json().get("documents", [])
            if documents:
                doc = documents[0]
                result: tuple[float, float] | None = (
                    float(doc["y"]),
                    float(doc["x"]),
                )
            else:
                result = None
        except Exception:
            result = None
        self._cache[address] = result
        self._save_cache()
        return result

    def geocode_batch(
        self,
        addresses: list[str],
        sleep_s: float = 0.05,
    ) -> list[tuple[float, float] | None]:
        results = []
        for addr in addresses:
            results.append(self.geocode(addr))
            if self._key:
                time.sleep(sleep_s)
        return results

    def geocode_dataframe(
        self,
        df: pd.DataFrame,
        address_col: str,
        lat_col: str = "geocoded_lat",
        lon_col: str = "geocoded_lon",
    ) -> pd.DataFrame:
        """주소 컬럼을 지오코딩해 위도·경도 컬럼 추가."""
        out = df.copy()
        coords = self.geocode_batch(df[address_col].astype(str).tolist())
        out[lat_col] = [c[0] if c else None for c in coords]
        out[lon_col] = [c[1] if c else None for c in coords]
        return out

    @property
    def has_key(self) -> bool:
        return bool(self._key)

    @property
    def cache_size(self) -> int:
        return len(self._cache)
