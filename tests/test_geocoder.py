import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.geocoder import KakaoGeocoder


def test_no_key_returns_none():
    g = KakaoGeocoder(api_key="", cache_path=Path(tempfile.mktemp(suffix=".json")))
    assert g.geocode("대전광역시 유성구 대학로 1") is None
    assert not g.has_key


def test_cache_hit_returns_stored():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    g = KakaoGeocoder(api_key="", cache_path=tmp)
    g._cache["대전시청"] = (36.35, 127.38)
    assert g.geocode("대전시청") == (36.35, 127.38)


def test_cache_persists_across_instances():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    g1 = KakaoGeocoder(api_key="fake_key", cache_path=tmp)
    g1._cache["주소A"] = (36.40, 127.40)
    g1._save_cache()

    g2 = KakaoGeocoder(api_key="", cache_path=tmp)
    assert g2.geocode("주소A") == (36.40, 127.40)
    assert g2.cache_size == 1


def test_geocode_batch_no_key():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    g = KakaoGeocoder(api_key="", cache_path=tmp)
    results = g.geocode_batch(["주소1", "주소2"])
    assert results == [None, None]


def test_geocode_dataframe_adds_columns():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    g = KakaoGeocoder(api_key="", cache_path=tmp)
    g._cache["주소A"] = (36.35, 127.38)
    df = pd.DataFrame({"name": ["학원A"], "address": ["주소A"]})
    out = g.geocode_dataframe(df, "address")
    assert "geocoded_lat" in out.columns
    assert "geocoded_lon" in out.columns
    assert out.loc[0, "geocoded_lat"] == 36.35
    assert out.loc[0, "geocoded_lon"] == 127.38


def test_geocode_dataframe_none_when_not_found():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    g = KakaoGeocoder(api_key="", cache_path=tmp)
    df = pd.DataFrame({"address": ["없는주소"]})
    out = g.geocode_dataframe(df, "address")
    assert out.loc[0, "geocoded_lat"] is None
