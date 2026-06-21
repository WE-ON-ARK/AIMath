from src.full_pipeline import (
    _highway_value,
    _parse_number,
    canonical_segment_id,
)


def test_canonical_segment_id_is_direction_independent():
    assert canonical_segment_id(10, 20, 123) == canonical_segment_id(20, 10, 123)


def test_parse_osm_tag_values():
    assert _parse_number("50 mph", 30) == 50
    assert _parse_number(["2", "3"], 1) == 2
    assert _highway_value(["residential", "service"]) == "residential"

