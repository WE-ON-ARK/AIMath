import numpy as np

from src.real_data_pipeline import extract_district, haversine_matrix


def test_extract_district_from_daejeon_address():
    assert extract_district("대전광역시 유성구 대학로 1") == "유성구"
    assert extract_district("주소 없음") is None


def test_haversine_matrix_zero_and_known_scale():
    distances = haversine_matrix(
        [36.35, 36.35],
        [127.38, 127.38],
        [36.35, 36.351],
        [127.38, 127.38],
    )
    assert distances.shape == (2, 2)
    assert distances[0, 0] == 0
    assert 100 < distances[0, 1] < 120
    assert np.isfinite(distances).all()

