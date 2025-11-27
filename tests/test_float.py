from quantization_playground.evaluation import evaluate_float
from quantization_playground.maps import ensure_multiplication_maps


def test_interpolated_signed_ext_stays_in_range():
    maps = ensure_multiplication_maps([16], map_type="signed_ext")
    _, _, _, mapped_value, error_percent = evaluate_float(1.5, -1.0, maps[16], map_type="signed_ext", method="interpolated")
    assert -2.0 <= mapped_value <= 2.0
    assert error_percent >= 0


def test_nearest_neighbor_handles_small_range():
    maps = ensure_multiplication_maps([16], map_type="signed_log", max_range=0.8)
    _, _, _, mapped_value, error_percent = evaluate_float(0.4, 0.5, maps[16], map_type="signed_log", method="nearest", float_range=(-0.8, 0.8), stochastic_round=True)
    assert -0.8 <= mapped_value <= 0.8
    assert error_percent >= 0
