from api.reference_value_utils import _non_negative_float, _positive_int


def test_positive_int_accepts_positive_values_only():
    assert _positive_int("3") == 3
    assert _positive_int(2.7) == 2
    assert _positive_int(0) == 0
    assert _positive_int(-1) == 0
    assert _positive_int("bad") == 0


def test_non_negative_float_accepts_positive_values_only():
    assert _non_negative_float("3.5") == 3.5
    assert _non_negative_float(0) == 0.0
    assert _non_negative_float(-0.1) == 0.0
    assert _non_negative_float("bad") == 0.0
