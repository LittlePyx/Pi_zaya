from __future__ import annotations


def _positive_int(x) -> int:
    try:
        v = int(x)
    except Exception:
        return 0
    return v if v > 0 else 0


def _non_negative_float(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return v if v > 0.0 else 0.0
