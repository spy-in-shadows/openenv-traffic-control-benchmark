from __future__ import annotations


EPS = 1e-6
SERIALIZATION_EPS = 1e-2


def strict_open_score(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return SERIALIZATION_EPS
    if x >= 1.0:
        return 1.0 - SERIALIZATION_EPS
    return max(SERIALIZATION_EPS, min(1.0 - SERIALIZATION_EPS, x))
