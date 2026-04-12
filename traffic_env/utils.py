from __future__ import annotations


EPS = 1e-6


def strict_open_score(x: float) -> float:
    x = float(x)
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return x
