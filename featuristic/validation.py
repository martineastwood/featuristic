"""Shared validation helpers for public estimator parameters and training data."""

from numbers import Integral, Real


def positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
