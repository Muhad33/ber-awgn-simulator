"""Tests for the Gray-coded mappers and demappers."""

import numpy as np
import pytest

from src.modulation import BITS_PER_SYMBOL, constellation, demodulate, modulate

SCHEMES = list(BITS_PER_SYMBOL)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_noiseless_roundtrip(scheme: str) -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=BITS_PER_SYMBOL[scheme] * 10_000, dtype=np.int8)
    assert np.array_equal(demodulate(modulate(bits, scheme), scheme), bits)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_unit_average_symbol_energy(scheme: str) -> None:
    points, _ = constellation(scheme)
    assert np.mean(np.abs(points) ** 2) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_constellation_size_and_unique_points(scheme: str) -> None:
    points, labels = constellation(scheme)
    assert len(points) == 2 ** BITS_PER_SYMBOL[scheme]
    assert len(np.unique(np.round(points, 9))) == len(points)
    assert len({tuple(lbl) for lbl in labels}) == len(points)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gray_coding_nearest_neighbours_differ_by_one_bit(scheme: str) -> None:
    points, labels = constellation(scheme)
    dist = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(dist, np.inf)
    d_min = dist.min()
    neighbours = np.argwhere(np.isclose(dist, d_min))
    assert len(neighbours) > 0
    for a, b in neighbours:
        assert np.sum(labels[a] != labels[b]) == 1


def test_bad_bit_length_raises() -> None:
    with pytest.raises(ValueError):
        modulate(np.array([0, 1, 1]), "16QAM")
