"""Additive white Gaussian noise (AWGN) channel."""

from __future__ import annotations

import numpy as np


def ebn0_to_n0(ebn0_db: float, bits_per_symbol: int, es: float = 1.0) -> float:
    """Noise spectral density N0 for a given Eb/N0 (dB).

    Es/N0 = k * Eb/N0, so N0 = Es / (k * Eb/N0_linear).
    """
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return es / (bits_per_symbol * ebn0)


def awgn(
    symbols: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int,
    rng: np.random.Generator,
    es: float = 1.0,
) -> np.ndarray:
    """Add complex AWGN with variance N0/2 per real dimension."""
    n0 = ebn0_to_n0(ebn0_db, bits_per_symbol, es)
    sigma = np.sqrt(n0 / 2.0)
    noise = rng.standard_normal(symbols.shape) + 1j * rng.standard_normal(symbols.shape)
    return symbols + sigma * noise
