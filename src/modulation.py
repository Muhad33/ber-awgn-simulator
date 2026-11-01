"""Gray-coded BPSK, QPSK and 16-QAM mappers and hard-decision demappers.

All constellations are normalized to unit average symbol energy (Es = 1).
"""

from __future__ import annotations

import numpy as np

BITS_PER_SYMBOL: dict[str, int] = {"BPSK": 1, "QPSK": 2, "16QAM": 4}

# Gray-coded 4-PAM levels used on each axis of 16-QAM:
# bits (b0, b1): 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
_QAM16_SCALE = np.sqrt(10.0)  # average energy of {±1, ±3} x {±1, ±3} is 10


def _check_bits(bits: np.ndarray, k: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    if bits.ndim != 1:
        raise ValueError("bits must be a 1-D array")
    if bits.size % k != 0:
        raise ValueError(f"number of bits must be a multiple of {k}")
    return bits


def _pam4_gray_map(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Map bit pairs to Gray-coded 4-PAM levels {-3, -1, +1, +3}."""
    gray = 2 * b0 + b1
    index = gray ^ (gray >> 1)  # Gray -> natural binary position 0..3
    return (2 * index - 3).astype(np.float64)


def _pam4_gray_demap(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard-decide 4-PAM samples (unnormalized scale) back to bit pairs."""
    index = np.digitize(x, [-2.0, 0.0, 2.0])  # nearest level position 0..3
    gray = index ^ (index >> 1)  # natural binary -> Gray
    return (gray >> 1).astype(np.int8), (gray & 1).astype(np.int8)


def modulate(bits: np.ndarray, scheme: str) -> np.ndarray:
    """Map a 1-D bit array to complex unit-energy symbols for the given scheme."""
    k = BITS_PER_SYMBOL[scheme]
    bits = _check_bits(bits, k)

    if scheme == "BPSK":
        return (2.0 * bits - 1.0).astype(np.complex128)

    if scheme == "QPSK":
        b = bits.reshape(-1, 2)
        return ((2.0 * b[:, 0] - 1.0) + 1j * (2.0 * b[:, 1] - 1.0)) / np.sqrt(2.0)

    b = bits.reshape(-1, 4)  # 16QAM: (b0, b1) -> I, (b2, b3) -> Q
    i = _pam4_gray_map(b[:, 0], b[:, 1])
    q = _pam4_gray_map(b[:, 2], b[:, 3])
    return (i + 1j * q) / _QAM16_SCALE


def demodulate(symbols: np.ndarray, scheme: str) -> np.ndarray:
    """Hard-decision (minimum-distance) demapping of received symbols to bits."""
    r = np.asarray(symbols)

    if scheme == "BPSK":
        return (r.real > 0).astype(np.int8)

    if scheme == "QPSK":
        out = np.empty((r.size, 2), dtype=np.int8)
        out[:, 0] = r.real > 0
        out[:, 1] = r.imag > 0
        return out.ravel()

    if scheme == "16QAM":
        scaled = r * _QAM16_SCALE
        out = np.empty((r.size, 4), dtype=np.int8)
        out[:, 0], out[:, 1] = _pam4_gray_demap(scaled.real)
        out[:, 2], out[:, 3] = _pam4_gray_demap(scaled.imag)
        return out.ravel()

    raise KeyError(f"unknown scheme: {scheme}")


def constellation(scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Return all constellation points and their bit labels (shape M x k)."""
    k = BITS_PER_SYMBOL[scheme]
    m = 2**k
    labels = ((np.arange(m)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.int8)
    return modulate(labels.ravel(), scheme), labels
