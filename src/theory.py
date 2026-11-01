"""Theoretical BER expressions for Gray-coded BPSK, QPSK and 16-QAM over AWGN."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def qfunc(x: np.ndarray | float) -> np.ndarray:
    """Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))


def ber_bpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """Exact BPSK BER: Q(sqrt(2 Eb/N0))."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return qfunc(np.sqrt(2.0 * ebn0))


def ber_qpsk(ebn0_db: np.ndarray | float) -> np.ndarray:
    """Exact Gray-coded QPSK BER (same as BPSK per bit)."""
    return ber_bpsk(ebn0_db)


def ber_16qam(ebn0_db: np.ndarray | float) -> np.ndarray:
    """Standard Gray-coded 16-QAM approximation: (3/4) Q(sqrt(4/5 Eb/N0))."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db, dtype=np.float64) / 10.0)
    return 0.75 * qfunc(np.sqrt(0.8 * ebn0))


THEORY = {"BPSK": ber_bpsk, "QPSK": ber_qpsk, "16QAM": ber_16qam}


def theoretical_ber(scheme: str, ebn0_db: np.ndarray | float) -> np.ndarray:
    """Dispatch to the theoretical BER function for a scheme."""
    return THEORY[scheme](ebn0_db)
