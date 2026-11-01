"""Tests for the channel, theory and Monte Carlo BER against theory."""

import numpy as np
import pytest

from src.channel import awgn, ebn0_to_n0
from src.simulate import simulate_point
from src.theory import ber_16qam, ber_bpsk, qfunc


def test_qfunc_known_values() -> None:
    assert qfunc(0.0) == pytest.approx(0.5)
    assert qfunc(3.0) == pytest.approx(1.3499e-3, rel=1e-3)


def test_bpsk_theory_reference_value() -> None:
    # Well-known: BPSK BER at 9.6 dB is about 1e-5
    assert ber_bpsk(9.6) == pytest.approx(1e-5, rel=0.1)


def test_noise_variance_matches_eb_n0() -> None:
    rng = np.random.default_rng(1)
    k, ebn0_db = 4, 6.0
    noise = awgn(np.zeros(1_000_000, dtype=complex), ebn0_db, k, rng)
    assert np.var(noise) == pytest.approx(ebn0_to_n0(ebn0_db, k), rel=0.01)


@pytest.mark.parametrize(
    ("scheme", "ebn0_db", "theory"),
    [
        ("BPSK", 2.0, ber_bpsk),
        ("BPSK", 6.0, ber_bpsk),
        ("QPSK", 4.0, ber_bpsk),
        ("QPSK", 7.0, ber_bpsk),
        ("16QAM", 6.0, ber_16qam),
        ("16QAM", 10.0, ber_16qam),
    ],
)
def test_simulated_ber_close_to_theory(scheme: str, ebn0_db: float, theory) -> None:
    rng = np.random.default_rng(2024)
    point = simulate_point(scheme, ebn0_db, rng, min_errors=1000, max_bits=20_000_000)
    assert point.errors >= 1000
    assert point.ber == pytest.approx(float(theory(ebn0_db)), rel=0.15)
