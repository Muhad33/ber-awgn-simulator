"""Monte Carlo BER simulation over an AWGN channel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.channel import awgn
from src.modulation import BITS_PER_SYMBOL, demodulate, modulate


@dataclass
class BerPoint:
    """Result of simulating one (scheme, Eb/N0) point."""

    scheme: str
    ebn0_db: float
    bits: int
    errors: int

    @property
    def ber(self) -> float:
        return self.errors / self.bits if self.bits else float("nan")


def simulate_point(
    scheme: str,
    ebn0_db: float,
    rng: np.random.Generator,
    min_errors: int = 100,
    min_bits: int = 100_000,
    max_bits: int = 10_000_000,
    batch_bits: int = 1_000_000,
) -> BerPoint:
    """Simulate in batches until min_errors (and min_bits) are reached or max_bits is hit."""
    k = BITS_PER_SYMBOL[scheme]
    batch_bits -= batch_bits % k  # whole symbols only
    bits_done = errors = 0

    while bits_done < max_bits:
        n = min(batch_bits, max_bits - bits_done)
        n -= n % k
        if n == 0:
            break
        tx_bits = rng.integers(0, 2, size=n, dtype=np.int8)
        rx = awgn(modulate(tx_bits, scheme), ebn0_db, k, rng)
        errors += int(np.count_nonzero(demodulate(rx, scheme) != tx_bits))
        bits_done += n
        if errors >= min_errors and bits_done >= min_bits:
            break

    return BerPoint(scheme, float(ebn0_db), bits_done, errors)


def simulate_scheme(
    scheme: str,
    ebn0_db: np.ndarray,
    seed: int = 12345,
    **kwargs: int,
) -> list[BerPoint]:
    """Simulate BER over a range of Eb/N0 values with a reproducible seed."""
    rng = np.random.default_rng([seed, sum(map(ord, scheme))])
    return [simulate_point(scheme, float(e), rng, **kwargs) for e in ebn0_db]
