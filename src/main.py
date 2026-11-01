"""Run the BER vs Eb/N0 simulation, save a CSV and plot the curves.

Usage: python -m src.main [--max-bits N] [--min-errors N] [--seed N]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.simulate import BerPoint, simulate_scheme  # noqa: E402
from src.theory import theoretical_ber  # noqa: E402

SCHEMES = ["BPSK", "QPSK", "16QAM"]
STYLE = {"BPSK": ("C0", "o", "-"), "QPSK": ("C1", "s", "--"), "16QAM": ("C2", "^", "-")}
ROOT = Path(__file__).resolve().parents[1]


def save_csv(points: list[BerPoint], path: Path) -> None:
    """Write simulated and theoretical BER for every point to CSV."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheme", "ebn0_db", "bits", "errors", "ber_sim", "ber_theory"])
        for p in points:
            theory = float(theoretical_ber(p.scheme, p.ebn0_db))
            w.writerow([p.scheme, p.ebn0_db, p.bits, p.errors, f"{p.ber:.6e}", f"{theory:.6e}"])


def plot(results: dict[str, list[BerPoint]], ebn0_db: np.ndarray, path: Path) -> None:
    """Semilog-y plot: theory as lines, simulation as markers (zero-error points omitted)."""
    fine = np.linspace(ebn0_db.min(), ebn0_db.max(), 300)
    fig, ax = plt.subplots(figsize=(8, 6))
    for scheme, points in results.items():
        color, marker, line = STYLE[scheme]
        ax.semilogy(fine, theoretical_ber(scheme, fine), line, color=color, label=f"{scheme} theory")
        sim = [(p.ebn0_db, p.ber) for p in points if p.errors > 0]
        if sim:
            x, y = zip(*sim)
            ax.semilogy(x, y, marker, color=color, mfc="none", ms=7, label=f"{scheme} simulated")
    ax.set_xlabel(r"$E_b/N_0$ (dB)")
    ax.set_ylabel("Bit error rate (BER)")
    ax.set_title("BER vs $E_b/N_0$ over AWGN (Gray-coded BPSK, QPSK, 16-QAM)")
    ax.set_xlim(ebn0_db.min(), ebn0_db.max())
    ax.set_ylim(1e-8, 1)
    ax.grid(True, which="both", ls=":", alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-errors", type=int, default=100)
    parser.add_argument("--min-bits", type=int, default=100_000)
    parser.add_argument("--max-bits", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    ebn0_db = np.arange(0, 15, 1, dtype=float)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[BerPoint]] = {}
    for scheme in SCHEMES:
        t0 = time.perf_counter()
        results[scheme] = simulate_scheme(
            scheme,
            ebn0_db,
            seed=args.seed,
            min_errors=args.min_errors,
            min_bits=args.min_bits,
            max_bits=args.max_bits,
        )
        print(f"{scheme:>6}: done in {time.perf_counter() - t0:.1f} s")
        for p in results[scheme]:
            print(f"  Eb/N0={p.ebn0_db:4.1f} dB  bits={p.bits:>9d}  errors={p.errors:>6d}  BER={p.ber:.3e}")

    save_csv([p for pts in results.values() for p in pts], args.out_dir / "ber_results.csv")
    plot(results, ebn0_db, args.out_dir / "ber_curves.png")
    print(f"Saved results to {args.out_dir}")


if __name__ == "__main__":
    main()
