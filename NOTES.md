# Theory Notes

Short, plain-language notes behind the simulator.

## 1. Eb/N0 vs Es/N0

- **Es**: energy per *symbol*. **Eb**: energy per *bit*. **N0**: noise power spectral density.
- A symbol carries `k = log2(M)` bits, so `Es = k · Eb`, and

  ```
  Es/N0 = k · Eb/N0        (linear)
  Es/N0 [dB] = Eb/N0 [dB] + 10·log10(k)
  ```

- Comparing schemes at the same **Eb/N0** is fair: every scheme spends the same energy per bit.
- In the simulator, symbols have `Es = 1`, so `N0 = 1 / (k · Eb/N0)`. Complex noise gets
  variance `N0/2` on the real part and `N0/2` on the imaginary part (`src/channel.py`).

| Scheme | k | Es/N0 − Eb/N0 |
|--------|---|---------------|
| BPSK   | 1 | 0 dB          |
| QPSK   | 2 | 3.01 dB       |
| 16-QAM | 4 | 6.02 dB       |

## 2. Constellations (all normalized to Es = 1)

- **BPSK**: two points, `±1` on the real axis. Bit 0 → −1, bit 1 → +1.
- **QPSK**: four points `(±1 ± j)/√2`. First bit picks the sign of I, second bit the sign of Q.
  It is really two independent BPSK signals on I and Q.
- **16-QAM**: a 4×4 grid with levels `{−3, −1, +1, +3}` on each axis, divided by `√10`
  (the average energy of the raw grid is 10). Two bits pick the I level and two bits the Q level,
  each with the Gray map `00 → −3, 01 → −1, 11 → +1, 10 → +3`.

Hard-decision demodulation picks the nearest point. For these square constellations that is just
independent thresholds on I and Q (0 for BPSK/QPSK; −2, 0, +2 for 16-QAM before scaling).

## 3. The Q-function

`Q(x)` is the probability that a standard normal variable exceeds `x`:

```
Q(x) = 0.5 · erfc(x / √2)
```

If a decision boundary is distance `d` from the transmitted point and the noise standard deviation
per dimension is `σ = √(N0/2)`, the chance of crossing it is `Q(d/σ)`. This gives:

| Scheme | Theoretical BER (Gray coding) | |
|--------|-------------------------------|---|
| BPSK   | `Q(√(2·Eb/N0))`               | exact |
| QPSK   | `Q(√(2·Eb/N0))`               | exact (same as BPSK per bit) |
| 16-QAM | `(3/4) · Q(√((4/5)·Eb/N0))`   | standard approximation, tight above ~6 dB |

The 16-QAM formula comes from the general square M-QAM approximation
`(4/k)(1 − 1/√M) · Q(√(3k/(M−1) · Eb/N0))` with `M = 16`, `k = 4`.

## 4. Why Gray coding matters

At reasonable SNR, almost every symbol error lands on a *neighbouring* point. Gray coding labels
neighbours so they differ in exactly **one bit**, so one symbol error ≈ one bit error and
`BER ≈ SER / k`. With plain binary labels, some neighbours differ in two or more bits
(e.g. `01 → 10`), which increases BER for no gain. The test suite checks the one-bit property for
every nearest-neighbour pair.

## 5. Worked examples

**Example — noise level.** 16-QAM at Eb/N0 = 10 dB: `Eb/N0 = 10`, `Es/N0 = 40`,
so `N0 = 1/40 = 0.025` and `σ = √(0.0125) ≈ 0.112` per dimension.

**Example — BPSK at 9.6 dB.** `Eb/N0 = 10^0.96 ≈ 9.12`, `√(2·9.12) ≈ 4.27`, `Q(4.27) ≈ 1.0e−5`.
This is the classic "BPSK needs 9.6 dB for BER 1e−5" figure.

**Theoretical BER table** (from `src/theory.py`):

| Eb/N0 (dB) | BPSK | QPSK | 16-QAM |
|-----------:|-----:|-----:|-------:|
| 0 | 7.86e-02 | 7.86e-02 | 1.39e-01 |
| 2 | 3.75e-02 | 3.75e-02 | 9.76e-02 |
| 4 | 1.25e-02 | 1.25e-02 | 5.86e-02 |
| 6 | 2.39e-03 | 2.39e-03 | 2.79e-02 |
| 8 | 1.91e-04 | 1.91e-04 | 9.25e-03 |
| 10 | 3.87e-06 | 3.87e-06 | 1.75e-03 |
| 12 | 9.01e-09 | 9.01e-09 | 1.39e-04 |
| 14 | 6.81e-13 | 6.81e-13 | 2.76e-06 |

Takeaways: QPSK matches BPSK per bit while sending twice as many bits per symbol.
16-QAM doubles that again but needs roughly 4 dB more Eb/N0 for the same BER.
