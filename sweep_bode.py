# sweep_bode.py
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lab_instruments.devices.owon_dge2070 import OwonDGE2070
from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15


# =========================
# USER SETTINGS
# =========================
START_HZ = 1.0
STOP_HZ = 100_000.0
POINTS_PER_DECADE = 10

AWG_CHANNEL = 1
AWG_VPP = 0.25
AWG_LOAD_OHMS = 50  # if your driver supports it; otherwise ignored

# DSO channels: CH_REF is "input to DUT" (from AWG), CH_DUT is "output of DUT"
CH_DUT = 1
CH_REF = 2


COUPLING = "AC"         # "AC" or "DC"
V_PER_DIV_REF = 0.2     # tweak these so both traces are nicely sized
V_PER_DIV_DUT = 2.0

SETTLE_S = 0.25         # time to wait after changing frequency
MEAS_AVG = 2            # acquisitions per point (simple averaging of gain/phase)

OUT_CSV = Path(__file__).with_name("bode_sweep.csv")


# =========================
# TIMEBASE CHOICE
# =========================
def _allowed_timebases():
    # (2, 5, 10) × 10^n from 2 ns to 100 s
    vals = []
    for n in range(-9, 2):
        for m in (2, 5, 10):
            v = m * (10.0 ** n)
            if 2e-9 <= v <= 100.0:
                vals.append(v)
    return sorted(set(vals))


def choose_timebase_for_freq(f_hz: float, divs: int = 14, cycles_min: float = 4.0, cycles_max: float = 10.0) -> float:
    """
    Choose seconds/div so that total time window spans ~4–10 cycles.
    """
    if f_hz <= 0:
        raise ValueError("f_hz must be > 0")
    T = 1.0 / f_hz
    tb_low = (cycles_min / divs) * T
    tb_high = (cycles_max / divs) * T

    cands = _allowed_timebases()
    inside = [tb for tb in cands if tb_low <= tb <= tb_high]
    if inside:
        center = 0.5 * (tb_low + tb_high)
        return min(inside, key=lambda x: abs(x - center))

    # fallback: closest to midpoint cycles
    cycles_mid = 0.5 * (cycles_min + cycles_max)

    def cycles(tb):
        return (tb * divs) / T

    return min(cands, key=lambda tb: abs(cycles(tb) - cycles_mid))


# =========================
# SIGNAL ANALYSIS (gain/phase)
# =========================
@dataclass
class FitResult:
    amp: float
    phase: float  # radians


def fit_sine_harmonics(x: np.ndarray, t: np.ndarray, f0: float, max_harm: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear least squares fit of harmonics:
      x(t) ≈ Σ_k [a_k cos(2πk f0 t) + b_k sin(2πk f0 t)]
    Returns amps[k-1], phases[k-1] for k=1..max_harm
    """
    cols = []
    for k in range(1, max_harm + 1):
        cols.append(np.cos(2 * np.pi * k * f0 * t))
        cols.append(np.sin(2 * np.pi * k * f0 * t))
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)

    amps = []
    phis = []
    for k in range(max_harm):
        a = c[2 * k]
        b = c[2 * k + 1]
        amps.append(float(np.hypot(a, b)))
        phis.append(float(np.arctan2(b, a)))
    return np.array(amps), np.array(phis)


def principal_angle(rad: float) -> float:
    # map to [-pi, pi)
    return (rad + np.pi) % (2 * np.pi) - np.pi


def gain_phase_from_waveforms(v_ref: np.ndarray, v_dut: np.ndarray, t: np.ndarray, f_nom: float) -> tuple[float, float]:
    """
    Returns: gain_db (DUT/REF), phase_deg (DUT - REF)
    Uses fundamental from the nominal frequency f_nom (assumes AWG is accurate enough).
    """
    # remove DC
    x1 = v_ref - np.mean(v_ref)
    x2 = v_dut - np.mean(v_dut)

    a1, p1 = fit_sine_harmonics(x1, t, f_nom, max_harm=10)
    a2, p2 = fit_sine_harmonics(x2, t, f_nom, max_harm=10)

    Aref = a1[0]
    Adut = a2[0]

    gain_db = 20.0 * math.log10(Adut / Aref) if Aref > 0 else float("inf")
    dphi = principal_angle(p2[0] - p1[0])
    phase_deg = float(np.degrees(dphi))
    return gain_db, phase_deg


# =========================
# MAIN SWEEP
# =========================
def logspace_points(f_start: float, f_stop: float, pts_per_dec: int) -> np.ndarray:
    decades = math.log10(f_stop) - math.log10(f_start)
    n = int(decades * pts_per_dec) + 1
    return np.logspace(math.log10(f_start), math.log10(f_stop), n)


def configure_dso_frontend(dso: HantekDSO2D15, f_hz: float) -> None:
    tb = choose_timebase_for_freq(f_hz)
    # Note: your Hantek seems to like :CHAN1 / :CHAN2 forms for setting.
    dso.scpi.write(f":TIMEBASE:SCALE {tb:.12g}")

    dso.scpi.write(f":CHAN{CH_REF}:COUPling {COUPLING}")
    dso.scpi.write(f":CHAN{CH_DUT}:COUPling {COUPLING}")

    dso.scpi.write(f":CHAN{CH_REF}:SCALe {V_PER_DIV_REF}")
    dso.scpi.write(f":CHAN{CH_DUT}:SCALe {V_PER_DIV_DUT}")

    # small settle after frontend changes
    time.sleep(0.15)


def read_two_channels(dso: HantekDSO2D15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = dso.read_waveform_all()
    meta = wave["metadata"]

    ch_data = {}
    for ch in wave["channels"]:
        if ch["enabled"]:
            ch_data[ch["channel"]] = (ch["time"], ch["voltage"])

    if CH_REF not in ch_data or CH_DUT not in ch_data:
        raise RuntimeError(f"Need CH{CH_REF} and CH{CH_DUT} enabled on the scope.")

    t_ref, v_ref = ch_data[CH_REF]
    t_dut, v_dut = ch_data[CH_DUT]

    # time vectors should match; use ref
    return np.asarray(t_ref), np.asarray(v_ref), np.asarray(v_dut)


def main() -> int:
    freqs = logspace_points(START_HZ, STOP_HZ, POINTS_PER_DECADE)
    OUT_CSV.write_text("")  # truncate

    print("Connecting instruments...")
    awg = OwonDGE2070.connect()
    dso = HantekDSO2D15.connect_by_pattern()

    try:
        # CSV header
        with OUT_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["freq_hz", "gain_db", "phase_deg"])

        print(f"Configuring AWG: {AWG_VPP:.3f} Vpp")
        # Use whichever methods your driver has; keep it simple and explicit:
        # If your driver names differ, swap these two lines to match.
        awg.set_sine(channel=AWG_CHANNEL, freq_hz=freqs[0], amplitude_vpp=AWG_VPP, load_ohms=AWG_LOAD_OHMS)
        awg.output(AWG_CHANNEL, True)

        gains = []
        phases = []

        for f_hz in freqs:
            # Set AWG freq
            awg.set_sine(channel=AWG_CHANNEL, freq_hz=float(f_hz), amplitude_vpp=AWG_VPP, load_ohms=AWG_LOAD_OHMS)

            # Set DSO frontend for this frequency
            configure_dso_frontend(dso, float(f_hz))

            time.sleep(SETTLE_S)

            # Simple averaging across MEAS_AVG acquisitions
            g_list = []
            p_list = []
            for _ in range(MEAS_AVG):
                t, v_ref, v_dut = read_two_channels(dso)
                g_db, p_deg = gain_phase_from_waveforms(v_ref, v_dut, t, float(f_hz))
                g_list.append(g_db)
                p_list.append(p_deg)

            gain_db = float(np.mean(g_list))
            # phase averaging: do it on the unit circle
            p_rad = np.deg2rad(p_list)
            phase_deg = float(np.degrees(math.atan2(float(np.mean(np.sin(p_rad))), float(np.mean(np.cos(p_rad))))))

            gains.append(gain_db)
            phases.append(phase_deg)

            print(f"{f_hz:9.2f} Hz  gain={gain_db:+7.3f} dB   phase={phase_deg:+7.2f} deg")

            with OUT_CSV.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([float(f_hz), gain_db, phase_deg])

        # ---- Plots ----
        freqs = np.asarray(freqs)
        gains = np.asarray(gains)
        phases = np.asarray(phases)

        fig1 = plt.figure(figsize=(10, 5))
        plt.semilogx(freqs, gains, marker="o")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain (dB)")
        plt.title("Bode magnitude")

        fig2 = plt.figure(figsize=(10, 5))
        plt.semilogx(freqs, phases, marker="o")
        plt.grid(True, which="both", ls="--")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (deg)")
        plt.title("Bode phase (DUT - REF)")

        plt.show()
        print(f"\nSaved: {OUT_CSV}")
        return 0

    finally:
        # Safety: turn off AWG output and close
        try:
            awg.output(AWG_CHANNEL, False)
        except Exception:
            pass
        try:
            awg.scpi.close()
        except Exception:
            pass
        try:
            dso.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
