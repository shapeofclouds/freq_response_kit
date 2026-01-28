# measure_1khz.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import warnings

warnings.filterwarnings(
    "ignore",
    message="TCPIP:instr resource discovery is limited*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="TCPIP::hislip resource discovery requires*",
    category=UserWarning,
)


@dataclass
class Result:
    f_hz: float
    ch1_rms_v: float
    ch2_rms_v: float
    gain_db: float
    phase_deg: float


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def _phasor_at(x: np.ndarray, t: np.ndarray, f_hz: float) -> complex:
    """
    Least-squares complex phasor at frequency f_hz.
    Returns complex amplitude (not RMS), whose angle is phase.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    # Remove DC
    x = x - np.mean(x)

    # complex reference
    ref = np.exp(-1j * 2.0 * np.pi * f_hz * t)

    # Project (like a single-bin DFT but on arbitrary time grid)
    # Scale so that |phasor| is approximately peak amplitude for a clean sine.
    ph = (2.0 / len(x)) * np.sum(x * ref)
    return ph


def analyse_two_channels(
    t: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    f_nom_hz: float,
) -> Result:
    # RMS
    v1_rms = _rms(v1)
    v2_rms = _rms(v2)

    # Phasors at nominal frequency
    p1 = _phasor_at(v1, t, f_nom_hz)
    p2 = _phasor_at(v2, t, f_nom_hz)

    # Gain/phase (ch1 relative to ch2, adjust if you prefer opposite)
    gain = abs(p1) / abs(p2) if abs(p2) > 0 else float("inf")
    gain_db = 20.0 * math.log10(gain) if gain > 0 else float("-inf")

    phase_rad = np.angle(p1) - np.angle(p2)
    # Wrap to [-pi, pi]
    phase_rad = (phase_rad + np.pi) % (2.0 * np.pi) - np.pi
    phase_deg = float(np.degrees(phase_rad))

    return Result(
        f_hz=f_nom_hz,
        ch1_rms_v=v1_rms,
        ch2_rms_v=v2_rms,
        gain_db=float(gain_db),
        phase_deg=phase_deg,
    )


def main() -> int:
    # Import from the installed lab_instruments wheel (consumer mode)
    from lab_instruments.devices.owon_dge2070 import OwonDGE2070
    from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15

    F_HZ = 1000.0
    AMP_VPP = 0.25
    LOAD_OHMS = 50

    print("Connecting...")
    awg = OwonDGE2070.connect()
    dso = HantekDSO2D15.connect_by_pattern()

    # --- Configure AWG ---
    print(f"Configuring AWG: {F_HZ:.1f} Hz, {AMP_VPP:.3f} Vpp")
    awg.set_sine(channel=1, freq_hz=F_HZ, amplitude_vpp=AMP_VPP, load_ohms=LOAD_OHMS)
    awg.output(1, True)

    # Give everything a moment to settle
    time.sleep(0.5)

    try:
        # --- Configure scope frontend (minimal) ---
        # These method names depend on your driver; if your driver already sets
        # timebase/coupling/scale internally, you can remove this block.
        try:
            dso.set_coupling(1, "DC")
            dso.set_coupling(2, "DC")
            dso.set_scale(1, 0.1)  # V/div (adjust if clipping)
            dso.set_scale(2, 0.1)
            dso.auto_timebase_for_freq(F_HZ)  # if you have a helper like this
        except Exception:
            print("Failed to configure scope frontend; proceeding anyway.")
            # If your current driver doesn’t expose these, that’s OK for a first pass.
            pass

        print("Acquiring waveform...")
        wave = dso.read_waveform_all()  # expected: {"metadata":..., "channels":[...]}
        chs = [c for c in wave["channels"] if c.get("enabled")]
        if len(chs) < 2:
            raise RuntimeError("Need at least two enabled scope channels for gain/phase.")

        # Use first two enabled channels
        t = np.asarray(chs[0]["time"], dtype=float)
        v1 = np.asarray(chs[0]["voltage"], dtype=float)
        v2 = np.asarray(chs[1]["voltage"], dtype=float)

        res = analyse_two_channels(t, v1, v2, F_HZ)

        print("\n=== 1 kHz measurement ===")
        print(f"f_nom:      {res.f_hz:.1f} Hz")
        print(f"CH1 RMS:    {res.ch1_rms_v:.6f} V")
        print(f"CH2 RMS:    {res.ch2_rms_v:.6f} V")
        print(f"Gain:       {res.gain_db:+.3f} dB  (CH1/CH2)")
        print(f"Phase:      {res.phase_deg:+.2f} deg (CH1-CH2)")
        print("=========================\n")

        return 0

    finally:
        # Leave AWG in a safe state
        try:
            awg.output(1, False)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
