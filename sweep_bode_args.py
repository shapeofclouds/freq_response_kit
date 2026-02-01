#sweep_bode_args.py
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lab_instruments.devices.owon_dge2070 import OwonDGE2070
from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15

from matplotlib.ticker import FuncFormatter

import warnings
import logging

# Suppress noisy pyvisa-py TCPIP warnings (adjust if needed)
warnings.filterwarnings("ignore", module=r"pyvisa_py\.tcpip")
warnings.filterwarnings("ignore", category=ResourceWarning, module=r"pyvisa_py\..*")

logging.getLogger("pyvisa").setLevel(logging.ERROR)
logging.getLogger("pyvisa_py").setLevel(logging.ERROR)

# =========================
# USER SETTINGS (DEFAULTS)
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

def find_local_venv_python(script_path: str) -> str:
    try:
        base = Path(script_path).resolve().parent
        cand = base / ".venv" / "Scripts" / "python.exe"
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return ""


# =========================
# CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Log sweep Bode (gain/phase) using Owon DGE2070 + Hantek DSO2D15")

    # Sweep
    p.add_argument("--start-hz", type=float, default=START_HZ, help=f"Start frequency in Hz (default {START_HZ})")
    p.add_argument("--stop-hz", type=float, default=STOP_HZ, help=f"Stop frequency in Hz (default {STOP_HZ})")
    p.add_argument(
        "--points-per-decade",
        type=int,
        default=POINTS_PER_DECADE,
        help=f"Log points per decade (default {POINTS_PER_DECADE})",
    )

    # AWG
    p.add_argument("--awg-channel", type=int, default=AWG_CHANNEL, help=f"AWG channel (default {AWG_CHANNEL})")
    p.add_argument("--awg-vpp", type=float, default=AWG_VPP, help=f"AWG amplitude in Vpp (default {AWG_VPP})")
    p.add_argument(
        "--awg-load-ohms",
        type=float,
        default=AWG_LOAD_OHMS,
        help=f"AWG output load in ohms (default {AWG_LOAD_OHMS})",
    )

    # DSO channels + frontend
    p.add_argument("--ch-dut", type=int, default=CH_DUT, help=f"Scope channel for DUT output (default {CH_DUT})")
    p.add_argument("--ch-ref", type=int, default=CH_REF, help=f"Scope channel for reference input (default {CH_REF})")
    p.add_argument(
        "--coupling",
        type=str,
        default=COUPLING,
        choices=["AC", "DC"],
        help=f"Scope coupling AC/DC (default {COUPLING})",
    )
    p.add_argument(
        "--vdiv-ref",
        type=float,
        default=V_PER_DIV_REF,
        help=f"Volts/div for REF channel (default {V_PER_DIV_REF})",
    )
    p.add_argument(
        "--vdiv-dut",
        type=float,
        default=V_PER_DIV_DUT,
        help=f"Volts/div for DUT channel (default {V_PER_DIV_DUT})",
    )

    # Acquisition
    p.add_argument("--settle-s", type=float, default=SETTLE_S, help=f"Settle time in seconds (default {SETTLE_S})")
    p.add_argument("--meas-avg", type=int, default=MEAS_AVG, help=f"Acquisitions averaged per point (default {MEAS_AVG})")

    # Output
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_CSV,
        help=f"CSV output path (default {OUT_CSV})",
    )

    p.add_argument(
    "--markers",
    action="store_true",
    help="Show data-point markers on plots (default: off)",
)

    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.start_hz <= 0 or args.stop_hz <= 0:
        raise SystemExit("Error: --start-hz and --stop-hz must be > 0")
    if args.stop_hz <= args.start_hz:
        raise SystemExit("Error: --stop-hz must be greater than --start-hz")
    if args.points_per_decade <= 0:
        raise SystemExit("Error: --points-per-decade must be >= 1")
    if args.awg_channel <= 0:
        raise SystemExit("Error: --awg-channel must be >= 1")
    if args.ch_dut <= 0 or args.ch_ref <= 0:
        raise SystemExit("Error: --ch-dut and --ch-ref must be >= 1")
    if args.ch_dut == args.ch_ref:
        raise SystemExit("Error: --ch-dut and --ch-ref must be different channels")
    if args.awg_vpp <= 0:
        raise SystemExit("Error: --awg-vpp must be > 0")
    if args.vdiv_ref <= 0 or args.vdiv_dut <= 0:
        raise SystemExit("Error: --vdiv-ref and --vdiv-dut must be > 0")
    if args.settle_s < 0:
        raise SystemExit("Error: --settle-s must be >= 0")
    if args.meas_avg <= 0:
        raise SystemExit("Error: --meas-avg must be >= 1")


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


def configure_dso_frontend(
    dso: HantekDSO2D15,
    f_hz: float,
    *,
    ch_ref: int,
    ch_dut: int,
    coupling: str,
    v_per_div_ref: float,
    v_per_div_dut: float,
) -> None:
    tb = choose_timebase_for_freq(f_hz)
    dso.scpi.write(f":TIMEBASE:SCALE {tb:.12g}")

    dso.scpi.write(f":CHAN{ch_ref}:COUPling {coupling}")
    dso.scpi.write(f":CHAN{ch_dut}:COUPling {coupling}")

    dso.scpi.write(f":CHAN{ch_ref}:SCALe {v_per_div_ref}")
    dso.scpi.write(f":CHAN{ch_dut}:SCALe {v_per_div_dut}")

    time.sleep(0.15)


def read_two_channels(dso: HantekDSO2D15, *, ch_ref: int, ch_dut: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = dso.read_waveform_all()

    ch_data = {}
    for ch in wave["channels"]:
        if ch["enabled"]:
            ch_data[ch["channel"]] = (ch["time"], ch["voltage"])

    if ch_ref not in ch_data or ch_dut not in ch_data:
        raise RuntimeError(f"Need CH{ch_ref} and CH{ch_dut} enabled on the scope.")

    t_ref, v_ref = ch_data[ch_ref]
    t_dut, v_dut = ch_data[ch_dut]

    return np.asarray(t_ref), np.asarray(v_ref), np.asarray(v_dut)


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    validate_args(args)

    freqs = logspace_points(args.start_hz, args.stop_hz, args.points_per_decade)

    out_csv: Path = args.out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("")  # truncate

    print("Connecting instruments...")
    awg = OwonDGE2070.connect()
    dso = HantekDSO2D15.connect_by_pattern()

    try:
        # CSV header
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["freq_hz", "gain_db", "phase_deg"])

        print(f"Configuring AWG: {args.awg_vpp:.3f} Vpp")
        awg.set_sine(
            channel=args.awg_channel,
            freq_hz=float(freqs[0]),
            amplitude_vpp=args.awg_vpp,
            load_ohms=args.awg_load_ohms,
        )
        awg.output(args.awg_channel, True)

        gains: list[float] = []
        phases: list[float] = []

        for f_hz in freqs:
            awg.set_sine(
                channel=args.awg_channel,
                freq_hz=float(f_hz),
                amplitude_vpp=args.awg_vpp,
                load_ohms=args.awg_load_ohms,
            )

            configure_dso_frontend(
                dso,
                float(f_hz),
                ch_ref=args.ch_ref,
                ch_dut=args.ch_dut,
                coupling=args.coupling,
                v_per_div_ref=args.vdiv_ref,
                v_per_div_dut=args.vdiv_dut,
            )

            time.sleep(args.settle_s)

            g_list = []
            p_list = []
            for _ in range(args.meas_avg):
                t, v_ref, v_dut = read_two_channels(dso, ch_ref=args.ch_ref, ch_dut=args.ch_dut)
                g_db, p_deg = gain_phase_from_waveforms(v_ref, v_dut, t, float(f_hz))
                g_list.append(g_db)
                p_list.append(p_deg)

            gain_db = float(np.mean(g_list))
            p_rad = np.deg2rad(p_list)
            phase_deg = float(np.degrees(math.atan2(float(np.mean(np.sin(p_rad))), float(np.mean(np.cos(p_rad))))))

            gains.append(gain_db)
            phases.append(phase_deg)

            print(f"{f_hz:9.2f} Hz  gain={gain_db:+7.3f} dB   phase={phase_deg:+7.2f} deg")

            with out_csv.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([float(f_hz), gain_db, phase_deg])

        # ---- Plots ----
        

        # ---- Plots (single figure, two panels) ----
        freqs = np.asarray(freqs)
        gains = np.asarray(gains)
        phases = np.asarray(phases)  # wrapped-ish degrees from measurement

        # Marker control
        marker = "o" if args.markers else None

        # Unwrap phase to avoid +/-180 jumps, then shift into a sensible band
        phase_rad = np.deg2rad(phases)
        phase_unwrapped_deg = np.degrees(np.unwrap(phase_rad))

        # Optional: bring values near the usual [-180, +180] neighborhood, but keep continuity
        # Choose an offset so the median lies in (-180, 180]
        offset = 360.0 * round(np.median(phase_unwrapped_deg) / 360.0)
        phase_plot_deg = phase_unwrapped_deg - offset

        # Formatter: display values wrapped to (-180, 180] even if we plot unwrapped
        def wrap180(y, _pos=None):
            y = (y + 180.0) % 360.0 - 180.0
            # Make +180 show as 180 not -180
            if abs(y + 180.0) < 1e-9:
                y = 180.0
            return f"{int(y):d}" if abs(y - round(y)) < 1e-9 else f"{y:g}"

        fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Magnitude
        ax_mag.semilogx(freqs, gains, marker=marker)
        ax_mag.grid(True, which="both", ls="--")
        ax_mag.set_ylabel("Gain (dB)")
        ax_mag.set_title("Bode plot")

        # Phase (continuous line; labels wrapped)
        ax_ph.semilogx(freqs, phase_plot_deg, marker=marker)
        ax_ph.grid(True, which="both", ls="--")
        ax_ph.set_xlabel("Frequency (Hz)")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.yaxis.set_major_formatter(FuncFormatter(wrap180))

        # If you want the exact tick style around 180 when data is near that region:
        # (creates ticks like 160,170,180,-170,-160 via formatter)
       
        # ---- Auto-scale phase axis tightly around data ----
        # Work in the unwrapped plotting coordinates
        y = phase_plot_deg

        # Robust bounds (ignore extreme outliers)
        p_lo, p_hi = np.percentile(y, [5, 95])

        # Minimum visible span (degrees) so flat responses don't look silly
        min_span = 20.0

        span = max(min_span, p_hi - p_lo)

        # Pad a little so the trace doesn't touch the frame
        pad_frac = 0.15
        pad = pad_frac * span

        ymin = p_lo - pad
        ymax = p_hi + pad

        # ---- Choose a "nice" tick step based on span ----
        def nice_step(span):
            if span <= 20:
                return 2
            elif span <= 40:
                return 5
            elif span <= 80:
                return 10
            elif span <= 160:
                return 20
            else:
                return 30

        tick_step = nice_step(ymax - ymin)

        # Snap limits to tick grid
        start = tick_step * math.floor(ymin / tick_step)
        stop  = tick_step * math.ceil(ymax / tick_step)

        ticks = np.arange(start, stop + 0.5 * tick_step, tick_step)

        ax_ph.set_ylim(start, stop)
        ax_ph.set_yticks(ticks)



        plt.tight_layout()
        plt.show()

        
        print(f"\nSaved: {out_csv}")
        return 0

    finally:
        try:
            awg.output(args.awg_channel, False)
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
