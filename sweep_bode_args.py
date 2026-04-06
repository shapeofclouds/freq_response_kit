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

def _interp_crossing_freq(
    f1: float, g1: float, f2: float, g2: float, g_target: float, *, log_f: bool = True
) -> float:
    """
    Interpolate frequency where gain crosses g_target between (f1,g1) and (f2,g2).
    Assumes g_target lies between g1 and g2.
    If log_f=True, interpolate in log10(f).
    """
    if g2 == g1:
        return float(f1)
    t = (g_target - g1) / (g2 - g1)
    t = max(0.0, min(1.0, float(t)))

    if log_f:
        lf1 = math.log10(f1)
        lf2 = math.log10(f2)
        lf = lf1 + t * (lf2 - lf1)
        return 10.0 ** lf
    else:
        return f1 + t * (f2 - f1)


def find_3db_bandwidth(
    freqs_hz: np.ndarray, gains_db: np.ndarray, *, log_f_interp: bool = True
) -> tuple[str, str, float, float]:
    """
    Returns:
      (f_lo_str, f_hi_str, gmax_db, g3_db)

    f_lo_str / f_hi_str are either numeric like "123.4 Hz" or boundary indicators
    like "<1 Hz" / ">100 kHz" when no -3 dB crossing is found in the scanned range.
    """
    freqs = np.asarray(freqs_hz, dtype=float)
    gains = np.asarray(gains_db, dtype=float)

    if freqs.size < 3:
        return ("n/a", "n/a", float("nan"), float("nan"))

    i_max = int(np.nanargmax(gains))
    gmax = float(gains[i_max])
    g3 = gmax - 3.0

    f_start = float(freqs[0])
    f_stop = float(freqs[-1])

    # ---- search left side ----
    f_lo = None
    for i in range(i_max, 0, -1):
        gA, gB = float(gains[i]), float(gains[i - 1])
        # crossing if target lies between gA and gB (inclusive)
        if (gA - g3) * (gB - g3) <= 0.0:
            f_lo = _interp_crossing_freq(
                float(freqs[i]), gA,
                float(freqs[i - 1]), gB,
                g3,
                log_f=log_f_interp
            )
            break

    # If never crossed on the left: it never got 3 dB down within range
    if f_lo is None:
        f_lo_str = f"<{f_start:g} Hz"
    else:
        f_lo_str = f"{f_lo:g} Hz"

    # ---- search right side ----
    f_hi = None
    for i in range(i_max, len(freqs) - 1):
        gA, gB = float(gains[i]), float(gains[i + 1])
        if (gA - g3) * (gB - g3) <= 0.0:
            f_hi = _interp_crossing_freq(
                float(freqs[i]), gA,
                float(freqs[i + 1]), gB,
                g3,
                log_f=log_f_interp
            )
            break

    if f_hi is None:
        f_hi_str = f">{f_stop:g} Hz"
    else:
        f_hi_str = f"{f_hi:g} Hz"

    return f_lo_str, f_hi_str, gmax, g3



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

def thd_ratio_from_amps(amps: np.ndarray, max_harm: int | None = None) -> float:
    """
    amps: array of harmonic amplitudes where amps[0] is fundamental (k=1)
    Returns THD as a ratio (e.g. 0.01 = 1%)
    """
    if max_harm is None:
        max_harm = len(amps)
    max_harm = min(max_harm, len(amps))
    A1 = float(amps[0])
    if A1 <= 0:
        return float("nan")
    harm = amps[1:max_harm]
    return float(np.sqrt(np.sum(harm * harm)) / A1)


def corrected_thd_ratio(
    amps_ref: np.ndarray,
    amps_dut: np.ndarray,
    gain_linear: float,
    max_harm: int | None = None,
) -> float:
    """
    Estimate DUT-added THD by subtracting (in power) the transferred source harmonics.
    Assumes harmonic transfer gain ~= fundamental gain (gain_linear).
    Returns corrected THD ratio referenced to DUT fundamental.
    """
    if max_harm is None:
        max_harm = min(len(amps_ref), len(amps_dut))
    max_harm = min(max_harm, len(amps_ref), len(amps_dut))

    A1_out = float(amps_dut[0])
    if A1_out <= 0:
        return float("nan")

    # k=2..max_harm
    added_sq = 0.0
    for k in range(1, max_harm):  # index 1 => 2nd harmonic
        pred = gain_linear * float(amps_ref[k])
        obs = float(amps_dut[k])
        a2 = obs * obs - pred * pred
        if a2 > 0:
            added_sq += a2

    return float(np.sqrt(added_sq) / A1_out)



def principal_angle(rad: float) -> float:
    # map to [-pi, pi)
    return (rad + np.pi) % (2 * np.pi) - np.pi

def gain_phase_from_waveforms(
    v_ref: np.ndarray, v_dut: np.ndarray, t: np.ndarray, f_nom: float
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns: gain_db (DUT/REF), phase_deg (DUT - REF), amps_ref, amps_dut
    amps_* are harmonic amplitude arrays for k=1..10 (fundamental is [0]).
    """
    x1 = v_ref - np.mean(v_ref)
    x2 = v_dut - np.mean(v_dut)

    amps_ref, ph_ref = fit_sine_harmonics(x1, t, f_nom, max_harm=10)
    amps_dut, ph_dut = fit_sine_harmonics(x2, t, f_nom, max_harm=10)

    Aref = float(amps_ref[0])
    Adut = float(amps_dut[0])

    gain_db = 20.0 * math.log10(Adut / Aref) if Aref > 0 else float("inf")
    dphi = principal_angle(float(ph_dut[0]) - float(ph_ref[0]))
    phase_deg = float(np.degrees(dphi))
    return gain_db, phase_deg, amps_ref, amps_dut

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


def _allowed_vdivs() -> list[float]:
    """
    Common 1-2-5 vertical scales in V/div.
    Adjust if your Hantek model supports a different set.
    """
    vals = []
    for n in range(-3, 3):   # 1 mV/div up to 500 V/div
        base = 10.0 ** n
        for m in (1, 2, 5):
            vals.append(m * base)
    return sorted(set(v for v in vals if 1e-3 <= v <= 500.0))


def choose_vdiv_for_vpp(
    vpp: float,
    *,
    target_div_pp: float = 4.0,
    min_div_pp: float = 2.0,
    max_div_pp: float = 6.0,
    allowed_vdivs: list[float] | None = None,
) -> float:
    """
    Choose V/div so the waveform is preferably about target_div_pp divisions p-p,
    while accepting anything between min_div_pp and max_div_pp divisions p-p.

    Since p-p span in divisions is:
        div_pp = vpp / vdiv
    the ideal choice is:
        vdiv ~= vpp / target_div_pp
    """
    if vpp <= 0 or not np.isfinite(vpp):
        raise ValueError("vpp must be finite and > 0")

    if allowed_vdivs is None:
        allowed_vdivs = _allowed_vdivs()

    ideal_vdiv = vpp / target_div_pp

    acceptable = []
    for vdiv in allowed_vdivs:
        div_pp = vpp / vdiv
        if min_div_pp <= div_pp <= max_div_pp:
            acceptable.append(vdiv)

    if acceptable:
        return min(acceptable, key=lambda x: abs(x - ideal_vdiv))

    # If nothing lands in the 2–6 div window, choose closest to target.
    return min(allowed_vdivs, key=lambda x: abs(x - ideal_vdiv))


def estimate_vpp(x: np.ndarray) -> float:
    """
    Robust-ish Vpp estimate from captured waveform.
    Percentiles help a little against spikes/noise.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    return float(hi - lo)


def autorange_dut_vertical(
    dso: HantekDSO2D15,
    *,
    ch_ref: int,
    ch_dut: int,
    initial_vdiv_dut: float,
    settle_s: float,
) -> float:
    """
    Perform one quick read using the current DUT V/div, estimate DUT Vpp,
    then choose a better DUT V/div to place the trace roughly 2–6 div p-p
    (prefer ~4 div p-p). Returns the chosen V/div.

    If the first estimate is unusable, returns the original setting.
    """
    time.sleep(max(0.05, 0.5 * settle_s))

    try:
        _, _, v_dut = read_two_channels(dso, ch_ref=ch_ref, ch_dut=ch_dut)
    except Exception:
        return initial_vdiv_dut

    vpp_dut = estimate_vpp(v_dut)
    if not np.isfinite(vpp_dut) or vpp_dut <= 0:
        return initial_vdiv_dut

    try:
        new_vdiv = choose_vdiv_for_vpp(vpp_dut)
    except Exception:
        return initial_vdiv_dut

    # Avoid pointless writes
    if abs(new_vdiv - initial_vdiv_dut) / max(initial_vdiv_dut, 1e-12) < 0.05:
        return initial_vdiv_dut

    dso.scpi.write(f":CHAN{ch_dut}:SCALe {new_vdiv}")
    time.sleep(max(0.05, 0.5 * settle_s))
    return new_vdiv


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
            w.writerow(["Frequency_Hz", "Gain_dB", "Phase_deg", "THD_DUT_pct", "THD_Corr_pct", "THD_Ref_pct"])

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
        thd_duts = []
        thd_refs = []
        thd_corrs = []

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
                v_per_div_dut=args.vdiv_dut,   # initial guess only
            )

            time.sleep(args.settle_s)

            actual_vdiv_dut = autorange_dut_vertical(
                dso,
                ch_ref=args.ch_ref,
                ch_dut=args.ch_dut,
                initial_vdiv_dut=args.vdiv_dut,
                settle_s=args.settle_s,
            )

            g_list = []
            p_list = []
            
            thd_dut_list = []
            thd_ref_list = []
            thd_corr_list = []

            for _ in range(args.meas_avg):
                t, v_ref, v_dut = read_two_channels(dso, ch_ref=args.ch_ref, ch_dut=args.ch_dut)

                g_db, p_deg, a_ref, a_dut = gain_phase_from_waveforms(v_ref, v_dut, t, float(f_hz))

                thd_ref = 100.0 * thd_ratio_from_amps(a_ref, max_harm=10)
                thd_dut = 100.0 * thd_ratio_from_amps(a_dut, max_harm=10)

                gain_lin = 10.0 ** (g_db / 20.0)
                thd_corr = 100.0 * corrected_thd_ratio(a_ref, a_dut, gain_linear=gain_lin, max_harm=10)


                g_list.append(g_db)
                p_list.append(p_deg)

                thd_ref_list.append(thd_ref)
                thd_dut_list.append(thd_dut)
                thd_corr_list.append(thd_corr)

            gain_db = float(np.mean(g_list))
            p_rad = np.deg2rad(p_list)
            phase_deg = float(np.degrees(math.atan2(float(np.mean(np.sin(p_rad))), float(np.mean(np.cos(p_rad))))))

            thd_ref_pct = float(np.mean(thd_ref_list))
            thd_dut_pct = float(np.mean(thd_dut_list))
            thd_corr_pct = float(np.mean(thd_corr_list))

            if not np.isfinite(thd_corr_pct) or thd_corr_pct < 0:
                thd_corr_pct = thd_dut_pct

            gains.append(gain_db)
            phases.append(phase_deg)
            thd_duts.append(thd_dut_pct)
            thd_refs.append(thd_ref_pct)
            thd_corrs.append(thd_corr_pct)

            print(
                f"{f_hz:9.2f} Hz  "
                f"gain={gain_db:+7.3f} dB   phase={phase_deg:+7.2f} deg   "
                f"THD(dut)={thd_dut_pct:6.3f}%  THD(corr)={thd_corr_pct:6.3f}%  THD(ref)={thd_ref_pct:6.3f}%  "
                f"DUT_scale={actual_vdiv_dut:g} V/div"
            )

            with out_csv.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([float(f_hz), gain_db, phase_deg, thd_dut_pct, thd_corr_pct, thd_ref_pct])

        # ---- Plots (single figure, three panels) ----
        freqs = np.asarray(freqs)
        gains = np.asarray(gains)
        phases = np.asarray(phases)

        thd_duts = np.asarray(thd_duts)
        thd_corrs = np.asarray(thd_corrs)
        thd_refs = np.asarray(thd_refs)

        # Marker control
        marker = "o" if args.markers else None

        # -------- Phase: unwrap to avoid +/-180 jumps, then shift into sensible band --------
        phase_rad = np.deg2rad(phases)
        phase_unwrapped_deg = np.degrees(np.unwrap(phase_rad))

        # Choose an offset so the median lies near (-180, 180]
        offset = 360.0 * round(np.median(phase_unwrapped_deg) / 360.0)
        phase_plot_deg = phase_unwrapped_deg - offset

        # Formatter: display values wrapped to (-180, 180] even if we plot unwrapped
        def wrap180(y, _pos=None):
            y = (y + 180.0) % 360.0 - 180.0
            # Make +180 show as 180 not -180
            if abs(y + 180.0) < 1e-9:
                y = 180.0
            return f"{int(y):d}" if abs(y - round(y)) < 1e-9 else f"{y:g}"

        # -------- Figure layout --------
        fig, (ax_mag, ax_ph, ax_thd) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # -------- Magnitude --------
        ax_mag.semilogx(freqs, gains, marker=marker)
        ax_mag.grid(True, which="both", ls="--")
        ax_mag.set_ylabel("Gain (dB)")
        ax_mag.set_title(
            f"Bode + THD  —  AWG = {args.awg_vpp:g} Vpp"
        )

        f_lo_str, f_hi_str, gmax_db, g3_db = find_3db_bandwidth(freqs, gains)

        # Optional: also compute numeric f_lo/f_hi if present, for drawing vertical lines.
        # We'll parse only if it doesn't start with < or >
        def _parse_freq_str(s: str) -> float | None:
            s = s.strip()
            if s.startswith("<") or s.startswith(">") or s == "n/a":
                return None
            # s like "123.4 Hz"
            return float(s.split()[0])

        f_lo_num = _parse_freq_str(f_lo_str)
        f_hi_num = _parse_freq_str(f_hi_str)
        
        # Draw -3 dB reference line (optional but nice)
        ax_mag.axhline(g3_db, linestyle=":", linewidth=1)

        # Vertical lines at -3 dB points if found
        if f_lo_num is not None:
            ax_mag.axvline(f_lo_num, linestyle=":", linewidth=1)
        if f_hi_num is not None:
            ax_mag.axvline(f_hi_num, linestyle=":", linewidth=1)

        # Put a small info box on the magnitude plot
        info = (
            f"Max gain: {gmax_db:+.2f} dB\n"
            f"-3 dB level: {g3_db:+.2f} dB\n"
            f"fL (-3 dB): {f_lo_str}\n"
            f"fH (-3 dB): {f_hi_str}"
        )
        ax_mag.text(
            0.98, 0.98, info,
            transform=ax_mag.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.85),
        )

        # Optional: also compute numeric f_lo/f_hi if present, for drawing vertical lines.
        # We'll parse only if it doesn't start with < or >
        def _parse_freq_str(s: str) -> float | None:
            s = s.strip()
            if s.startswith("<") or s.startswith(">") or s == "n/a":
                return None
            # s like "123.4 Hz"
            return float(s.split()[0])

        f_lo_num = _parse_freq_str(f_lo_str)
        f_hi_num = _parse_freq_str(f_hi_str)


        # -------- Phase --------
        ax_ph.semilogx(freqs, phase_plot_deg, marker=marker)
        ax_ph.grid(True, which="both", ls="--")
        ax_ph.set_ylabel("Phase (deg)")
        ax_ph.yaxis.set_major_formatter(FuncFormatter(wrap180))

        # Auto-scale phase axis tightly around data (robust, padded)
        y = phase_plot_deg
        p_lo, p_hi = np.percentile(y, [5, 95])
        min_span = 20.0
        span = max(min_span, p_hi - p_lo)
        pad = 0.15 * span
        ymin = p_lo - pad
        ymax = p_hi + pad

        def nice_step(span_):
            if span_ <= 20:
                return 2
            elif span_ <= 40:
                return 5
            elif span_ <= 80:
                return 10
            elif span_ <= 160:
                return 20
            else:
                return 30

        tick_step = nice_step(ymax - ymin)
        start = tick_step * math.floor(ymin / tick_step)
        stop  = tick_step * math.ceil(ymax / tick_step)
        ticks = np.arange(start, stop + 0.5 * tick_step, tick_step)

        ax_ph.set_ylim(start, stop)
        ax_ph.set_yticks(ticks)

        # -------- THD --------
        # Decide what to show:
        # - corrected THD (preferred)
        # - raw DUT THD (useful sanity check)
        # - ref THD (useful to see if correction is meaningful)

        # If corrected contains NaNs, fall back to raw for plotting that point
        thd_plot = np.where(np.isfinite(thd_corrs), thd_corrs, thd_duts)

        ax_thd.semilogx(freqs, thd_plot, label="THD corrected (%)", marker=marker)
        ax_thd.semilogx(freqs, thd_duts, label="THD DUT raw (%)", marker=marker, alpha=0.6)
        ax_thd.semilogx(freqs, thd_refs, label="THD REF (%)", marker=marker, alpha=0.4)

        ax_thd.grid(True, which="both", ls="--")
        ax_thd.set_ylabel("THD (%)")
        ax_thd.set_xlabel("Frequency (Hz)")
        ax_thd.legend(loc="best")

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
