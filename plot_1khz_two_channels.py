# plot_1khz_two_channels.py
from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt

# Optional: silence the two pyvisa-py TCPIP discovery warnings you saw earlier
import warnings
warnings.filterwarnings("ignore", message="TCPIP:instr resource discovery is limited*")
warnings.filterwarnings("ignore", message="TCPIP::hislip resource discovery requires*")


FREQ_HZ = 1000.0
VPP = 0.250          # 250 mVpp
LOAD_OHMS = 50       # AWG output impedance setting
SETTLE_S = 0.3

# Scope frontend choices (tweak if needed)
COUPLING = "DC"
CH1_VDIV = 1.0       # volts/div (adjust to your circuit)
CH2_VDIV = 0.2       # volts/div (adjust to your circuit)
TIMEBASE_SDIV = 200e-6  # 200 us/div -> ~2.8 ms across 14 div ~ 2.8 cycles at 1 kHz


def rms(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x * x)))


def pick_channel(wave: dict, ch_num: int) -> tuple[np.ndarray, np.ndarray]:
    for ch in wave["channels"]:
        if ch.get("enabled") and ch.get("channel") == ch_num:
            return np.asarray(ch["time"]), np.asarray(ch["voltage"])
    raise RuntimeError(f"Channel {ch_num} not enabled / not present in waveform.")


def main() -> int:
    from lab_instruments.devices.owon_dge2070 import OwonDGE2070
    from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15

    print("Connecting instruments...")
    awg = OwonDGE2070.connect()
    dso = HantekDSO2D15.connect_by_pattern()

    try:
        # --- AWG setup ---
        print(f"Configuring AWG: {FREQ_HZ} Hz, {VPP:.3f} Vpp, {LOAD_OHMS} ohm")
        # Adjust method names if yours differ:
        awg.set_sine(channel=1, freq_hz=FREQ_HZ, amplitude_vpp=VPP, load_ohms=LOAD_OHMS)
        awg.output(1, True)
        awg.output(2, False)

        time.sleep(SETTLE_S)

        # --- DSO frontend setup ---
        print("Configuring scope frontend...")
        dso.set_timebase_scale(TIMEBASE_SDIV)
        dso.set_channel_coupling(1, COUPLING)
        dso.set_channel_coupling(2, COUPLING)
        dso.set_channel_scale(1, CH1_VDIV)
        dso.set_channel_scale(2, CH2_VDIV)

        time.sleep(SETTLE_S)

        # --- Acquire waveform ---
        print("Acquiring waveform...")
        wave = dso.read_waveform_all()

        t1, v1 = pick_channel(wave, 1)
        t2, v2 = pick_channel(wave, 2)

        # Ensure same timebase (they should be)
        t = t1 if len(t1) <= len(t2) else t2
        n = min(len(v1), len(v2), len(t))
        t = t[:n]
        v1 = v1[:n]
        v2 = v2[:n]

        print("\n=== 1 kHz waveform check ===")
        print(f"CH1 RMS:  {rms(v1):.6f} V")
        print(f"CH2 RMS:  {rms(v2):.6f} V")
        print(f"Samples:  {n}")
        print("============================\n")

        # --- Plot ---
        plt.figure(figsize=(10, 5))
        plt.plot(t * 1e3, v1, label="CH1")
        plt.plot(t * 1e3, v2, label="CH2")
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Hantek DSO2D15: CH1 & CH2 waveforms @ {FREQ_HZ:g} Hz")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return 0

    finally:
        # Be polite to hardware
        try:
            awg.output(1, False)
        except Exception:
            pass
        try:
            dso.close()
        except Exception:
            pass
        try:
            awg.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
