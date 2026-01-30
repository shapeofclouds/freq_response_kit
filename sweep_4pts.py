import time
import numpy as np
import matplotlib.pyplot as plt

from lab_instruments.devices.owon_dge2070 import OwonDGE2070
from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15


def rms(x):
    return np.sqrt(np.mean(x**2))


def gain_phase(ch_ref, ch_dut):
    """
    Compute gain (dB) and phase (deg) using FFT fundamental.
    """
    n = len(ch_ref)
    w = np.hanning(n)

    Xr = np.fft.rfft(ch_ref * w)
    Xd = np.fft.rfft(ch_dut * w)

    k = np.argmax(np.abs(Xr[1:])) + 1

    Ar = np.abs(Xr[k])
    Ad = np.abs(Xd[k])

    ph_r = np.angle(Xr[k])
    ph_d = np.angle(Xd[k])

    gain_db = 20 * np.log10(Ad / Ar)
    phase_deg = np.degrees(ph_d - ph_r)

    # wrap to [-180, 180]
    phase_deg = (phase_deg + 180) % 360 - 180

    return gain_db, phase_deg


def main():
    freqs = [100, 300, 1000, 3000]

    print("Connecting instruments...")
    awg = OwonDGE2070.connect()
    dso = HantekDSO2D15.connect_by_pattern()

    gains = []
    phases = []

    for f in freqs:
        print(f"\n--- {f} Hz ---")

        awg.set_sine(channel=1, freq_hz=f, amplitude_vpp=0.25, load_ohms=50)
        awg.output(1, True)

        time.sleep(0.3)  # let everything settle

        wave = dso.read_waveform_all()

        chs = {c["channel"]: c for c in wave["channels"] if c["enabled"]}

        t = chs[1]["time"]
        
        v_dut = chs[1]["voltage"]
        v_ref = chs[2]["voltage"]
        
        g, p = gain_phase(v_ref, v_dut)

        gains.append(g)
        phases.append(p)

        print(f"Gain  = {g:+6.2f} dB")
        print(f"Phase = {p:+6.1f} deg")

    awg.output(1, False)
    dso.close()

    # ---- plot ----
    fig, ax = plt.subplots(2, 1, sharex=True)

    ax[0].semilogx(freqs, gains, "o-")
    ax[0].set_ylabel("Gain (dB)")
    ax[0].grid(True, which="both")

    ax[1].semilogx(freqs, phases, "o-")
    ax[1].set_ylabel("Phase (deg)")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].grid(True, which="both")

    plt.show()


if __name__ == "__main__":
    main()
