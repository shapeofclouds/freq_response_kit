from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message="TCPIP:instr resource discovery is limited*")
warnings.filterwarnings("ignore", message="TCPIP::hislip resource discovery requires*")
warnings.filterwarnings("ignore", message="read string doesn't end with termination characters*")


def q(scpi, cmd: str) -> str:
    """Query and return raw string (not stripped) so we can diagnose weird replies."""
    resp = scpi.query(cmd)
    if resp is None:
        return ""
    return str(resp)


def q_stripped(scpi, cmd: str) -> str:
    return q(scpi, cmd).strip()


def show(scpi, cmd: str) -> None:
    r = q(scpi, cmd)
    print(f"{cmd} -> {r.strip()}   (repr={r!r})")


def try_set_and_readback(scpi, label: str, set_cmd: str, get_cmd: str):
    print(f"\n--- {label} ---")
    print("SET:", set_cmd)
    try:
        scpi.write(set_cmd)
        time.sleep(0.1)
        show(scpi, get_cmd)
    except Exception as e:
        print("ERROR:", repr(e))


def try_err(scpi):
    try:
        show(scpi, "SYST:ERR?")
    except Exception as e:
        print("SYST:ERR? not available:", repr(e))


def main() -> int:
    from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15

    dso = HantekDSO2D15.connect_by_pattern()
    scpi = dso.scpi

    # Make state deterministic: ensure CH1 is displayed/enabled (some scopes answer oddly if channel is OFF)
    try:
        scpi.write(":CHANnel1:DISPlay ON")
        time.sleep(0.05)
    except Exception:
        pass

    print("Baseline queries:")
    # Prefer long-form for queries (your earlier experiments strongly suggest this is safer)
    show(scpi, ":CHANnel1:SCALe?")
    show(scpi, ":CHANnel1:COUPling?")
    #try_err(scpi)

    # Use CHAN1 for SET (seems accepted), but CHANnel1 for GET (seems more reliable)
    try_set_and_readback(scpi, "Set scale", ":CHAN1:SCALe 0.2", ":CHANnel1:SCALe?")
    #try_err(scpi) 

    try_set_and_readback(scpi, "Set coupling", ":CHAN1:COUPling DC", ":CHANnel1:COUPling?")
    #try_err(scpi)

    try_set_and_readback(scpi, "Set timebase", ":TIMEBASE:SCALE 1e-3", ":TIMEBASE:SCALE?")
    #try_err(scpi)

    dso.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
