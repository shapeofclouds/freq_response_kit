from __future__ import annotations

from lab_instruments.util.connect_errors import DeviceNotFoundError
import warnings

warnings.filterwarnings("ignore", message="TCPIP:instr resource discovery is limited*")
warnings.filterwarnings("ignore", message="TCPIP::hislip resource discovery requires*")


def check_hantek() -> bool:
    print("Checking Hantek DSO2D15...")
    try:
        from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15
        dso = HantekDSO2D15.connect_by_pattern()
        dso.close()
        print("  OK: Hantek DSO2D15 detected")
        return True
    except DeviceNotFoundError as e:
        print("  FAIL:", e)
        return False


def check_awg() -> bool:
    print("Checking OWON DGE2070 (AWG)...")
    try:
        from lab_instruments.devices.owon_dge2070 import OwonDGE2070
        awg = OwonDGE2070.connect()
        awg.scpi.close()
        print("  OK: OWON DGE2070 detected")
        return True
    except DeviceNotFoundError as e:
        print("  FAIL:", e)
        return False


def check_dmm() -> bool:
    print("Checking OWON XDM1041 (DMM)...")
    try:
        from lab_instruments.devices.owon_xdm1041 import OwonXDM1041
        dmm = OwonXDM1041.connect()
        dmm.scpi.close()
        print("  OK: OWON XDM1041 detected")
        return True
    except DeviceNotFoundError as e:
        print("  FAIL:", e)
        return False


def main() -> int:
    print("\n=== Lab hardware presence check ===\n")

    ok = True
    ok &= check_hantek()
    ok &= check_awg()
    ok &= check_dmm()

    print("\n----------------------------------")
    if ok:
        print("All devices detected. Plumbing looks good 👍")
        return 0
    else:
        print("One or more devices missing.")
        print("Check power, USB cables, and re-run.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
