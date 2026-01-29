from __future__ import annotations

def try_cmd(label, fn):
    try:
        out = fn()
        print(f"OK  {label}: {out}")
        return True
    except Exception as e:
        print(f"FAIL {label}: {e}")
        return False

def main() -> int:
    from lab_instruments.devices.hantek_dso2d15 import HantekDSO2D15

    dso = HantekDSO2D15.connect_by_pattern()

    # Identity (if supported)
    try_cmd("*IDN?", lambda: dso.query("*IDN?").strip())

    # Timebase
    try_cmd("TIMEBASE:SCALE?", lambda: dso.get_timebase())
    try_cmd("TIMEBASE:SCALE 1e-3", lambda: (dso.set_timebase(1e-3), "set"))

    # Per-channel
    for ch in (1, 2):
        try_cmd(f"CH{ch} COUPLING?", lambda ch=ch: dso.get_coupling(ch))
        try_cmd(f"CH{ch} COUPLING DC", lambda ch=ch: (dso.set_coupling(ch, "DC"), "set"))

        try_cmd(f"CH{ch} SCALE?", lambda ch=ch: dso.get_scale(ch))
        try_cmd(f"CH{ch} SCALE 0.2", lambda ch=ch: (dso.set_scale(ch, 0.2), "set"))

        try_cmd(f"CH{ch} OFFSET?", lambda ch=ch: dso.get_offset(ch))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
