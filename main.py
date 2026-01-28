# main.py
"""
freq_response_kit: minimal consumer app for lab_instruments.

Phase 1: prove we can import and discover devices.
Phase 2: add a tiny 1 kHz measurement.
Phase 3: full sweep + CSV + plots.
"""

from __future__ import annotations


def main() -> int:
    # Import from installed wheel (consumer mode)
    import lab_instruments.scpi.session as session_mod

    print("freq_response_kit OK")
    print("Using lab_instruments from:", session_mod.__file__)

    # Optional: quick visibility checks (no hardware commands yet)
    try:
        import pyvisa
        rm = pyvisa.ResourceManager("@py")
        print("VISA resources:")
        for r in rm.list_resources():
            print(" ", r)
    except Exception as e:
        print("VISA probe skipped/failed:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
