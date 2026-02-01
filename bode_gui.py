from __future__ import annotations
import json
import os
import queue
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import warnings
import logging

# Suppress noisy pyvisa-py TCPIP warnings (adjust if needed)
#warnings.filterwarnings("ignore", module=r"pyvisa_py\.tcpip")
#warnings.filterwarnings("ignore", category=ResourceWarning, module=r"pyvisa_py\..*")

#logging.getLogger("pyvisa").setLevel(logging.ERROR)
#logging.getLogger("pyvisa_py").setLevel(logging.ERROR)


APP_NAME = "bode_gui"
CONFIG_DIR = Path(os.environ.get("APPDATA", ".")) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    python_exe: str = ""      # path to python.exe (prefer venv python)
    script_path: str = ""     # path to sweep_bode.py
    start_hz: float = 1.0
    stop_hz: float = 100_000.0
    points_per_decade: int = 10
    awg_channel: int = 1
    awg_vpp: float = 0.25
    awg_load_ohms: float = 50.0
    ch_dut: int = 1
    ch_ref: int = 2
    coupling: str = "AC"
    vdiv_ref: float = 0.2
    vdiv_dut: float = 2.0
    settle_s: float = 0.25
    meas_avg: int = 2
    out_csv: str = ""         # optional; blank = default inside sweep script
    markers: bool = False

def load_config() -> Config:
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return Config(**data)
    except Exception:
        pass
    return Config()


def save_config(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bode Sweep Launcher")
        self.geometry("980x650")

        self.cfg = load_config()

        self.proc: subprocess.Popen[str] | None = None
        self.q: "queue.Queue[str]" = queue.Queue()

        self._build_ui()
        self._load_cfg_into_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        self.markers_var = tk.BooleanVar(value=False)  # default OFF

        # Python interpreter picker
        self.python_var = tk.StringVar()
        ttk.Label(top, text="Python interpreter:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.python_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(top, text="Browse…", command=self._browse_python).grid(row=0, column=2)

        # Script picker
        self.script_var = tk.StringVar()
        ttk.Label(top, text="Sweep script:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.script_var, width=70).grid(row=1, column=1, sticky="we")
        ttk.Button(top, text="Browse…", command=self._browse_script).grid(row=1, column=2)

        top.columnconfigure(1, weight=1)

        # Parameters
        params = ttk.LabelFrame(self, text="Parameters")
        params.pack(fill="x", **pad)

        def add_field(row, col, label, var, width=12):
            ttk.Label(params, text=label).grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Entry(params, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        self.start_var = tk.StringVar()
        self.stop_var = tk.StringVar()
        self.ppd_var = tk.StringVar()

        self.awg_ch_var = tk.StringVar()
        self.awg_vpp_var = tk.StringVar()
        self.awg_load_var = tk.StringVar()

        self.ch_dut_var = tk.StringVar()
        self.ch_ref_var = tk.StringVar()
        self.coupling_var = tk.StringVar()

        self.vdiv_ref_var = tk.StringVar()
        self.vdiv_dut_var = tk.StringVar()
        self.settle_var = tk.StringVar()
        self.measavg_var = tk.StringVar()

        self.outcsv_var = tk.StringVar()

        add_field(0, 0, "Start Hz", self.start_var)
        add_field(0, 2, "Stop Hz", self.stop_var)
        add_field(0, 4, "Pts/Dec", self.ppd_var)

        add_field(1, 0, "AWG ch", self.awg_ch_var)
        add_field(1, 2, "AWG Vpp", self.awg_vpp_var)
        add_field(1, 4, "AWG load Ω", self.awg_load_var)

        add_field(2, 0, "CH DUT", self.ch_dut_var)
        add_field(2, 2, "CH REF", self.ch_ref_var)

        ttk.Label(params, text="Coupling").grid(row=2, column=4, sticky="w", padx=8, pady=4)
        self.coupling_combo = ttk.Combobox(params, textvariable=self.coupling_var, values=["AC", "DC"], width=10, state="readonly")
        self.coupling_combo.grid(row=2, column=5, sticky="w", padx=8, pady=4)

        add_field(3, 0, "V/div REF", self.vdiv_ref_var)
        add_field(3, 2, "V/div DUT", self.vdiv_dut_var)
        add_field(3, 4, "Settle s", self.settle_var)

        add_field(4, 0, "Meas avg", self.measavg_var)

        ttk.Label(params, text="Out CSV (optional)").grid(row=4, column=2, sticky="w", padx=8, pady=4)
        ttk.Entry(params, textvariable=self.outcsv_var, width=36).grid(row=4, column=3, columnspan=2, sticky="we", padx=8, pady=4)
        ttk.Button(params, text="Browse…", command=self._browse_outcsv).grid(row=4, column=5, padx=8, pady=4)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)

        ttk.Checkbutton(btns, text="Show markers", variable=self.markers_var).pack(side="left", padx=8)

        self.run_btn = ttk.Button(btns, text="Run sweep", command=self._run)
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        ttk.Button(btns, text="Save settings", command=self._save).pack(side="left", padx=8)
        ttk.Button(btns, text="Clear log", command=self._clear_log).pack(side="left", padx=8)

        # Log output
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)

        self.log = tk.Text(log_frame, wrap="word", height=20)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", padx=12, pady=4)

    def _load_cfg_into_ui(self) -> None:
        c = self.cfg
        self.python_var.set(c.python_exe)
        self.script_var.set(c.script_path)

        self.start_var.set(str(c.start_hz))
        self.stop_var.set(str(c.stop_hz))
        self.ppd_var.set(str(c.points_per_decade))

        self.awg_ch_var.set(str(c.awg_channel))
        self.awg_vpp_var.set(str(c.awg_vpp))
        self.awg_load_var.set(str(c.awg_load_ohms))

        self.ch_dut_var.set(str(c.ch_dut))
        self.ch_ref_var.set(str(c.ch_ref))
        self.coupling_var.set(c.coupling)

        self.vdiv_ref_var.set(str(c.vdiv_ref))
        self.vdiv_dut_var.set(str(c.vdiv_dut))
        self.settle_var.set(str(c.settle_s))
        self.measavg_var.set(str(c.meas_avg))

        self.outcsv_var.set(c.out_csv or "")

    def _ui_to_cfg(self) -> Config:
        def f(x: str) -> float: return float(x.strip())
        def i(x: str) -> int: return int(x.strip())

        return Config(
            python_exe=self.python_var.get().strip(),
            script_path=self.script_var.get().strip(),
            start_hz=f(self.start_var.get()),
            stop_hz=f(self.stop_var.get()),
            points_per_decade=i(self.ppd_var.get()),
            awg_channel=i(self.awg_ch_var.get()),
            awg_vpp=f(self.awg_vpp_var.get()),
            awg_load_ohms=f(self.awg_load_var.get()),
            ch_dut=i(self.ch_dut_var.get()),
            ch_ref=i(self.ch_ref_var.get()),
            coupling=self.coupling_var.get().strip().upper(),
            vdiv_ref=f(self.vdiv_ref_var.get()),
            vdiv_dut=f(self.vdiv_dut_var.get()),
            settle_s=f(self.settle_var.get()),
            meas_avg=i(self.measavg_var.get()),
            out_csv=self.outcsv_var.get().strip(),
        )

    def _validate(self, c: Config) -> None:
        if not c.python_exe or not Path(c.python_exe).exists():
            raise ValueError("Please select a valid python.exe (preferably your venv).")
        if not c.script_path or not Path(c.script_path).exists():
            raise ValueError("Please select the sweep_bode.py script.")
        if c.start_hz <= 0 or c.stop_hz <= 0 or c.stop_hz <= c.start_hz:
            raise ValueError("Start/Stop Hz must be > 0 and Stop > Start.")
        if c.points_per_decade < 1:
            raise ValueError("Points/Dec must be >= 1.")
        if c.ch_dut == c.ch_ref:
            raise ValueError("CH DUT and CH REF must be different.")
        if c.coupling not in ("AC", "DC"):
            raise ValueError("Coupling must be AC or DC.")
        if c.out_csv:
            outp = Path(c.out_csv)
            if outp.parent and not outp.parent.exists():
                raise ValueError("Out CSV folder does not exist (or pick a different path).")

    def _build_cmd(self, c: Config) -> list[str]:
        cmd = [
            c.python_exe,
            c.script_path,
            "--start-hz", str(c.start_hz),
            "--stop-hz", str(c.stop_hz),
            "--points-per-decade", str(c.points_per_decade),
            "--awg-channel", str(c.awg_channel),
            "--awg-vpp", str(c.awg_vpp),
            "--awg-load-ohms", str(c.awg_load_ohms),
            "--ch-dut", str(c.ch_dut),
            "--ch-ref", str(c.ch_ref),
            "--coupling", c.coupling,
            "--vdiv-ref", str(c.vdiv_ref),
            "--vdiv-dut", str(c.vdiv_dut),
            "--settle-s", str(c.settle_s),
            "--meas-avg", str(c.meas_avg),
        ]
        if c.out_csv:
            cmd += ["--out", c.out_csv]

        if self.markers_var.get():
            cmd += ["--markers"]
        
        return cmd

    def _run(self) -> None:
        if self.proc is not None:
            messagebox.showinfo("Already running", "A sweep is already running.")
            return

        try:
            c = self._ui_to_cfg()
            self._validate(c)
            save_config(c)
            self.cfg = c
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return

        cmd = self._build_cmd(self.cfg)

        # Important: set working directory to the script's folder so relative imports/files behave
        workdir = str(Path(self.cfg.script_path).resolve().parent)

        self._append_log("Running:\n  " + " ".join(cmd) + f"\nWorking dir: {workdir}\n\n")
        self.status_var.set("Running sweep…")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        # Start process
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            self.proc = None
            self.run_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.status_var.set("Ready.")
            messagebox.showerror("Failed to start", str(e))
            return

        # Reader thread
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self.proc is not None
        try:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.q.put(line)
        finally:
            rc = self.proc.wait()
            self.q.put(f"\nProcess exited with code {rc}\n")
            self.q.put("__DONE__")

    def _stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self._append_log("\nTerminating process…\n")
        except Exception as e:
            self._append_log(f"\nTerminate failed: {e}\n")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item == "__DONE__":
                    self.proc = None
                    self.run_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("Ready.")
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, s: str) -> None:
        self.log.insert("end", s)
        self.log.see("end")

    def _clear_log(self) -> None:
        self.log.delete("1.0", "end")

    def _save(self) -> None:
        try:
            c = self._ui_to_cfg()
            self._validate(c)
            save_config(c)
            self.cfg = c
            self.status_var.set(f"Saved settings to {CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))

    def _browse_python(self) -> None:
        path = filedialog.askopenfilename(
            title="Select python.exe (preferably from your venv)",
            filetypes=[("Python", "python.exe"), ("Executables", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self.python_var.set(path)

    def _browse_script(self) -> None:
        path = filedialog.askopenfilename(
            title="Select sweep_bode.py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
        )
        if path:
            self.script_var.set(path)

            auto_py = find_local_venv_python(path)
            if auto_py and not self.python_var.get():
                self.python_var.set(auto_py)


    def _browse_outcsv(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Choose CSV output file",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.outcsv_var.set(path)


if __name__ == "__main__":
    App().mainloop()
