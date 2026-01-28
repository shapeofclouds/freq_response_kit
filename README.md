# freq_response_kit

A small application for measuring frequency response (gain, phase, THD)
using laboratory instruments via the `lab_instruments` library.

This repository is a **consumer** of `lab_instruments` and installs it from
a released wheel (not from a working copy).

## Requirements

- Windows 11
- Python 3.12+
- Git
- Supported instruments (e.g. OWON AWG, Hantek DSO)

## Setup

In Powershell clone the repository:

```powershell
git clone https://github.com/shapeofclouds/freq_response_kit.git
cd freq_response_kit
```

#### Create and populate the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\deploy.ps1
```

This will:

- create a local .venv

- install dependencies

- install lab_instruments from a wheel

```powershell
.\.venv\Scripts\Activate.ps1
python main.py 
```
#### Notes

This project intentionally runs against an installed release
of lab_instruments, not a live repo checkout.

To upgrade the instrument library, rebuild a wheel in
lab_instruments and re-run deploy.ps1.