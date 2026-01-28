$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv")) { python -m venv .\.venv }
& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip

# Install latest wheel from your local producer repo
python -m pip install --upgrade ..\lab_instruments\dist\lab_instruments-*.whl

# App deps
python -m pip install --upgrade numpy matplotlib scipy pyvisa pyvisa-py pyserial pyusb

Write-Host "[deploy] freq_response_kit OK"
