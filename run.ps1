# FALLBACK_CAMOTE: start the app, then open http://127.0.0.1:8765/ in Chrome.
# Uses the Python that already has everything installed (hey-trafficam's venv). Nothing is installed or downloaded.
$py = if ($env:FALLBACK_PYTHON) { $env:FALLBACK_PYTHON } else { "C:\Users\ahmir\Desktop\hey-trafficam\venv\Scripts\python.exe" }
Set-Location $PSScriptRoot
& $py server.py
