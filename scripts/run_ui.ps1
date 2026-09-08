# Start the Streamlit UI.
#
#   .\scripts\run_ui.ps1              default port 8501
#   .\scripts\run_ui.ps1 -Port 8502
#
# The backend must already be running -- start it with .\scripts\run_api.ps1,
# or start both at once with .\scripts\dev.ps1.

param(
    [int]$Port = 8501,
    [string]$ApiBase = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. See scripts/run_api.ps1 for setup." -ForegroundColor Yellow
    exit 1
}

if ($ApiBase) { $env:ASHEN_API_BASE = $ApiBase }
$target = if ($env:ASHEN_API_BASE) { $env:ASHEN_API_BASE } else { "http://127.0.0.1:8000" }

# Warn rather than fail: the UI renders a "Backend unreachable" panel with
# recovery instructions, which is more useful than refusing to start.
try {
    Invoke-WebRequest -Uri "$target/health" -TimeoutSec 3 -UseBasicParsing | Out-Null
    Write-Host "Backend reachable at $target" -ForegroundColor Green
} catch {
    Write-Host "Backend not reachable at $target" -ForegroundColor Yellow
    Write-Host "  Start it in another terminal:  .\scripts\run_api.ps1"
    Write-Host "  The UI will still open and will say the backend is down."
    Write-Host ""
}

Write-Host "UI       http://localhost:$Port" -ForegroundColor Green
Write-Host ""

& $python -m streamlit run src/ui/app.py --server.port $Port
