# Start the backend and the UI together, each in its own window.
#
#   .\scripts\dev.ps1            start both
#   .\scripts\dev.ps1 -Stop      stop whatever is on both ports
#
# Two windows rather than one, on purpose: when something breaks mid-demo you
# want to see which half produced the error. The API window shows every request
# and any traceback; the Streamlit window shows the UI's own log.

param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Stop-Port([int]$port, [string]$label) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn[0].OwningProcess -Force
        Write-Host "Stopped $label on port $port (PID $($conn[0].OwningProcess))" -ForegroundColor Yellow
    } else {
        Write-Host "Nothing listening on port $port" -ForegroundColor DarkGray
    }
}

if ($Stop) {
    Stop-Port $ApiPort "API"
    Stop-Port $UiPort "UI"
    exit 0
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Set one up first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

foreach ($port in @($ApiPort, $UiPort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "Port $port is already in use. Free both ports with:" -ForegroundColor Yellow
        Write-Host "  .\scripts\dev.ps1 -Stop"
        exit 1
    }
}

Write-Host "Starting backend on port $ApiPort..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$repo'; & '$python' -m uvicorn src.api.main:app --reload --port $ApiPort"
)

# Give uvicorn a moment to bind before the UI probes /health, so the first
# thing on screen is not a spurious "backend unreachable".
Start-Sleep -Seconds 4

Write-Host "Starting UI on port $UiPort..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$repo'; & '$python' -m streamlit run src/ui/app.py --server.port $UiPort"
)

Write-Host ""
Write-Host "  UI      http://localhost:$UiPort" -ForegroundColor Green
Write-Host "  API     http://127.0.0.1:$ApiPort" -ForegroundColor Green
Write-Host "  Docs    http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
Write-Host ""
Write-Host "  Stop both:  .\scripts\dev.ps1 -Stop" -ForegroundColor DarkGray
