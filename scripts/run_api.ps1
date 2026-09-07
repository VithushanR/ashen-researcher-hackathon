# Start the FastAPI backend.
#
#   .\scripts\run_api.ps1              default port 8000
#   .\scripts\run_api.ps1 -Port 8080   somewhere else
#   .\scripts\run_api.ps1 -NoReload    no auto-restart on file changes
#
# Run from anywhere; the script locates the repo root itself.

param(
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Prefer the project venv over whatever python happens to be on PATH, so this
# works the same whether or not the venv is activated in the caller's shell.
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Create one first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    exit 1
}

# A stale server on this port is the most common reason a change appears to
# have no effect, so say so plainly rather than letting uvicorn fail obscurely.
$inUse = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($inUse) {
    Write-Host "Port $Port is already in use by PID $($inUse[0].OwningProcess)." -ForegroundColor Yellow
    Write-Host "  Stop it:  taskkill /PID $($inUse[0].OwningProcess) /F"
    Write-Host "  Or:       .\scripts\run_api.ps1 -Port 8001"
    exit 1
}

$args = @("-m", "uvicorn", "src.api.main:app", "--port", $Port)
if (-not $NoReload) { $args += "--reload" }

Write-Host "API      http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Docs     http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host "Health   http://127.0.0.1:$Port/health" -ForegroundColor Green
Write-Host ""

& $python @args
