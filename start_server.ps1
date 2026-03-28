param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonExe = $venvPython
} else {
    $pythonExe = "python"
}

Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Starting FastAPI server for realtime_risk_engine..."

$uvicornArgs = @(
    "-m", "uvicorn",
    "realtime_risk_engine.src.server:app",
    "--host", $HostAddress,
    "--port", $Port.ToString()
)

if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

Write-Host ("Command: {0} {1}" -f $pythonExe, ($uvicornArgs -join " "))
& $pythonExe @uvicornArgs
