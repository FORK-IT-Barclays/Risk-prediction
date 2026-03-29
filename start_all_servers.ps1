param(
    [string]$HostAddress = "127.0.0.1",
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $repoRoot "prod\FullStack"

Write-Host "Repo root: $repoRoot"
Write-Host "Starting backend first..."

Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $repoRoot "start_server.ps1"),
    "-HostAddress", $HostAddress,
    "-Port", $BackendPort,
    "-NoReload"
)

Start-Sleep -Seconds 3

Write-Host "Starting frontend dev server..."

Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", "Set-Location '$frontendDir'; npm.cmd run dev -- --host $HostAddress --port $FrontendPort"
)

Write-Host ""
Write-Host "Servers launched:"
Write-Host "  Backend:  http://$HostAddress`:$BackendPort"
Write-Host "  Frontend: http://$HostAddress`:$FrontendPort"
