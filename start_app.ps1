param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload,
    [switch]$SkipDocker,
    [switch]$SkipBuild,
    [switch]$InitData,
    [switch]$WithKafka
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot "venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

$fullStackDir = Join-Path $repoRoot "prod\FullStack"
$composeFile = Join-Path $repoRoot "prod\docker-compose.yml"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Message
    Write-Host ("=" * 72)
}

function Ensure-Command {
    param(
        [string]$CommandName,
        [string]$HelpText
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName was not found. $HelpText"
    }
}

Write-Step "VECTOR unified startup"
Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Frontend: $fullStackDir"

if (-not $SkipDocker) {
    Ensure-Command -CommandName "docker" -HelpText "Install Docker Desktop or rerun with -SkipDocker."

    $dockerServices = @("mongo", "mongo-express")
    if ($WithKafka) {
        $dockerServices += "kafka"
    }

    Write-Step "Starting infrastructure with Docker"
    Write-Host ("Services: " + ($dockerServices -join ", "))
    & docker compose -f $composeFile up -d @dockerServices
} else {
    Write-Step "Skipping Docker startup"
}

if (-not $SkipBuild) {
    Ensure-Command -CommandName "npm.cmd" -HelpText "Install Node.js or rerun with -SkipBuild."

    Write-Step "Building frontend"
    Push-Location $fullStackDir
    try {
        & npm.cmd run build
    }
    finally {
        Pop-Location
    }
} else {
    Write-Step "Skipping frontend build"
}

if ($InitData) {
    Write-Step "Initializing sample data"
    & $pythonExe "realtime_risk_engine\scripts\init_mongo.py"
    & $pythonExe "realtime_risk_engine\scripts\seed_mongo_sample.py"
}

Write-Step "Starting FastAPI server"
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
Write-Host ""
Write-Host "App URLs:"
Write-Host "  UI:    http://$HostAddress`:$Port"
Write-Host "  Docs:  http://$HostAddress`:$Port/docs"
Write-Host "  Mongo: http://127.0.0.1:8081"

& $pythonExe @uvicornArgs
