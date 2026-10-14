# Run this file from PowerShell in the extracted project folder.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install Docker Desktop and open a new PowerShell window."
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not running. Open Docker Desktop and wait until the engine is running."
}

docker compose ps
docker compose exec -T backend python verify_stack.py | Tee-Object -FilePath verification-results.txt
if ($LASTEXITCODE -ne 0) {
    throw "A live check failed. Read the message above and docker compose logs."
}

Write-Host "Verification passed. Results saved to verification-results.txt."
