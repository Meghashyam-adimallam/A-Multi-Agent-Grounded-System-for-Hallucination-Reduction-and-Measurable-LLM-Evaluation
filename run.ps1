# Multi-Agent Grounded RAG — Run script
# Usage: .\run.ps1 index   — index documents
#        .\run.ps1 api     — start API server
#        .\run.ps1 both   — index then start API

param([string]$Action = "both")

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# Activate venv
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating venv..."
    python -m venv venv
}
& ".\venv\Scripts\Activate.ps1"

if ($Action -eq "index" -or $Action -eq "both") {
    Write-Host "Indexing documents..."
    python scripts/index_documents.py
}

if ($Action -eq "api" -or $Action -eq "both") {
    Write-Host "Starting API on http://127.0.0.1:8000"
    Write-Host "  Docs: http://127.0.0.1:8000/docs  |  Health: http://127.0.0.1:8000/health"
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000
}
