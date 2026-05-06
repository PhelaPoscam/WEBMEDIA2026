# ASA CLI Demonstration Script
# Generates a research-grade stimulus playlist using clean, descriptive profiles.

$ErrorActionPreference = "Stop"

# Locate Project Root (parent directory of the demo folder)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExecutable)) {
    $PythonExecutable = "python"
}

# Configuration
$Participant = "demo/inputs/profiles/subject_001.json"
$Blueprint = "demo/inputs/blueprints/stress_detection.json"
$Dataset = "dataset/All_Models_Normalized_Comparison.csv"
$Strategy = "euclidean"
$OutputDir = "demo/outputs/research_session"

# Change to Project Root to resolve relative paths and modules correctly
Push-Location $ProjectRoot
try {
    # Setup Output
    if (-not (Test-Path $OutputDir)) {
        New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    }

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "   ASA Stress Detection Demo" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan

    Write-Host "[1/2] Processing participant and blueprint..." -ForegroundColor Yellow
    Write-Host "Strategy: $Strategy" -ForegroundColor Gray

    # Execute CLI using the identified Python executable
    & $PythonExecutable -m src.prepare_session_cli prepare-session `
        --participant-file $Participant `
        --blueprint-file $Blueprint `
        --dataset-csv $Dataset `
        --matching-strategy $Strategy `
        --output-json "$OutputDir/playlist_stress.json" `
        --output-csv "$OutputDir/playlist_stress.csv" `
        --output-report "$OutputDir/transparency_report_stress.md"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[2/2] Success! Stress-detection playlist generated." -ForegroundColor Green
        Write-Host "`nGenerated Files:" -ForegroundColor White
        Get-ChildItem $OutputDir | Where-Object { $_.Name -like "*stress*" } | Select-Object Name, @{Name="Size(KB)"; Expression={[math]::round($_.Length/1KB, 2)}} | Format-Table -AutoSize
        
        Write-Host "Preview of generated CSV:" -ForegroundColor Gray
        Import-Csv "$OutputDir/playlist_stress.csv" | Select-Object -First 5 | Format-Table -AutoSize
    } else {
        Write-Host "`nError: Session generation failed." -ForegroundColor Red
    }
} finally {
    # Restore original directory
    Pop-Location
}

