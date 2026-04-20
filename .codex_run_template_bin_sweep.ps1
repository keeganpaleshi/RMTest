$ErrorActionPreference = "Stop"

$workspace = "C:\Users\keega\Radon\RMTest"
$baseConfig = Join-Path $workspace ".codex_run_merged_output_config.yaml"
$inputCsv = Join-Path $workspace "merged_output.csv"
$sessionId = Get-Date -Format "yyyyMMddTHHmmssZ"
$sweepRoot = Join-Path $workspace ".codex_run_template_bin_sweep_$sessionId"
$configDir = Join-Path $sweepRoot "configs"
$logDir = Join-Path $sweepRoot "logs"
$outputRoot = Join-Path $sweepRoot "outputs"
$summaryCsv = Join-Path $sweepRoot "sweep_summary.csv"
$statusJson = Join-Path $sweepRoot "status.json"

New-Item -ItemType Directory -Force -Path $sweepRoot, $configDir, $logDir, $outputRoot | Out-Null

$baseLines = Get-Content $baseConfig
$summaryHeader = "hour,bin_seconds,status,exit_code,run_dir,timestamp_dir,template_plot_dir,base_plot_count,log_plot_count,start_utc,end_utc"
Set-Content -Path $summaryCsv -Value $summaryHeader -Encoding UTF8

function Write-Status {
    param(
        [string]$Phase,
        [int]$CurrentHour,
        [string]$CurrentRunDir,
        [string]$UpdatedAt
    )
    $status = [ordered]@{
        session_id = $sessionId
        phase = $Phase
        current_hour = $CurrentHour
        current_run_dir = $CurrentRunDir
        base_config = $baseConfig
        input_csv = $inputCsv
        summary_csv = $summaryCsv
        updated_at = $UpdatedAt
    }
    $status | ConvertTo-Json | Set-Content -Path $statusJson -Encoding UTF8
}

Write-Status -Phase "starting" -CurrentHour 0 -CurrentRunDir "" -UpdatedAt ((Get-Date).ToString("o"))

for ($hour = 1; $hour -le 24; $hour++) {
    $binSeconds = $hour * 3600
    $hourTag = "{0:D2}h" -f $hour
    $configPath = Join-Path $configDir "template_bin_$hourTag.yaml"
    $runDir = Join-Path $outputRoot "template_bin_$hourTag"
    $stdoutLog = Join-Path $logDir "template_bin_$hourTag.stdout.log"
    $stderrLog = Join-Path $logDir "template_bin_$hourTag.stderr.log"

    $contentLines = foreach ($line in $baseLines) {
        if ($line -match '^\s*plot_time_bin_width_s:\s*\d+\s*$') {
            "  plot_time_bin_width_s: $binSeconds"
        } else {
            $line
        }
    }
    Set-Content -Path $configPath -Value $contentLines -Encoding UTF8
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    Write-Status -Phase "running" -CurrentHour $hour -CurrentRunDir $runDir -UpdatedAt ((Get-Date).ToString("o"))
    $startUtc = (Get-Date).ToUniversalTime().ToString("o")

    $proc = Start-Process `
        -FilePath "python" `
        -ArgumentList @(
            "analyze.py",
            "--config", $configPath,
            "--input", $inputCsv,
            "--output-dir", $runDir
        ) `
        -WorkingDirectory $workspace `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -PassThru `
        -Wait `
        -NoNewWindow

    $endUtc = (Get-Date).ToUniversalTime().ToString("o")
    $exitCode = [int]$proc.ExitCode
    $status = if ($exitCode -eq 0) { "ok" } else { "failed" }

    $timestampDir = Get-ChildItem -Path $runDir -Directory | Sort-Object Name | Select-Object -Last 1
    $templatePlotDir = $null
    $basePlotCount = 0
    $logPlotCount = 0
    $timestampDirPath = ""
    $templatePlotDirPath = ""

    if ($null -ne $timestampDir) {
        $timestampDirPath = $timestampDir.FullName
        $templatePlotDir = Join-Path $timestampDir.FullName "template_bin_fits"
        if (Test-Path $templatePlotDir) {
            $templatePlotDirPath = $templatePlotDir
            $basePlotCount = (
                Get-ChildItem -Path $templatePlotDir -Filter "template_fit_bin_*.png" |
                Where-Object { $_.Name -notlike "*_log.png" } |
                Measure-Object
            ).Count
            $logPlotCount = (
                Get-ChildItem -Path $templatePlotDir -Filter "*_log.png" |
                Measure-Object
            ).Count
        }
    }

    $row = @(
        $hour,
        $binSeconds,
        $status,
        $exitCode,
        $runDir,
        $timestampDirPath,
        $templatePlotDirPath,
        $basePlotCount,
        $logPlotCount,
        $startUtc,
        $endUtc
    ) -join ","
    Add-Content -Path $summaryCsv -Value $row -Encoding UTF8
}

Write-Status -Phase "completed" -CurrentHour 24 -CurrentRunDir "" -UpdatedAt ((Get-Date).ToString("o"))
