<#
.SYNOPSIS
  Join this laptop to the cluster: one GPU worker and several CPU worker processes.
  Run on every laptop, including laptop A. Each opens in its own window; closing a
  window stops those workers (that's how you "unplug" a laptop in the fault demo).

.PARAMETER Scheduler
  Laptop A's hotspot IP (printed by start_scheduler.ps1). On laptop A itself, use 127.0.0.1.

.PARAMETER Name
  A short unique name for this laptop: laptopA, laptopB, laptopC.

.PARAMETER CpuWorkers
  Tokenizer processes. Default 4 (capped at physical cores - 2): 4 tokenizers already
  outpace one laptop GPU's embedding speed, and each process costs memory.

.PARAMETER PortBase
  First worker port (default 9000; the workers use PortBase..PortBase+160). Only change it
  when simulating two laptops on one machine.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File cluster\start_workers.ps1 -Scheduler 192.168.43.10 -Name laptopB
#>
param(
    [Parameter(Mandatory = $true)][string]$Scheduler,
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$CpuWorkers = -1,
    [int]$PortBase = 9000,
    [switch]$Hidden
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:PYTHONPATH = $Repo

if ($Name -match "-(gpu|cpu)") { throw "Name must not contain '-gpu' or '-cpu' (they are added automatically)." }
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match "--name $([regex]::Escape($Name))-(gpu|cpu)(\s|$)" }
if ($running) {
    throw "$Name's workers are already running. Stop them first: powershell -ExecutionPolicy Bypass -File cluster\stop_workers.ps1 -Name $Name"
}
if ($CpuWorkers -lt 0) {
    $cores = (Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum
    $CpuWorkers = [Math]::Max(1, [Math]::Min(4, $cores - 2))
}
if ($CpuWorkers -gt 60) { $CpuWorkers = 60 }

# The address the other laptops can reach this one on (the interface that routes to the scheduler).
$route = Find-NetRoute -RemoteIPAddress $Scheduler -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress } | Select-Object -First 1
if (-not $route) { throw "No network route to $Scheduler. Is this laptop on the same hotspot?" }
$MyIp = $route.IPAddress

if (-not (Test-NetConnection -ComputerName $Scheduler -Port 8786 -InformationLevel Quiet -WarningAction SilentlyContinue)) {
    throw "Can't reach the scheduler at ${Scheduler}:8786. Is start_scheduler.ps1 running on laptop A, and did laptop A run open_firewall.ps1?"
}

$LocalDir = Join-Path $env:TEMP "scholarsense-dask\$Name"
New-Item -ItemType Directory -Force $LocalDir | Out-Null
$PidFile = Join-Path $env:TEMP "scholarsense-dask\$Name.pids"

$common = @("-m", "distributed.cli.dask_worker", "tcp://${Scheduler}:8786", "--host", $MyIp,
            "--nthreads", "1", "--memory-limit", "0", "--local-directory", $LocalDir)
$gpu = $common + @("--name", "$Name-gpu", "--nworkers", "1", "--resources", "GPU=1",
                   "--worker-port", "$PortBase", "--nanny-port", "$($PortBase + 100)")
$cpu = $common + @("--name", "$Name-cpu", "--nworkers", "$CpuWorkers", "--resources", "CPU=1",
                   "--worker-port", "$($PortBase + 1):$($PortBase + 60)",
                   "--nanny-port", "$($PortBase + 101):$($PortBase + 160)")

function Start-Group([string]$title, [string[]]$pyArgs, [string]$log) {
    if ($Hidden) {
        $p = Start-Process python -ArgumentList $pyArgs -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
    } else {
        $cmd = "title $title && python " + ($pyArgs -join " ")
        $p = Start-Process cmd -ArgumentList "/k", $cmd -PassThru
    }
    return $p.Id
}

$pids = @()
$pids += Start-Group "ScholarSense $Name GPU worker" $gpu (Join-Path $LocalDir "gpu")
$pids += Start-Group "ScholarSense $Name CPU workers ($CpuWorkers)" $cpu (Join-Path $LocalDir "cpu")
$pids | Set-Content $PidFile

Write-Host ""
Write-Host "$Name joined the cluster at $Scheduler from $MyIp" -ForegroundColor Green
Write-Host "  1 GPU worker  ($Name-gpu)"
Write-Host "  $CpuWorkers CPU workers ($Name-cpu-0 ... )"
Write-Host "Stop them by closing their windows, or: powershell -ExecutionPolicy Bypass -File cluster\stop_workers.ps1 -Name $Name"
