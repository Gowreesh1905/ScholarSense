<#
.SYNOPSIS
  Start the Dask scheduler. Run on laptop A only, and leave the window open.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File cluster\start_scheduler.ps1
#>
param(
    [int]$Port = 8786,
    [int]$DashboardPort = 8787
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$env:PYTHONPATH = $Repo
# Notice a laptop that drops off the network within 30 s (Dask's default is 5 minutes).
$env:DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL = "30s"

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" }
Write-Host ""
Write-Host "Starting the ScholarSense scheduler." -ForegroundColor Cyan
Write-Host "The other laptops connect to this laptop's hotspot (Wi-Fi) address:"
foreach ($ip in $ips) { Write-Host ("  {0,-16} {1}" -f $ip.IPAddress, $ip.InterfaceAlias) }
Write-Host "Dashboard: http://localhost:$DashboardPort/status"
Write-Host ""

python -m distributed.cli.dask_scheduler --host 0.0.0.0 --port $Port --dashboard-address ":$DashboardPort"
