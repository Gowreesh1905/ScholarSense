<#
.SYNOPSIS
  Stop this laptop's workers (the same as closing their windows).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File cluster\stop_workers.ps1 -Name laptopB
#>
param([Parameter(Mandatory = $true)][string]$Name)

# Find the workers by the --name they were started with, so this works even if
# start_workers.ps1 was run more than once.
$pattern = "--name $([regex]::Escape($Name))-(gpu|cpu)(\s|$)"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }
foreach ($p in $procs) {
    # /T also stops the worker processes each Dask nanny started.
    taskkill /PID $p.ProcessId /T /F 2>$null | Out-Null
}

# Close the console windows start_workers.ps1 opened.
$PidFile = Join-Path $env:TEMP "scholarsense-dask\$Name.pids"
if (Test-Path $PidFile) {
    foreach ($id in Get-Content $PidFile) { taskkill /PID $id /T /F 2>$null | Out-Null }
    Remove-Item $PidFile
}
Write-Host "Stopped $($procs.Count) worker group(s) for $Name."
