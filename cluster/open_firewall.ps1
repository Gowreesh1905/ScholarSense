<#
.SYNOPSIS
  One-time network setup on every laptop. Run in an ADMINISTRATOR PowerShell.

  - Marks the hotspot network as Private (Windows blocks more on Public networks).
  - Allows the cluster ports in: 8786 scheduler, 8787 dashboard, 9000-9360 workers.
  - Removes "block python" rules left by clicking Cancel on an earlier firewall prompt
    (block rules win over allow rules).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File cluster\open_firewall.ps1
  powershell -ExecutionPolicy Bypass -File cluster\open_firewall.ps1 -InterfaceAlias "Wi-Fi 2"
#>
#Requires -RunAsAdministrator
param([string]$InterfaceAlias = "Wi-Fi")
$ErrorActionPreference = "Stop"

Set-NetConnectionProfile -InterfaceAlias $InterfaceAlias -NetworkCategory Private
Write-Host "Network '$InterfaceAlias' set to Private."

Remove-NetFirewallRule -DisplayName "ScholarSense Dask" -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName "ScholarSense Dask" -Direction Inbound -Protocol TCP `
    -LocalPort 8786-8787, 9000-9360 -Action Allow -Profile Private | Out-Null
Write-Host "Allowed inbound TCP 8786-8787 and 9000-9360 on Private networks."

$blocks = Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue |
    Where-Object { (Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $_).Program -like "*python*" }
foreach ($rule in $blocks) {
    Remove-NetFirewallRule -Name $rule.Name
    Write-Host "Removed blocking rule: $($rule.DisplayName)"
}
Write-Host "Done."
