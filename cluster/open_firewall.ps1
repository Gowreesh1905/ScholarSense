<#
.SYNOPSIS
  One-time network setup on every laptop. Run from an ADMINISTRATOR window.

  - Allows the cluster ports in (8786 scheduler, 8787 dashboard, 9000-9360 workers), but only
    from devices on the same local network (the hotspot), on any network type, so it works
    whether Windows calls the hotspot Public or Private and whatever the Wi-Fi adapter is named.
  - Marks the currently connected network(s) Private (best effort).
  - Removes "block python" rules left by clicking Cancel on an earlier firewall prompt
    (block rules win over allow rules).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File D:\ScholarSense\cluster\open_firewall.ps1
  powershell -ExecutionPolicy Bypass -File D:\ScholarSense\cluster\open_firewall.ps1 -Remove   # undo after the demo
#>
#Requires -RunAsAdministrator
param([switch]$Remove)
$RuleName = "ScholarSense Dask"

Remove-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue
if ($Remove) {
    Write-Host "Removed the '$RuleName' firewall rule."
    exit 0
}

New-NetFirewallRule -DisplayName $RuleName -Direction Inbound -Protocol TCP `
    -LocalPort 8786-8787, 9000-9360 -RemoteAddress LocalSubnet -Action Allow -Profile Any | Out-Null
Write-Host "Allowed inbound TCP 8786-8787 and 9000-9360 from the local network."

foreach ($p in Get-NetConnectionProfile) {
    try {
        Set-NetConnectionProfile -InterfaceIndex $p.InterfaceIndex -NetworkCategory Private -ErrorAction Stop
        Write-Host "Network '$($p.Name)' ($($p.InterfaceAlias)) set to Private."
    } catch {
        Write-Host "Could not set '$($p.Name)' to Private (fine: the rule above works either way)."
    }
}

$blocks = Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue |
    Where-Object { (Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $_).Program -like "*python*" }
foreach ($rule in $blocks) {
    Remove-NetFirewallRule -Name $rule.Name
    Write-Host "Removed blocking rule: $($rule.DisplayName)"
}
Write-Host "Done."
