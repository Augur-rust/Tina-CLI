$ErrorActionPreference = 'Stop'
$packageRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$count = 0
Get-Content -LiteralPath (Join-Path $packageRoot 'SHA256SUMS.txt') -Encoding UTF8 | ForEach-Object {
    if ($_ -notmatch '^([0-9a-f]{64})  (.+)$') { throw 'Malformed checksum entry' }
    $expected = $Matches[1]
    $target = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $Matches[2]))
    if (-not $target.StartsWith($packageRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Checksum path escaped package' }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw ('Hash mismatch: ' + $Matches[2]) }
    $count++
}
Write-Host ('PASS: ' + $count + ' files verified')
