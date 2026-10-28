param([string]$OutputRoot)
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('tina-demo-' + [guid]::NewGuid().ToString('N')) }
if (Test-Path -LiteralPath $OutputRoot) { throw 'OutputRoot must be a new directory.' }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$workspace = Join-Path $OutputRoot 'workspace'
New-Item -ItemType Directory -Path $workspace | Out-Null
[System.IO.File]::WriteAllText((Join-Path $workspace 'sample.txt'), "alpha`nreceipt-first harness`n", [System.Text.UTF8Encoding]::new($false))
& git -C $workspace init --quiet
if ($LASTEXITCODE -ne 0) { throw 'git init failed' }
$binary = Join-Path $packageRoot 'bin\tina.exe'
Write-Host 'CPU scripted replay: no model inference. Tool calls and archives are real.'
& $binary doctor --workspace $workspace
if ($LASTEXITCODE -ne 0) { throw 'doctor failed' }
& $binary replay --workspace $workspace --scenario (Join-Path $packageRoot 'fixtures\read-and-status.json') --archive (Join-Path $OutputRoot 'archive')
if ($LASTEXITCODE -ne 0) { throw 'replay failed' }
Write-Host ('Retained workspace and archive: ' + $OutputRoot)
