param([switch]$Offline)
$ErrorActionPreference = 'Stop'
$packageRoot = Split-Path -Parent $PSScriptRoot
$extra = @()
if ($Offline) { $extra += '--offline' }
Push-Location -LiteralPath $packageRoot
try {
    & cargo build --workspace --locked @extra
    if ($LASTEXITCODE -ne 0) { throw 'build failed' }
    $metadataText = & cargo metadata --no-deps --locked --format-version 1 @extra
    if ($LASTEXITCODE -ne 0) { throw 'metadata failed' }
    $targetDirectory = ($metadataText | ConvertFrom-Json).target_directory
    $tina = Join-Path $targetDirectory 'debug/tina.exe'
    $accepted = (& $tina context-check --input-tokens 258048 | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $accepted.admitted) { throw 'exact context boundary rejected' }
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $rejectedText = & $tina context-check --input-tokens 258049 2>$null
        $rejectedCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousErrorPreference }
    $rejected = $rejectedText | ConvertFrom-Json
    if ($rejectedCode -eq 0 -or $rejected.admitted -or $rejected.rejection.kind -ne 'usable-input-limit') { throw 'overflow admission did not fail correctly' }
    & cargo test --workspace --all-features --locked @extra
    if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
    & python (Join-Path $PSScriptRoot 'test_live_http.py') --binary $tina
    if ($LASTEXITCODE -ne 0) { throw 'live HTTP protocol tests failed' }
    & cargo fmt --all -- --check
    if ($LASTEXITCODE -ne 0) { throw 'format check failed' }
    & cargo clippy --workspace --all-targets --all-features --locked @extra -- -D warnings
    if ($LASTEXITCODE -ne 0) { throw 'clippy failed' }
} finally { Pop-Location }
Write-Host 'PASS: build, tests, format and clippy. Protocol fixtures do not qualify live model quality; see separate model acceptance.'
