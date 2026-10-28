param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
Push-Location -LiteralPath $repoRoot
try {
    $dirty = & git status --porcelain
    if ($LASTEXITCODE -ne 0) { throw 'Run from a Git checkout.' }
    if ($dirty) { throw 'Commit source changes before packaging.' }
    $revision = (& git rev-parse HEAD).Trim()
    $versionLine = Get-Content -LiteralPath 'crates/tina-harness/Cargo.toml' | Where-Object { $_ -match '^version = "([^"]+)"' } | Select-Object -First 1
    if ($versionLine -notmatch '^version = "([^"]+)"') { throw 'Missing package version.' }
    $version = 'v' + $Matches[1]
    if (-not $SkipBuild) {
        & cargo build --workspace --release --locked
        if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }
    }
    $metadataText = & cargo metadata --no-deps --format-version 1 --locked
    if ($LASTEXITCODE -ne 0) { throw 'Cargo metadata failed.' }
    $metadata = $metadataText | ConvertFrom-Json
    $binaryRoot = Join-Path $metadata.target_directory 'release'
    $dist = Join-Path $repoRoot ('dist/' + $version + '-' + $revision.Substring(0, 8))
    if (Test-Path -LiteralPath $dist) { throw ('Package output already exists: ' + $dist) }
    New-Item -ItemType Directory -Path $dist -Force | Out-Null
    $bundleName = 'Tina-CLI-Windows-x64-' + $version
    $bundle = Join-Path $dist $bundleName
    New-Item -ItemType Directory -Path $bundle | Out-Null
    $files = & git -c core.quotepath=false ls-files
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate tracked files.' }
    foreach ($relative in $files) {
        $source = Join-Path $repoRoot $relative
        $target = [System.IO.Path]::GetFullPath((Join-Path $bundle $relative))
        if (-not $target.StartsWith($bundle + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Package path escaped output directory.' }
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $source -Destination $target
    }
    New-Item -ItemType Directory -Path (Join-Path $bundle 'bin') | Out-Null
    foreach ($binary in @('tina.exe','cq-tool-host.exe')) {
        Copy-Item -LiteralPath (Join-Path $binaryRoot $binary) -Destination (Join-Path $bundle ('bin/' + $binary))
    }
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText((Join-Path $bundle 'SOURCE_COMMIT.txt'), $revision + "`n", $utf8)
    $entries = Get-ChildItem -LiteralPath $bundle -Recurse -File | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($bundle.Length + 1).Replace('\','/')
        (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $relative
    }
    [System.IO.File]::WriteAllLines((Join-Path $bundle 'SHA256SUMS.txt'), [string[]]$entries, $utf8)
    & (Join-Path $bundle 'scripts/Verify-Integrity.ps1')
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory($bundle, (Join-Path $dist ($bundleName + '.zip')), [System.IO.Compression.CompressionLevel]::Optimal, $true)
    & tar -czf (Join-Path $dist ($bundleName + '.tar.gz')) -C $dist $bundleName
    if ($LASTEXITCODE -ne 0) { throw 'Windows tar archive failed.' }
    $sourceName = 'Tina-CLI-source-' + $version
    & git archive --format=zip ('--prefix=' + $sourceName + '/') ('--output=' + (Join-Path $dist ($sourceName + '.zip'))) HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Source zip archive failed.' }
    & git archive --format=tar.gz ('--prefix=' + $sourceName + '/') ('--output=' + (Join-Path $dist ($sourceName + '.tar.gz'))) HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Source tar archive failed.' }
    $archives = Get-ChildItem -LiteralPath $dist -File | Where-Object { $_.Name -match '\.(zip|tar\.gz)$' } | Sort-Object Name
    $sums = $archives | ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() + '  ' + $_.Name }
    [System.IO.File]::WriteAllLines((Join-Path $dist 'SHA256SUMS.txt'), [string[]]$sums, $utf8)
    Write-Host ('Packages: ' + $dist)
} finally { Pop-Location }
