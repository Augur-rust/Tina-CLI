# Building and running

Use the pinned Rust 1.96.0 toolchain and committed Cargo.lock. Git must be on PATH for Git tools and demo workspaces. Windows builds require the MSVC C++ build tools.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Verify.ps1
cargo build --workspace --release --locked
.\target\release\tina.exe doctor --workspace .
.\target\release\tina.exe context-check --input-tokens 258048
.\target\release\tina.exe replay --workspace . --scenario fixtures/read-and-status.json --archive C:\path\to\new-archive
```

The supplied fixture expects `sample.txt`; use `scripts/Run-Demo.ps1` from a packaged release to create the example workspace automatically. Keep `tina.exe` and `cq-tool-host.exe` in the same directory, or pass `--tool-host` explicitly for doctor/replay.

For an offline build after dependencies are cached, pass `-Offline` to Verify.ps1. USEARCH is optional and exercised by the all-features tests. Release binaries use default features and do not bundle an embedding model.

To build release archives and their checksums:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Package.ps1
```

Output goes under `dist/`. Packaging uses committed files (`git ls-files`), so commit source changes first. It excludes local session history, secrets, target directories and development logs. The Windows bundle contains the source plus the two executables. `scripts/Verify-Integrity.ps1` verifies its file checksums.
