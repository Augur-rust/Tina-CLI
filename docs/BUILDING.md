# Building, testing and packaging

Use the pinned Rust 1.96.0 toolchain and committed Cargo.lock. Windows builds require MSVC C++ build tools. Git must be on PATH. Verification scripts require Python 3.11+ using only its standard library; the two runtime executables do not depend on Python.

```powershell
.\scripts\Verify.ps1
cargo build --workspace --release --locked
.\target\release\tina.exe chat --workspace C:\work\my-project
```

`Verify.ps1` builds both executables before running tests: some tests launch the real sibling `cq-tool-host.exe`. It checks caller-supplied context boundaries, all-feature Rust tests, executable HTTP fixtures, rustfmt and Clippy with warnings denied. Use `-Offline` after dependencies are cached.

USEARCH is optional and exercised in all-feature tests. Release binaries use default features; no embedding model is bundled. `Run-Demo.cmd` still runs a CPU-only read/Git replay in a fresh temporary workspace. Real model quality is evaluated separately from deterministic transport fixtures.

## License inventory

After changing dependencies:

```powershell
python scripts/update_notices.py
```

The script uses the locked Windows all-feature dependency graph and preserves the recorded upstream USearch notice. Review newly added licenses and keep the inventory, lockfile and notices together. It does not export Cargo registry paths or credentials.

## Windows archives

After verification, commit the intended source changes and run:

```powershell
.\scripts\Package.ps1
```

Output goes under `dist/vVERSION-COMMIT/`. Packaging requires a clean checkout and uses committed tracked files. The Windows ZIP and TAR.GZ include source, documentation, both executables, `Start-Tina.cmd`, `Configure-Tina.cmd`, `Run-Demo.cmd`, a source commit marker and per-file hashes. Separate source ZIP and TAR.GZ archives and outer `SHA256SUMS.txt` are produced. `scripts/Verify-Integrity.ps1` checks the extracted bundle.

Run `python scripts/test_live_http.py --binary <bundle>\bin\tina.exe` against the extracted release too. Test approved writes and cancellation in an actual interactive terminal; redirected protocol tests alone do not exercise raw-mode input. Record real-model checks separately before promoting a candidate release. A local package does not itself publish a Git tag or GitHub release.

Windows x86_64 is the qualified build platform. There is no Linux binary package in this release.

## Optional real API acceptance

With `DEEPSEEK_API_KEY` already set, run:

```powershell
python scripts/test_deepseek_live.py --run-live --binary target/release/tina.exe --evidence-root .validation/deepseek-live-new
```

Use a new evidence directory. This opt-in command consumes paid API tokens and creates a disposable synthetic Git workspace. It checks English/Chinese responses, a read-tool continuation with reasoning, process resume, arithmetic-only generated Python, and handoff/resume. Raw responses and archives remain under the ignored evidence directory; its `report.json` contains only checks and metrics. CI does not invoke live API tests. Interactive edit approval and cancellation are checked separately in a terminal.
