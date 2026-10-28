# Changelog

## Unreleased

- Fix a Windows CI race in the loopback web-search test: explicitly restore blocking mode after accepting a connection from the nonblocking listener. Production runtime behavior is unchanged; published alpha.2 archives remain immutable.

## v0.1.0-alpha.2 — 2026-10-28

- Connect the terminal UI to streaming Chat Completions and the real workspace tool host.
- Add first-run setup, Augur config import, credential references and Windows launchers.
- Wire explicit edit/execute approvals, socket cancellation, sessions, resume, context handoff and read-only review.
- Bound stream parsing and reject truncated or malformed tool completions before execution.
- Reuse server token usage only after checking unchanged message prefixes and tool definitions; label estimates and HTTP throughput explicitly.
- Add executable protocol tests, terminal acceptance evidence and updated third-party notices.
- Preserve provider reasoning in hash-protected assistant archive records and restore it for DeepSeek tool continuations, process recovery and retained handoff turns. Legacy archive hashes remain readable.
- Use native DeepSeek thinking controls: off disables reasoning, low/high select effort, and Tina xhigh maps to max.
- Pass 187 Rust tests, 9 HTTP protocol fixtures, and 7 scripted plus 4 interactive real DeepSeek acceptance cases. These are integration checks, not capability benchmarks.

## v0.1.0-alpha.1 — 2026-10-28

First public standalone Tina CLI release.

- Rust CPU harness, independent tool host, protocol helpers and optional retrieval workspace.
- Structured context admission from upstream cb2bcda3, plus agent-only host-context behavior adapted from 3a218bdd.
- Preserve output reservations for small contexts; remove the replay adapter's stale 1 MiB prompt ceiling in favor of the bounded 32 MiB document guard.
- `context-check` command with machine-readable rejection details and deterministic exit status.
- Reproducible Windows packages, source archives, checksums, source provenance and license inventory.

Live LLM/API chat and a newly validated GPU integration are not included in this release.
