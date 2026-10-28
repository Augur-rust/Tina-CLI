# Changelog

## v0.1.0-alpha.1 — 2026-10-28

First public standalone Tina CLI release.

- Rust CPU harness, independent tool host, protocol helpers and optional retrieval workspace.
- Structured context admission from upstream cb2bcda3, plus agent-only host-context behavior adapted from 3a218bdd.
- Preserve output reservations for small contexts; remove the replay adapter's stale 1 MiB prompt ceiling in favor of the bounded 32 MiB document guard.
- `context-check` command with machine-readable rejection details and deterministic exit status.
- Reproducible Windows packages, source archives, checksums, source provenance and license inventory.

Live LLM/API chat and a newly validated GPU integration are not included in this release.
