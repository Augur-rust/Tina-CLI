# Tina CLI

A Rust terminal agent harness with explicit context budgets, persistent conversation history, and a separate tool-host process.

[中文说明](README.zh-CN.md) · [Context management](docs/CONTEXT.md) · [Build and run](docs/BUILDING.md) · [Security boundaries](SECURITY.md)

## What you can run

This first standalone release provides **CPU diagnostics, prompt export, context admission checks, and scripted agent-loop replay**. Tool calls, filesystem operations and conversation archives in replay are real; the model completions are fixtures. It needs neither a GPU nor model weights.

The original interactive terminal UI and model integration are included as source. **This standalone binary does not yet connect to an LLM API or provide live chat.** MCP and subagent execution are not implemented.

## Quick start on Windows

Download and extract the Windows archive from Releases. Install Git if it is not already available, then double-click `Run-Demo.cmd`. The demo creates a new temporary Git workspace, reads a sample file, obtains Git status and preserves the resulting receipts and archive. It does not modify your existing project.

```powershell
.\bin\tina.exe --help
.\bin\tina.exe doctor --workspace .
.\bin\tina.exe prompt --workspace . --request 'Inspect the project and explain the next step'
.\bin\tina.exe context-check --input-tokens 258048
```

`context-check` uses a **caller-supplied token count**; it does not tokenize text. Rejected input produces structured JSON and a nonzero exit code.

## Features

- Explicit context pressure levels: normal, prepare, freeze, handoff and hard limit.
- Output-token reservation and structured capacity rejection, including exact boundary checks.
- Append-only conversation archives, immutable handoffs, retry/resume support and project memory modules.
- An independent JSONL tool host for workspace reads, Git inspection and permission-checked mutations.
- Receipt-first execution: completed tool effects retain their real terminal result even when the remaining budget is exhausted.
- Bilingual terminal controls and Sixel/text splash modules, with a separate model-free UI demonstration.
- Optional USEARCH-backed retrieval; index tests use fixed vectors, not a production embedding provider.

The standalone replay entry point denies writes, deletion and command execution. Its scenario JSON cannot grant approval. The original interactive integration contains the user approval flow.

## Context fixes in this release

The source incorporates the structured admission work from upstream `cb2bcda3` and adapts the agent-only execution-context correction from `3a218bdd`. The replay adapter's obsolete 1 MiB prompt cap is aligned with the 32 MiB document guard. Byte limits are distinct from model token capacity.

For the calibrated 262,144-position configuration, 4,096 positions are reserved for output, leaving **258,048 input tokens**; the 95% handoff threshold is **245,145**. A larger output request reduces the usable input budget. These are tested CPU admission rules, not evidence of a standalone 256K model run. See [the exact contract](docs/CONTEXT.md).

## Build and verify

Requires Rust 1.96.0, Git and a native linker. On Windows install Visual Studio Build Tools with Desktop development with C++; the optional USEARCH feature also compiles C++.

```powershell
cargo build --workspace --locked
cargo test --workspace --all-features --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo build --workspace --release --locked
```

Build before testing: the harness integration tests launch the sibling `cq-tool-host` executable. `scripts/Verify.ps1` performs the sequence. Windows x86_64 is the locally verified release platform; other platforms are not qualified by this release.

## Verification

The Windows release passes **174 unit/process tests**, **9 release-binary context admission cases**, formatting, Clippy with warnings denied, and a real tool-process demo. [Saved results and command output](docs/verification/summary.json) describe the tested scope. No live-model or GPU quality result is claimed.

## Layout

| Path | Purpose |
| --- | --- |
| `crates/tina-harness` | Context, history, terminal UI, prompts and standalone CLI |
| `crates/tina-protocol` | DSML encoding, tool-call parsing and strict JSON |
| `crates/cq-tool-host` | Independent workspace tool process |
| `crates/cq-retrieval` | Retrieval primitives and optional USEARCH backend |
| `fixtures`, `demo` | Scripted loop fixture and terminal UI demonstration |
| `integration` | Historical model integration and upstream repair patches; outside the standalone build |
| `provenance` | Source revisions, hashes and adaptation records |
| `third-party` | Dependency inventory and license copies |

## License and provenance

MIT OR Apache-2.0. Tina was extracted from Crystal Quant, base revision `10a16cb2bcd65377063912bfcbc9cbded034b220`, with separately recorded context fixes and standalone adaptations. Upstream attribution is preserved. Model weights and private artwork source files are not included.

See [NOTICE](NOTICE), [third-party notices](THIRD_PARTY_NOTICES.md), [contributing](CONTRIBUTING.md) and [the changelog](CHANGELOG.md).
