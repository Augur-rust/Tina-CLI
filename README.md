# Tina CLI

A Rust terminal agent with streaming chat, workspace tools and persistent conversation history.

[中文说明](README.zh-CN.md) · [User guide](docs/USAGE.md) · [Context contract](docs/CONTEXT.md) · [Build](docs/BUILDING.md)

**v0.1.0-alpha.2:** the terminal UI connects to a Chat Completions HTTP endpoint and the real tool host. Windows protocol, terminal and live DeepSeek integration acceptance have passed. This is product integration validation, not a model-quality benchmark or a local GPU performance qualification.

## Start on Windows

1. Start your model service. Tina connects to an existing Chat Completions-compatible `/v1` endpoint; it does not include model weights or an inference engine.
2. Extract the Windows package and double-click `Start-Tina.cmd`. First-run setup asks for the endpoint, model ID, workspace folder, credential-file reference and context/output budgets.
3. Type a request. Tina streams the response and can read workspace files or inspect Git. Editing, deletion and command execution require your explicit approval.

Use `Configure-Tina.cmd` to change the saved settings. To import an existing Augur endpoint and credential-file reference:

```powershell
.\bin\tina.exe configure --augur --workspace C:\work\my-project
.\bin\tina.exe chat
```

Augur must already be running. For another provider:

```powershell
.\bin\tina.exe configure --base-url http://127.0.0.1:8000/v1 --model my-model --workspace C:\work\my-project --api-key-env TINA_API_KEY
.\bin\tina.exe chat
```

Set the named environment variable in your own session when authentication is required, or use `--api-key-file C:\private\model-key.txt`. Tina saves the reference, not the secret value. Keep `tina.exe` and `cq-tool-host.exe` together in `bin`. Git is required for Git operations and project memory; Python and a GPU are not required to run Tina itself.

### DeepSeek quick start

With `DEEPSEEK_API_KEY` already set in your environment:

```powershell
.\bin\tina.exe configure --base-url https://api.deepseek.com/v1 --model deepseek-flash --api-key-env DEEPSEEK_API_KEY --thinking low --workspace C:\work\my-project
.\bin\tina.exe chat
```

The official DeepSeek endpoint uses its native thinking controls. `off` disables thinking, `low` selects low effort, `medium`/`high` select high, and Tina's `xhigh` selects DeepSeek `max`. Provider-returned reasoning remains separate from the final answer and is preserved through tool continuation, archive recovery and retained handoff messages.

## Interactive features

- Streaming reasoning/content and bounded, incrementally parsed tool calls.
- Workspace reads, search, Git inspection, approved file edits and bounded Cargo execution through a separate tool process.
- Request-specific confirmations: `APPLY` for writes, `DELETE` for deletion, `EXECUTE` for execution. Piped input cannot approve these operations.
- Ctrl-C cancellation during generation or tool execution; completed effects keep their receipts. Interrupted output is not archived as a complete answer.
- `/new`, `/sessions`, `/resume`, `/retry`, `/compact`, `/status` and `/context`.
- `/thinking off|low|medium|high|xhigh`, goals, todos, pins, project memory, exports, diff and read-only review.
- Immutable history and automatic handoff under context pressure, with the latest complete turn retained.

The UI is a terminal interface, not a web application. MCP servers, subagent execution and an unrestricted shell tool are not implemented in this release. Provider support for reasoning fields and tool calls varies; unsupported combinations need a compatible endpoint. Use `--chat-only --thinking off` for a minimal chat configuration.

## Context and measurements

The default configuration is 262,144 positions with 4,096 reserved output tokens: **258,048 usable input**, with the 95% handoff threshold at **245,145**. Set the context to your actual endpoint's capacity. Configuring a larger number does not enlarge the model server.

Live admission starts with conservative UTF-8/JSON byte accounting, clearly labelled **estimated**. Reported prompt usage can anchor the next estimate only if the previous message prefix and tool definitions are unchanged. This is not a local model tokenizer. Handoffs preserve full archived history; a single oversized input or the retained recent turn may still exceed the usable budget.

The UI reports provider usage and **HTTP round throughput**. Missing usage is shown as unavailable. Tina does not measure native decode/prefill TPS, DFlash acceptance or remote KV-cache occupancy.

## Verification and development

Windows checks cover **187 Rust unit/process tests**, **9 executable HTTP fixture cases**, exact context-admission boundaries, formatting and Clippy. Terminal tests exercise real `APPLY`, file creation, Ctrl-C socket closure and a subsequent fresh session. Fixture responses are deterministic protocol test data. Separately, **7 scripted and 4 interactive real DeepSeek cases** passed: English/Chinese answers, arithmetic-code validation, thinking-mode tool continuation, resume, handoff, approved editing, mode switching and cancellation followed by a successful request. See [verification results](docs/verification/summary.json).

```powershell
.\scripts\Verify.ps1
cargo build --workspace --release --locked
```

Building requires the pinned Rust 1.96.0 toolchain and MSVC C++ tools. Verification additionally uses Python 3.11+ (standard library only). See [BUILDING](docs/BUILDING.md) for packaging and offline builds.

## Repository layout

| Path | Purpose |
| --- | --- |
| `crates/tina-harness` | Live CLI, configuration, HTTP transport, context, memory and terminal UI |
| `crates/tina-protocol` | DSML encoding, tool parsing and strict JSON |
| `crates/cq-tool-host` | Separate workspace tool process |
| `crates/cq-retrieval` | Retrieval primitives; optional USEARCH backend |
| `scripts`, `fixtures`, `demo` | Verification, packaging and CPU-only replay |
| `integration` | Historical engine integration outside the standalone build |
| `provenance` | Original revisions, hashes and standalone adaptations |
| `third-party` | Locked dependency inventory and license copies |

## License and provenance

MIT OR Apache-2.0. Extracted from Crystal Quant revision `10a16cb2bcd65377063912bfcbc9cbded034b220`, with separately recorded context fixes and standalone adaptations. Original attribution is preserved. Model weights and private artwork source files are not included.

See [NOTICE](NOTICE), [third-party notices](THIRD_PARTY_NOTICES.md), [security boundaries](SECURITY.md), [contributing](CONTRIBUTING.md) and [changelog](CHANGELOG.md).
