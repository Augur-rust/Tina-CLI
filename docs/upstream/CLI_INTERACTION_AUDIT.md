# Tina CLI Interaction Audit

Date: `2026-08-10`

This audit covers the terminal product surface in `crates/cq-infer`, not the
CUDA qualification runners. The CLI must remain a presentation and session
control layer; it must not invent GPU cache claims or become a second KV owner.

## Closed In This Slice

- `/context` and `/status` show the live owner cursor, the calibrated
  `258,048` usable input positions, the `95%` handoff threshold, output
  reserve, and pressure state. GPU prefix-cache hit rate is shown as
  `NOT_MEASURED` until a product qualification supplies it.
- `/new`, `/reset`, and `/clear` finish the current owner and create a fresh
  archive. The old archive remains on disk and the old KV owner is never
  reused.
- `/compact` creates an immutable archive handoff, retaining the configured
  recent complete turns. A live owner is finished only after the handoff is
  committed.
- Ctrl-C during interactive decode is sampled at token boundaries. The owner
  is drained/cancelled, no partial assistant turn is published, and a fresh
  archive is opened. This is intentionally a reset because the archive is
  append-only.
- `/export [jsonl|md] <path>` exports the active branch as an importable
  one-line JSONL document or a readable Markdown transcript. Relative paths
  resolve against the process directory; the parent must already exist,
  destination symlinks are rejected, publication uses a same-directory
  temporary file, and an existing file requires exact `OVERWRITE`.
- Agent model generation uses the same token-boundary Ctrl-C guard. A
  cancellation closes the authoritative owner, publishes no partial assistant
  turn, and rotates to a fresh archive just like normal chat.
- Mutating approvals (`APPLY` and `DELETE`) fail closed when
  stdin/stdout are not terminals; a piped chat can never consume its next
  prompt as an authorization line.
- Startup argument validation now rejects unknown flags, duplicate options,
  missing values, positional arguments, and the unsupported `--name=value`
  spelling before artifact or CUDA initialization. `--cache-smoke` is an
  exclusive command rather than a partially parsed mode.
- Chat mode now has a bounded 64-token default output budget; one-shot prompt
  and qualification paths retain their historical one-token default unless
  the caller explicitly supplies `--max-new-tokens`.
- Interactive chat and structured `--messages-json` requests default to the
  official DSV4 `chat` mode (`thinking=off`). `--thinking-mode thinking` and
  `/thinking medium|high|max` are explicit opt-ins, so the fast product path
  does not silently add reasoning latency or change request identity.
- The startup surface does not repeat a fixed command strip. Typing `/` opens
  the complete command popup, while `/help` remains the durable command
  reference. `/agent` is status-only and cannot mutate access policy or
  rebuild the model owner.
- `/goal set|clear` appends a bounded, hashed immutable archive event and is
  injected as explicitly user-defined context. A successful goal or thinking
  change closes the old owner before the next request. Explicit thinking modes
  buffer internal reasoning and publish/archive only the final answer text.
- `--help` documents the required artifact/placement inputs and the explicit
  Agent preview boundary, so a copied command does not fail later with an
  opaque placement error.
- Agent tool wall-time is now measured around each host execution and charged
  to `AgentBudgetLedger`; an over-budget turn receives terminal denied results
  for remaining calls and closes without executing another tool.
- A `//` prefix escapes slash-command parsing for literal paths and shell-like
  text, and assistant output escapes non-layout terminal control characters so
  model text cannot emit raw ANSI clear-screen or cursor-control sequences.
- Chinese aliases now cover context/session controls, memory/pins, TodoList,
  and plugin listing. UI labels remain presentation-only; canonical model
  roles stay `user`/`assistant`.
- `--resume-conversation latest` and `/sessions` use validated archive
  activity rather than filesystem mtime. Full IDs, activity, epoch, turn and
  Agent-event counts are discoverable without guessing a hash.
- `/retry` writes an immutable supersession/fork record, cancels unfinished
  tool transactions, rotates to a new epoch, and submits the source user turn
  again without copying the superseded assistant/tool context.
- Tool-host protocol v2 carries deadlines and cancel frames. The client reads
  stdout on a dedicated thread and reclaims the complete Windows Job Object or
  Unix process group when a terminal receipt does not arrive in the bounded
  grace period.
- Default Agent mode now supports the complete structured workspace lifecycle:
  `apply_patch`, `write_file`, `mkdir`, `rename`, and `delete`. Every operation
  is preflighted and confirmed; root, `.git`, `.tina`, parent escape, and
  link/junction paths fail closed. Full Access remains the only route to
  machine paths and shell execution.
- `--web-search-url` and launcher `-WebSearchUrl` provide an explicit SearXNG
  JSON endpoint. The tool host probes curl and advertises `web_search` only
  after a successful handshake. `/search` shows provider status and
  `/search <query>` displays the actual bounded receipt. External text is
  terminal-escaped and marked untrusted; provider absence or failure never
  produces a URL or citation.
- Agent tool schemas are now derived after the host capability handshake.
  Normal and review turns cannot request unavailable operations; review remains
  read-only and includes web search only when that provider is real.
- The Agent system contract is outcome-first and receipt-first. Change/fix
  requests must inspect, act, and verify rather than merely propose commands;
  read-only requests do not authorize mutation. Complex work uses the internal
  understanding, restatement, decomposition, principles, self-check,
  comparison, deepening, synthesis, and risk loop without printing private
  chain-of-thought.
- Every interactive model request receives a system message containing the
  detected OS, shell dialect, terminal host, process cwd and Agent workspace.
  `--shell`/`TINA_SHELL` provide an explicit override, and `/environment`
  exposes the exact context injected into the model.
- Composer Ctrl-C clears a nonempty draft before a second empty-draft Ctrl-C
  exits. Slash-prefix Tab completion does not consume normal prompt Tab input,
  and an oversized bracketed paste is rejected with a localized warning.
- Slash-command entry now uses a Codex-style unframed popup: exact and prefix
  matches are filtered as the command token changes, descriptions are dimmed,
  the selected row is highlighted, and at most eight rows are visible. Up/Down
  wraps selection, while Tab completes and Enter runs the selected command.
  Chinese aliases use the same path and ordinary prompt/history navigation is
  unchanged when the popup is absent.
- The assistant ID and streamed answer text use the same bright moon-white
  color as the startup frame. Every streamed piece resets its terminal style,
  so tool output, notices, and the next composer cannot inherit that color.
- The approved Demo interaction surface is now wired into the product path:
  `User` composer and transcript frames, bright moon-white `Tina` response
  frames, fixed-width non-flashing tide reflection, and boxed `edit`/`write`
  previews. The model/tool/archive protocol remains unchanged.
- `APPLY` approval now shows a bounded escaped boxed preview of exact
  replacements before accepting the authorization token. `TINA_NO_MOTION=1`
  disables the optional tide reflection without changing the transcript.
- Chinese presentation now covers conversation open/resume, DeepSeek import,
  automatic/manual handoff, pinned memory, TodoList mutations, declarative
  plugin notices, Agent tool completion, and tool-wall termination. Canonical
  archive reasons, tool names, status values, and machine metrics are unchanged.
- `cq-infer --doctor` is an exclusive CPU-only startup path. It validates the
  admitted model manifest/shard headers/auxiliary files, artifact-bound
  placement, host paths, archive parent, architecture, shell and terminal
  before CUDA initialization. GPU/VRAM remain explicitly `NOT_CHECKED`.

## Startup surface design (implemented)

The first interactive screen is intentionally a compact orientation surface,
not a command catalog. It renders, in order:

1. one moon-white frame headed by `TINA ENGINE · <assistant-id>` and the active
   `FAST` mode;
2. the fixed product identity (`0731-main`, single sequence, no-MTP/no-D-Spark);
3. a trusted short environment label, the usable `258K` context budget, KV
   opt-in state, and the default `thinking off` setting;
4. one input hint. Commands remain discoverable through the `/` popup and
   `/help`; details stay behind `/status`, `/environment`, and the other
   existing commands.

The frame is width-safe for the existing 64-112 column terminal range and uses
only trusted, normalized OS/shell labels. The full environment injected into
the model remains available from `/environment`; the startup frame is not a
second source of runtime truth. `scripts/tina-cli-demo.ps1` remains a
model-free, CUDA-free, non-mutating preview, while the product CLI now shares
the same visual contract on its real model/tool path.

## Remaining Priority Work

### P1: complete localization and discoverability

Low-level archive/storage and context-packing failures still retain canonical
English details after their localized presentation wrapper. Add stable error
categories before translating those details; do not infer categories by
matching error strings. Progress for long prefill/tool calls remains
intentionally absent; a future rate-limited status line must first prove it
does not starve the CUDA submit lane.

### P2: optional media and plugin execution

Image/OCR (`zh`/`en`/`ja`) should remain an optional cancellable CPU
worker/plugin; it must never enter the per-token CUDA path or write directly to
KV. A future `/attach` command must report extracted text provenance, source
hash, OCR model identity, coordinates, confidence and token cost before
submission. Plugin entry processes remain declarative/disabled until they gain
the same deadline, cancellation, bounded-output and approval contracts as the
tool host.

### P3: product polish after safety gates

Optional persisted command history, richer parameter hints, and `/stop`
progress rendering remain product polish.
History must be opt-in with a clear privacy boundary. None may add a background
worker that competes with the CUDA submission lane or alter the model/KV owner.

## Qualification Boundaries

The CLI may display policy values and CPU protocol probes, but it cannot claim
GPU payload restore, warm TTFT, or 32K/128K/256K hit ratios until CACHE-4 (or a
later identity) supplies paired product evidence. Any future cache telemetry
must preserve the existing hit/miss/failure and publication conservation
equations and remain independent from UI formatting.
