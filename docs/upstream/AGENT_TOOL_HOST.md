# Tina Agent Tool Host v2

`cq-tool-host` is the host-side boundary for the Agent loop. It is a separate
Rust binary and has no CUDA, model, tokenizer, resident-weight, or KV-cache
dependency. The parent process may restart it without mutating model state.
The default claim is a workspace-bounded development preview with read-only
inspection, user-approved structured file lifecycle operations, and an
approval-gated `workspace_exec` validation tool. `workspace_exec` accepts only
direct-argv `cargo check/build/test/clippy/fmt/metadata` from a
workspace-relative cwd. It is not an OS sandbox: build scripts and tests
inherit the host environment and network, and every call requires exact
`EXECUTE` approval, including Full Access. An explicit `--full-access` host
scope widens file operations to machine paths and adds shell execution plus
unrestricted network access after the parent CLI's interactive `FULL_ACCESS`
warning. An optional explicit SearXNG JSON endpoint adds one bounded read-only
`web_search` capability without granting Full Access. MCP and subagents are
future plugin adapters only; they are not implemented, advertised, or part of
the default Agent schema.

## Start

```text
cargo run -p cq-tool-host -- --workspace C:\path\to\workspace
cargo run -p cq-tool-host -- --workspace C:\path\to\workspace --web-search-url https://search.example/search
cargo run -p cq-tool-host -- --workspace C:\path\to\workspace --full-access --shell powershell
```

The corresponding preview entry is explicit:

```text
cq-infer.exe --agent --chat --workspace C:\path\to\workspace --tool-host C:\path\to\cq-tool-host.exe
cq-infer.exe --agent --chat --workspace C:\path\to\workspace --web-search-url https://search.example/search
cq-infer.exe --agent --agent-full-access --chat --shell powershell --workspace C:\path\to\workspace --tool-host C:\path\to\cq-tool-host.exe
```

`--tool-host` may be omitted when the default sibling binary is present. The
client performs a handshake before the first tool call and rejects identity or
safety drift.

The process reads one JSON object per line from stdin and writes one response
object per line to stdout. Empty lines are ignored. The protocol is
`tina.tool-host.v2` and accepts version `2` only. A request carries an absolute
Unix-millisecond deadline no more than five minutes in the future:

Request shape:

```json
{"v":2,"kind":"request","id":"r1","op":"read","deadline_unix_ms":1786070000000,"args":{"path":"src/main.rs"}}
```

A patch request is structured rather than a free-form shell diff:

```json
{"v":2,"kind":"request","id":"p1","op":"apply_patch","deadline_unix_ms":1786070000000,"args":{"path":"src/main.rs","expected_sha256":"<whole-file-read-sha256>","edits":[{"old":"exact unique text","new":"replacement text"}]}}
```

Response shape:

```json
{"v":2,"id":"r1","ok":true,"result":{"kind":"text","text":"...","bytes":42,"sha256":"...","truncated":false,"artifact":null},"error":null}
```

The parent may send `{"v":2,"kind":"cancel","id":"r1"}` while a request
is active. The stdout reader remains independent of the main CLI thread. If a
real terminal receipt does not arrive during the 250 ms cancellation grace
period, the client terminates the complete Windows Job Object or Unix process
group and restarts a clean host before another Agent turn. Normal shutdown is
also bounded to two seconds. This makes Ctrl-C and deadlines reclaimable even
when a synchronous filesystem or Git call cannot observe the cancel frame
itself; it does not imply cooperative interruption inside every host syscall.

## Initial capabilities

- `capabilities`: returns the version and enabled capability list.
- `read`: reads a UTF-8 workspace-relative file, optionally by bounded
  1-based line range.
- `list`: lists a directory; recursive mode skips `.git`, `target`, `.tina`
  and symlinks.
- `search`: deterministic exact text search with bounded files and matches.
- `git_status`: read-only `git status --short --branch`.
- `git_diff`: read-only, no-color, no-external-diff output; `cached: true`
  selects the index diff.
- `apply_patch`: modifies one existing UTF-8 regular file through 1-128 exact,
  non-overlapping replacements. It requires the SHA-256 from a whole-file
  `read`, rejects missing or ambiguous old text, rechecks the file immediately
  before commit, fsyncs a same-directory temporary file, and atomically
  replaces the destination. It cannot create, empty, fully overwrite, delete,
  rename, or modify `.git`/`.tina` data.
- `write_file`, `mkdir`, `delete`, and `rename` are available in workspace
  scope with path containment, link/junction rejection, `.git`/`.tina` and
  workspace-root protection, bounded arguments, and terminal approval. In
  machine scope they additionally accept absolute paths. Directory deletion
  requires `recursive=true`; replacing a file can bind its current SHA-256.
- `workspace_exec` runs one direct-argv Cargo validation command from a
  workspace-relative cwd. Only `check`, `build`, `test`, `clippy`, `fmt`, and
  `metadata` are accepted; boundary-changing flags and parent/absolute paths
  are rejected. The parent CLI requires exact `EXECUTE` per call, and the
  terminal receipt discloses that this is not an OS sandbox.
- `web_search` appears only when `--web-search-url` names a valid HTTP(S)
  SearXNG JSON endpoint and a directly executable `curl` client passes the
  startup probe. The host invokes curl with direct argv, not a shell, and
  bounds protocols, redirects, connect/wall time, response bytes, query bytes,
  result count, titles, and snippets. It normalizes and deduplicates actual
  HTTP(S) result URLs and marks every receipt
  `untrusted_external_data=true`. Provider absence means the capability is not
  advertised; no default public provider, URL, summary, or citation is
  fabricated.
- Machine scope also adds `shell`. It runs the selected `--shell` dialect
  (`powershell`, `cmd`, `bash`, `zsh`, `fish`, or `sh`) with the host process's
  normal environment and network stack. Command size, wall time, and captured
  stdout/stderr are bounded; the command itself is otherwise unrestricted.
  Arbitrary shell commands can alter `.tina`, system files, and external
  services, which is why startup requires explicit `FULL_ACCESS`.

In bounded scope, absolute paths, parent components, NUL bytes, symlink targets, unsupported
operations, unknown request fields, oversized requests, and invalid arguments
are rejected before a tool runs. Output is capped at 64 KiB by default. A
truncated file/search response contains a content hash and a bounded artifact
snapshot under `.tina/tool-artifacts/`; the artifact path is workspace-relative.
`workspace_exec` returns a structured terminal receipt with bounded
stdout/stderr prefixes, byte counts, exit status, and explicit policy and
environment/network disclosures. Its bounded prefix may be marked truncated
without an artifact; that prefix is incomplete evidence and must not be
presented as the full command output.
Request frames default to 256 KiB. The client accepts at most 512 KiB for one
JSON response frame because JSON escaping can expand a 64 KiB text payload;
error messages are capped at 16 KiB. Hash, byte count, truncation state and
content-addressed artifact paths are checked again by the client; truncated
artifacts are reopened and rehashed before their receipt is accepted.

The handshake reports schema/version `tina.tool-host.v2`, the canonical
workspace anchor, `scope=workspace` or `scope=machine`, `writes=true`, a
scope-matched unrestricted `network` and `shell` flag, an independent
provider-backed `web_search` flag, the selected `shell_dialect`,
`cancellation=true`, a bounded maximum deadline, and an exact capability set.
Machine scope must include
`capabilities/read/list/search/git_status/git_diff/apply_patch/write_file/mkdir/delete/rename/workspace_exec/shell`;
workspace scope includes the same set except `shell` and advertises
`network=false` and `shell=false`. Either scope includes `web_search` only when
the optional provider probe succeeded. Any extra, duplicate or missing
capability fails closed. Model tool schemas are constructed only after this
handshake; the client keeps read-only, mutation, and validation-execution
dispatch methods separate.

The CLI, not model text, authorizes mutations or validation execution. In workspace scope,
`apply_patch`, `write_file`, `mkdir`, and `rename` show bounded target/content
summaries and accept only an exact terminal line `APPLY`; `delete` requires
exact `DELETE`. Non-interactive approval fails closed, and denial is archived
as a terminal tool result. In machine scope, startup requires exact
`FULL_ACCESS`; after that acknowledgement, regular file mutations and shell
commands are auto-authorized without per-call prompts except for structured
`delete`. `workspace_exec` remains a separate approval class and still
requires exact `EXECUTE` on every call. A shell command remains unrestricted
after startup, so `rm` or `Remove-Item` is intentionally outside that
structured-delete interceptor.

## Interactive safety and session controls

The inline Tina chat keeps the Agent boundary visible and local:

- `/status` reports the current engine identity, owner/epoch, context pressure,
  tool-host state, last-turn timing, and cache event. A global GPU prefix-cache
  rate is shown as `NOT_MEASURED` until the independent qualification receipt
  exists; a local protocol smoke cannot fill that field.
- `/tools` lists the exact read-only, mutation, and validation capabilities
  advertised by the v2 host, including whether its scope is workspace or
  machine. MCP stays disabled; shell and network state follow the selected
  scope.
- `/search` reports `AVAILABLE` or `NOT_AVAILABLE`; `/search <query>` executes
  the same bounded provider and displays the successful normalized receipt as
  terminal-escaped untrusted data. It does not inject a local result into the
  model archive. The Agent can independently call `web_search` through its
  advertised schema.
- `/diff` fetches Git status plus staged and unstaged diffs through the
  read-only host operations. The output is escaped before it reaches the
  terminal, so repository text cannot inject terminal control sequences.
- `/review [focus]` starts a read-only model turn with
  `read/list/search/git_status/git_diff` and, only when configured,
  `web_search`. The runtime repeats the denial for any mutating call; changing
  between review and normal turns closes the old owner before a new schema
  prefix is encoded.
- `/resume latest` or `/resume <full conversation id>` switches archives in the
  current process. Tina closes the current device owner first, validates the
  selected immutable archive, and rotates the resumed archive into a fresh
  epoch. No old KV owner, logits, or open tool transaction is reused.
- `/agent-restart` performs a replacement v2 handshake before retiring the old
  process. A failed handshake retains the old host. A successful replacement
  closes the current model owner, preserves the immutable conversation archive
  and terminal tool results, and creates a fresh owner on the next turn.

These commands are local controls, not model prompts. They work only in the
explicit `--agent --chat --workspace <root>` preview; ordinary chat remains
tool-free and the default `--kv-cache-mode off` path is unchanged.

The public source projection includes `crates/cq-tool-host`, and the Windows
bundle places `cq-tool-host.exe` beside `cq-infer.exe`. The product launcher
accepts `-Chat -Agent -Workspace <root>` and resolves that sibling by default;
source-development layouts may pass `-ToolHost <path>` explicitly.

`shell` is enabled only in explicit full-access mode. The host captures output
concurrently, limits the inline receipt, enforces a maximum wall time, and the
parent reclaims the complete tool-host process tree after cancellation or a
deadline. Full access intentionally has no environment allowlist or network
sandbox; `AgentBudgetLedger::account_tool_wall_ms` still accounts command wall
time against the turn budget.

The host is not allowed to obtain CUDA streams, the model tokenizer, resident
weights or a KV owner. Tool results are archived as host evidence; they are not
KV snapshots. A non-terminal tool transaction cannot be handed off, compressed,
resumed or prefixed into another model request.

The CPU-side turn/call accounting contract is
`docs/AGENT_CPU_TRACE_CONTRACT.md`. `AgentTraceLedger` records bounded,
monotonic Agent events and refuses a receipt while a turn or tool call is open.
Its prefill and decode fields are structurally `GPU_NOT_MEASURED`; this source
foundation does not turn the development preview into product Agent support.

## Archive contract

`cq-infer` archives `tool_invocation`, `approval`, and `tool_result` events in
the immutable conversation event log. Each event has a stable event ID and
record hash. Arguments and output are hashed independently. File/search
truncation carries a validated artifact reference; a structured
`workspace_exec` receipt may instead carry only a bounded prefix and
`output_truncated=true`, which is explicitly incomplete. Replay rejects
duplicate calls, results before invocation, hash drift, invalid artifact paths,
and sequence gaps.

These events are host evidence only. They are not KV snapshots and cannot be
used to restore a device owner. An open tool transaction must reach a terminal
result before context handoff, compression, resume, or another model prefill.

## Git project memory

Project memory is owned by `cq-infer`, not by `cq-tool-host`, and is a separate
durable plane from the conversation archive, tool-artifact store, and KV/prefix
cache. The interactive commands are `/project-memory`,
`/project-remember <fact>`, and `/project-forget <id>` (with Chinese aliases).
Only an explicit user fact is accepted. The implementation advances the
dedicated `refs/tina/project-memory` ref with Git plumbing; every commit tree
contains exactly `project-memory.json`, and an expected-old ref value makes the
update compare-and-swap. It never stages files or changes the current branch,
HEAD, index, worktree, `.tina` state, conversation archive, environment, shell
output, tool result, or KV owner.

Opening the ref validates the Git object type, the current commit's one-parent binding, tree contents,
project identity, entry/content/snapshot hashes, schema and sequence. The
identity excludes the project-memory ref itself and hashes repository history
and origin; a non-Git workspace is `unavailable` and has no file fallback.
Corruption, identity drift and concurrent mutation fail closed. Facts are
inserted into the model prompt only as bounded historical, untrusted data and
never as instructions. The independent default budget is 2,048 tokens (512
entries, 16 KiB per fact); credential/private-key-like content is rejected.
Forgetting shows the selected fact and reuses the exact terminal `DELETE` gate;
it appends a new snapshot while retaining the parent history. A successful
mutation closes the current model owner so the next turn re-encodes the new
prefix. This source feature does not qualify cache hit rate, TTFT, TPS, KV
capacity, or Agent performance.

## Verification

The current source checks include:

```text
cargo test -p cq-tool-host --locked
cargo test -p cq-generation --locked
cargo test -p cq-model-deepseek-v4-flash --locked
cargo test -p cq-infer --locked
```

The current Windows SM70 check records 22 `cq-tool-host` unit tests, three
binary JSONL process tests, and 155 `cq-infer` tests. Coverage includes the
workspace lifecycle, escape/control/link failures, explicit machine scope,
selected shell, bounded direct-argv cargo execution, optional search handshake
and normalized untrusted receipts, dynamic schema construction, approval
summaries, archive round-trip, and non-interactive denial.
These tests validate both opt-in scopes; they do not qualify a third-party
provider's availability, web-result correctness, Agent cache performance, or
GPU throughput.
