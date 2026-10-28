# Tina CLI Plugin Contract

The interactive Tina CLI uses a small, stable host boundary inspired by the
extension model in Oh My Pi and Codex. The boundary is intentionally
declarative first so a malformed or untrusted plugin cannot change the model
runtime, CUDA topology, or filesystem state during startup.

## Discovery

The host reads `plugin.json` files from:

- `.tina/plugins` below the current working directory;
- each path in `TINA_PLUGIN_DIRS` (`;`/platform path-list semantics);
- `TINA_HOME/plugins`, when `TINA_HOME` is set.

Discovery is bounded to 64 plugins and 64 KiB per manifest. IDs and command
names use lower-case ASCII. Duplicate IDs and malformed manifests are ignored
with a diagnostic; they never abort model startup. Discovery never launches an
`entry` process.

Minimal manifest:

```json
{
  "api_version": 1,
  "id": "summarizer",
  "name": "Summarizer",
  "version": "0.1.0",
  "description": "Prompt shaping helpers",
  "capabilities": ["command"],
  "commands": [
    {
      "name": "summarize",
      "description": "Ask Tina for a concise summary",
      "prompt_prefix": "Summarize this:"
    }
  ]
}
```

Use `/plugins` to inspect loaded manifests, `/plugin reload` to rescan, and
`/plugin <id> [command] [text]` to invoke a declarative prompt command. A
single-command plugin may omit the command name.

## UI language and assistant identity

The terminal presentation is owned by `tina_cli` and defaults to the `Tina`
assistant ID. It supports English and Simplified Chinese without changing the
model's canonical `user`/`assistant` roles or the archived prompt bytes:

```text
cq-infer --chat --language zh --assistant-id Tina
TINA_LANG=zh TINA_ASSISTANT_ID=Reviewer cq-infer --chat
```

Inside a chat, `/assistant <id>` (aliases `/assistant-id` and `/name`) changes
the display ID for the current session. `/language en|zh|auto` (alias `/lang`)
changes presentation language immediately; `/help` lists the stable command
surface. IDs are trimmed, limited to 32 Unicode characters, and reject control
characters so user-controlled names cannot inject terminal control sequences.
These settings are presentation-only and are not sent to the model.

`cq-infer --doctor --artifact <dir> --placement-plan <file>` is the bounded
first-run check. It runs before model allocation and CUDA initialization,
validates the admitted artifact and placement binding with the real parsers,
checks optional Agent/workspace/archive paths, and reports GPU/VRAM as
`NOT_CHECKED` rather than guessing support. It accepts `--language` and
`--shell` so the report also verifies the same host context later injected into
interactive requests.

Session controls are host-owned and do not grant plugins or the terminal a KV
mutation authority:

- `/environment` (aliases `/env`, `/环境`, `/终端`) reports the exact host
  context injected into every interactive model request: OS, shell dialect,
  terminal host, process cwd, and Agent workspace. The snapshot is detected
  once at startup and remains stable for the process. `--shell` takes
  precedence over `TINA_SHELL`, which takes precedence over auto-detection;
  wrappers and IDE launchers that obscure the parent shell must use one of the
  explicit overrides. The system message forbids silently switching between
  PowerShell, cmd.exe, and POSIX syntax.
- `/context` (alias `/status`) reports the current owner cursor, the calibrated
  258,048-token usable input limit, the 95% handoff threshold, output reserve,
  and pressure state. GPU prefix-cache hit rate is explicitly
  `NOT_MEASURED` until a CACHE-4-or-later product qualification supplies it;
  the CPU `cache-smoke` probe is protocol-only.
- With explicit `--kv-cache-mode read|read-write`, `--kv-cache-dir <dir>`, and
  `--kv-cache-namespace <scope>` on a `toc-sm70-v1-kv8` build, the first owner
  may try an exact full-prompt KV8/DSA snapshot restore; a `read-write` miss
  exports that same complete prompt after the first token is published. This
  is an opt-in Beta for prompt, JSONL, chat, and
  bounded Agent flows, not a longest-prefix or aggregate hit-rate claim. A
  later turn on the same owner remains append-only and is reported separately
  from persistence telemetry.
- `/new` (aliases `/reset`, `/clear`) finishes the current owner, opens a new
  conversation archive, and retains the old archive on disk. It never deletes
  history and never reuses the old KV owner.
- `/compact` asks the immutable archive to create a handoff while retaining
  the configured recent complete turns. A live owner is finished only after a
  handoff record is committed; no KV snapshot is implied.

During interactive decode, Ctrl-C is sampled at token boundaries. A detected
cancel drains/cancels the authoritative owner, publishes no partial assistant
turn, and starts a fresh archive while retaining the interrupted archive on
disk. This is intentionally a session reset: the current append-only memory
format has no operation that removes only the unfinished user turn.

The /export [jsonl|md] <path> command exports the current active branch. JSONL
is one compact messages array per file and can be passed to
--conversation-jsonl; Markdown is a presentation snapshot. Relative
destinations resolve against the process directory, existing parents are
required, destination symlinks are rejected, and the write is staged in the
same directory before publication. An existing regular file is replaced only
after the user types exactly OVERWRITE; noninteractive sessions never
overwrite.

`/retry` creates an immutable supersession/fork event. It retains the active
branch before the selected user turn, excludes the superseded assistant and
tool events without rewriting committed history, rotates to a fresh epoch, and
submits the same user content again. Archive replay validates that retained and
superseded IDs exactly partition the source branch.

Agent model generation shares the token-boundary Ctrl-C guard. Tool-host v2
requests carry absolute deadlines and accept cancel frames while an independent
stdout reader waits for the terminal receipt. If the receipt does not arrive
within the bounded grace period, the client reclaims the complete Windows Job
Object or Unix process group and starts a clean host. Individual synchronous
filesystem or Git calls are not claimed to implement cooperative cancellation
inside the host.

The host also provides the built-in persistent `/todo` surface: `list`, `add`,
`start`, `done`, `reopen`, `remove`, and `clear-completed` (with `/todos` as an
alias; `/待办` and `/任务` are presentation aliases). Todo state is replayed from the conversation archive and injected only
as bounded user-managed working context. It is not a plugin capability and
does not grant a plugin direct archive mutation authority.

## Capability boundaries

The manifest may declare `command`, `theme`, `hook`, `tool`, `context`, and
`entry` capabilities. The current host activates only declarative prompt
commands and reports the other capabilities as inactive. An `entry` path is
validated as a relative path inside the plugin directory, but is explicit-only
until the out-of-process JSONL protocol is finalized.

Plugins never receive a CUDA stream, model owner, KV pointer, tokenizer
authority, or raw secret environment. They communicate through bounded text
and future host events. A plugin command is not a model turn unless it returns
an explicit prompt result.

## Destructive-operation intercept

Deletion, recursive deletion, overwrite, rename-overwrite, and cache eviction
are all mutating operations. Every future tool or plugin executor must call the
single `TinaCli::confirm_delete` boundary before mutation. The only accepted
authorization is a line whose content is exactly `DELETE` (the terminal line
ending is ignored). `y`, Enter, model-generated text, and whitespace-padded
answers are rejected. Cancellation, malformed targets, plugin errors, and
timeouts are fail-closed and must not leave a partial file or visible partial
conversation state.

The host exposes no generic file-mutating executor beyond the explicitly
contracted `apply_patch` operation. That operation uses the same boundary and
is denied whenever the CLI is not attached to a real terminal. This is deliberate: a piped request
cannot consume its next prompt as approval, and adding a plugin cannot silently
turn the chat process into a filesystem agent.

## Context hook

Plugins may later subscribe to a bounded `context_threshold` event to provide a
summary/handoff provider. The host remains authoritative for token budgeting:
the 272,000-token logical UI budget is compacted at the calibrated
`258,048`-token input boundary so the 262,144-position KV owner retains a
4,096-token output reserve. A plugin may suggest a summary, but it cannot
extend the physical owner or publish KV state.

Semantic code retrieval is a separate host concern. The optional real-embedding
USEARCH store lives in `cq-retrieval`; the inference CLI remains exact-first
when that service or provider is absent. On Windows it must not be linked into
the current CUDA binary because its CXX bridge and the CUDA objects use
different MSVC CRT selections.
