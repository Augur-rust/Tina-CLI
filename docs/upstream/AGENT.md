# Agent Access Modes

The public source projection includes a local Agent preview. It is not a
remote service. The default mode is workspace-bounded; explicit Full Access
adds machine-wide files, unrestricted shell execution, and network access.
The default mode also exposes the approval-gated `workspace_exec` validation
tool, which accepts only direct-argv Cargo validation commands from a
workspace-relative cwd. It is not an OS sandbox: build scripts and tests
inherit host permissions and network. Every `workspace_exec` call requires an
exact `EXECUTE` confirmation, including Full Access. MCP and subagents are
future plugin adapters only and are not implemented in the default CLI.

`cq-infer` owns the model, conversation archive, approvals, and GPU state.
`cq-tool-host` is a separate CPU-only process that owns filesystem and Git
operations. A Windows bundle places both executables under `bin/`.

The default Agent mode is workspace-bounded:

```powershell
.\scripts\run-toc-sm70.ps1 `
  -Artifact 'F:\path\to\DeepSeek-V4-Flash-0731-CQTensors' `
  -PlacementPlan 'F:\path\to\placement.json' `
  -Chat -Agent -Workspace 'C:\path\to\project'
```

An optional read-only web-search provider can be added without Full Access:

```powershell
.\scripts\run-toc-sm70.ps1 `
  -Artifact 'F:\path\to\DeepSeek-V4-Flash-0731-CQTensors' `
  -PlacementPlan 'F:\path\to\placement.json' `
  -Chat -Agent -Workspace 'C:\path\to\project' `
  -WebSearchUrl 'https://search.example/search'
```

For local development, a Codex-style Full Access mode is available:

```powershell
.\scripts\run-toc-sm70.ps1 `
  -Artifact 'F:\path\to\DeepSeek-V4-Flash-0731-CQTensors' `
  -PlacementPlan 'F:\path\to\placement.json' `
  -Chat -Agent -AgentFullAccess -Shell powershell -Workspace 'C:\path\to\project'
```

`-AgentFullAccess` is never silent. Tina prints a bilingual risk warning and
requires an interactive terminal line containing exactly `FULL_ACCESS`; piped
or non-interactive launches fail closed. After confirmation, the host accepts
absolute paths and can read, create, modify, rename, and delete regular files
and directories across the machine. It also exposes the selected host shell
with inherited environment and network access. The selected workspace remains
the Agent anchor for conversation artifacts and the default command cwd.
Structured file tools protect Tina host-control data under `.tina`, but an
unrestricted shell command can modify any accessible path. Directory deletion
requires `recursive=true`. The structured `delete` tool still requires an exact
`DELETE` confirmation in the terminal; shell commands such as `rm` or
`Remove-Item` remain outside that interceptor. `workspace_exec` is the
exception: it always requires the separate exact `EXECUTE` approval because
Cargo build scripts and tests execute with host permissions.

The bounded mode exposes `read`, `list`, `search`, `git_status`, `git_diff`,
and workspace-contained `apply_patch`, `write_file`, `mkdir`, `rename`, and
`delete`, plus `workspace_exec` for direct-argv Cargo validation. The first four mutations require exact `APPLY`; delete requires exact
`DELETE`. Workspace root, `.git`, `.tina`, parent escapes, and link/junction
paths fail closed. Full mode widens structured paths to the machine, adds
`shell`, and advertises unrestricted `network=true`.

`-WebSearchUrl` is separate from Full Access. It accepts an explicit HTTP(S)
SearXNG JSON endpoint, requires curl, and advertises `web_search` only after a
successful capability handshake. Queries and responses are bounded; returned
titles, snippets, and actual HTTP(S) URLs are marked untrusted external data.
There is no bundled public provider and Tina never fabricates missing results
or citations. File, command, and search results carry bounded output, byte
counts, and SHA-256 receipts; shell commands also have a bounded wall time.

Useful controls include `/status`, `/tools`, `/search [query]`, `/diff`, `/review [focus]`,
`/resume`, and `/agent-restart`. A restart performs a fresh handshake before
retiring the old host and retains the immutable conversation archive.

The exact full-prompt cache Beta is qualified on the frozen SM70 identity at
32K/128K/258,048 with paired warm TTFT, branch/scan read-only misses, and
decode interference below 2%. This does not measure aggregate production hit
rate, semantic retrieval, or longest-prefix reuse. The default CLI remains
`fast`, cache off, no MTP, and no DeepSeekSpark.
