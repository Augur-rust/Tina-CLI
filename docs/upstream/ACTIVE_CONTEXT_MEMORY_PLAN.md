# Tina 258K Active Context Memory Plan

Status: `CPU_SOURCE_FOUNDATION_COMPLETE_WITH_GIT_PROJECT_MEMORY`

Date: `2026-08-08`

Authorities: `CONSTITUTION.md`, `PROJECT_GOAL.md`, `SINGLE_GPU_ROADMAP.md`,
and `KV_CACHE_CONSTITUTION.md`

## 1. Product boundary

Tina keeps three strictly separate memory planes:

```text
Git project memory (dedicated Git ref, explicit user facts, bounded prompt projection)
  + persistent conversation archive (host storage, unbounded by model context)
  -> bounded exact/semantic retrieval and provenance
  -> active prompt assembly (258,048 input-token product target)
  -> normal tokenizer + prefill
  -> authoritative MLA/DSA KV owner (262,144 positions, P1 target)
```

Neither project memory nor the archive/retrieval layer writes KV, changes DSA
Top-K, or creates a second model-state owner. Both are historical host data and
must re-enter through the official message encoder and normal prefill path. The
current finite development owner can exercise handoff mechanics, but it cannot
qualify or claim 256K support.

## 2. Capacity and pressure states

The product target is exactly `262,144` KV positions with `4,096` positions
reserved for output, leaving `258,048` input tokens. The default pressure
thresholds are integer floors of that active input budget:

| State | Fraction | Default token boundary | Required action |
|---|---:|---:|---|
| Normal | `<80%` | `<206,438` | Archive and index incrementally |
| Prepare | `>=80%` | `206,438` | Prepare an extractive handoff skeleton |
| Freeze | `>=90%` | `232,243` | Freeze the candidate source set and provenance |
| Handoff | `>=95%` | `245,145` | Compact, close the old owner, and start a new epoch |
| Hard limit | `100%` | `258,048` | Reject before model submission if packing still fails |

If the requested output exceeds `4,096`, its complete reserve is subtracted
before the same percentage calculation. A smaller development owner uses the
same state machine over its own physically admitted input budget and must say
that the 258K product target is inactive.

## 3. Project memory and persistent archive

### 3.1 Git project memory

Project memory stores only explicit user facts that are useful across sessions.
Its authoritative ref is `refs/tina/project-memory`; each mutation creates an
append-only commit whose tree contains only `project-memory.json`, then advances
the ref with an expected-old compare-and-swap. It never stages or commits the
workspace, changes HEAD/index/worktree, writes a `.tina` fallback file, or
stores KV snapshots, environment variables, automatic shell output, or tool
results. The project identity uses Git object format, repository roots with the
project-memory ref excluded, and a hashed origin; an empty history uses the
hashed Git common-dir fallback.

Open and write paths validate the current commit's parent binding, schema, tree shape, project
identity, entry/content/snapshot hashes, sequence and capacity. Non-Git
workspaces are `unavailable`; malformed refs, identity drift and ref conflicts
are fail-closed. Facts are marked historical and untrusted before prompt
injection and have an independent 2,048-token budget, 512-entry cap and 16 KiB
per-fact cap. Credential/private-key-like content is rejected. `/project-forget`
requires exact `DELETE` and appends a new snapshot without erasing Git history.
After a successful mutation the active model owner is closed and rebuilt on the
next turn. This layer is not a KV cache, conversation archive, retrieval index,
or performance measurement.

### 3.2 Persistent conversation archive

Every interactive conversation has stable conversation, epoch, turn, pin,
TodoList item/event, and handoff IDs. User and assistant turns, pins, TodoList
mutations, and handoffs are immutable records with content/record hashes. New
records are written to a temporary file in the same directory, flushed, and
atomically renamed. Recovery ignores only incomplete temporary files; malformed
committed files, hash drift, sequence gaps, identity drift, invalid TodoList
transitions, and invalid role/epoch causality fail closed.

The visible TodoList is a replayed view over immutable add, status, remove, and
clear-completed events. Pending and in-progress items are injected ahead of
retrieved history with stable IDs; completed and removed items remain auditable
but do not consume model context. `/todo remove` and `/todo clear-completed`
cross the same exact `DELETE` confirmation boundary as other destructive CLI
operations.

The archive has no model-token retention limit and is never silently evicted.
Storage exhaustion is an explicit error. Destructive archive operations remain
behind the CLI's exact `DELETE` confirmation boundary and are not part of this
slice.

## 4. Handoff contract

At the 95% boundary Tina:

1. Finalizes a bounded, deterministic extractive summary from complete archived
   turns. It is labelled as historical context, not as a host instruction.
2. Records source turn IDs and hashes, retained recent turn IDs, pinned facts,
   retrieved original excerpts, source/target epoch IDs, and summary hash.
3. Retains recent complete turns and the current user message verbatim.
4. Closes the old authoritative model owner before submitting the new prompt.
5. Starts a fresh context epoch whose prompt contains active TodoList state,
   the summary, pins, recent complete turns, and token-budgeted original
   excerpts.

No handoff reuses old logits, receipts, workspace, stream, event, or mutable KV.
The full transcript remains available in the archive after compaction.

## 5. Retrieval policy

Default retrieval is deterministic and exact-first:

1. exact phrase, path/symbol/error/string, and normalized term overlap;
2. user-authored turns receive a higher authority weight than assistant or tool
   records, without overriding exact match quality;
3. optional USEARCH ANN candidate recall from a real, identity-bound embedding
   provider in the independent `cq-retrieval` host boundary;
4. exact reranking, content-hash deduplication, causal filtering, and stable tie
   breaking;
5. token-budget packing with explicit source labels.

The normal history-injection budget is `2K-8K` tokens, defaulting to `4K`.
Complex tasks may request at most `16K`. The actual budget is reduced when the
owner, output reserve, recent complete turns, pins, or summary leave less room.
Feature hashes must never be described as semantic embeddings. If no qualified
embedding provider is configured, USEARCH is unavailable and the exact path
continues without changing chat startup. On Windows the native USEARCH/CXX
objects are intentionally not linked into the CUDA inference binary: upstream
uses the dynamic MSVC CRT while the current CUDA objects use the static CRT.
The persistent ANN store therefore stays in `cq-retrieval` for a separate host
service/process; `cq-infer` remains native-dependency-free on its default exact
retrieval path.

## 6. CPU scheduling

This slice is an H0 client, not a second thread pool. Its target allocation on
the local 7800X3D is:

- one physical core reserved for CUDA submission/completion;
- one physical core for scheduling, tokenization, streaming, and prompt packing;
- up to six physical cores for bounded archive hashing, embedding, ANN search,
  exact reranking, and I/O;
- SMT siblings only for low-priority persistence and maintenance.

Background indexing must be cancellable and must shrink or pause when the same
workload's visible decode P95/TPS regresses by 2%. The initial implementation is
synchronous and bounded until the shared `HostExecutionEngine` scheduler owns
these lanes; it must not create an uncoordinated permanent worker pool.

## 7. Delivery stages

| Stage | Scope | Completion gate |
|---|---|---|
| `MEM-1` | Pressure-state arithmetic and owner-aware limits | Exact default thresholds and boundary tests |
| `MEM-2` | Immutable archive, IDs, hashes, atomic append, recovery | Restart, corruption, sequence, and causality tests |
| `MEM-3` | Structured summary provenance and recent-turn retention | Deterministic replay and bounded-output tests |
| `MEM-4` | Exact-first retrieval and token packing | Stable ranking, user weighting, dedupe, and budget tests |
| `MEM-5` | Optional real-embedding USEARCH host boundary | Identity/dimension/persistence/add/reload tests; no CUDA-binary link; exact fallback remains usable |
| `MEM-6` | Tina chat, TodoList, and fresh-owner handoff | Archive/retrieve/todo/handoff tests without a GPU support claim |
| `PROJECT-MEM-1` | Git-backed explicit project facts and CLI boundary | Ref/HEAD/index/worktree isolation, identity/corruption/conflict tests, bounded untrusted packing and exact `DELETE` |
| `MEM-7` | Product qualification after P1 | 258K owner, TTFT, retrieval P95, decode-interference, and Agent task evidence |

`MEM-1` through `MEM-6` are complete CPU/source foundations. `MEM-7` is blocked
until P1 supplies the real `262,144`-position owner.

## 8. Acceptance and claim ceiling

The CPU/source foundation requires:

- exact threshold, output-reserve, and development-owner tests;
- restart recovery with stable IDs and deterministic ordering;
- committed corruption and identity drift fail closed;
- no archive mutation on failed assistant publication;
- bounded summary, retrieval, and prompt packing;
- deterministic ranking with user-message weighting and causal exclusion;
- immutable TodoList replay, restart recovery, bounded prompt injection, and
  exact deletion confirmation;
- missing semantic provider falls back to exact retrieval;
- project-memory writes never touch the workspace/index/HEAD, malformed or non-Git state fails closed,
  credentials are rejected, and the independent 2,048-token projection remains historical/untrusted;
- existing JSONL and non-chat CLI behavior remains unchanged.

Product activation additionally requires P1's real 256K owner, retrieval P95
`<150ms`, USEARCH query P95 `<20ms` when enabled, paired TTFT/Agent-turn benefit,
and less than 2% visible decode interference. Until then the status is
`CPU_FOUNDATION`, not `256K_SUPPORTED`.
