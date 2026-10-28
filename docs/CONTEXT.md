# Context admission and handoff

The default contract has 262,144 physical positions and at least 4,096 reserved output tokens. Its usable input limit is 258,048. Pressure thresholds are integer floors of 80%, 90% and 95% of that usable limit: 206,438 / 232,243 / 245,145.

| Supplied input tokens | Output request | Result |
| --- | --- | --- |
| 258,047 | 4,096 | admitted; handoff pressure |
| 258,048 | 4,096 | admitted at exact hard boundary |
| 258,049 | 4,096 | rejected; usable-input-limit |
| 262,144 | 4,096 | rejected; output reservation does not fit |
| 262,145 | 4,096 | rejected; physical-owner-capacity |
| 253,952 | 8,192 | admitted at exact hard boundary |
| 253,953 | 8,192 | rejected |

At or above the soft handoff threshold, the model integration should assemble a smaller active context from immutable history and a handoff before its next model submission. `context-check` reports admission and pressure; it does not perform an automatic handoff or launch inference.

The input limit remains capped at 258,048 even for larger development owners. Small owners reserve the requested output without silently reducing it. Empty input and an exhausted output reserve are rejected.

The archive preserves full history independently of the active model context. Handoff/resume does not authorize copying stale logits or mutable KV into a new owner. The standalone replay uses conservative UTF-8 byte accounting for history packing and does not include a model tokenizer.

The 32 MiB serialized prompt/document guard bounds input bytes. It is not a 32-million-token context or a guarantee that every document below it fits a model. JSON escaping and prompt metadata are included in serialized size.

## Upstream revisions

- `10a16cb2`: extracted baseline, including the calibrated budget and long-document constants.
- `cb2bcda3724b9eb0efbccd145708e4106fa6cbc9`: structured capacity admission; copied then adapted for small-owner output reservations.
- `3a218bdd21cf47bf25a9569c792085c9928e30d4`: agent-only host context; adapted while preserving Tina's receipt-first Agent instructions.

The historical engine integration under `integration/` is outside the standalone Cargo workspace. Its upstream patches are retained for integrators and are not a claim of a newly qualified GPU build.
