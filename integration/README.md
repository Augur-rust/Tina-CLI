# Historical model integration

`cq-infer/main.rs` and `Cargo.toml.reference` are the frozen original integration from Crystal Quant `10a16cb2`. They refer to engine crates that are not included in this standalone harness workspace.

The patches preserve subsequent repair work for integrators:

- `001`: receipt preservation after post-effect tool budget exhaustion. The standalone replay already uses receipt-first behavior.
- `002`: upstream `cb2bcda3` structured context admission across host and model owner. Its context-budget module is incorporated into the standalone crate with an additional small-owner reservation fix.
- `003`: upstream `3a218bdd` execution-context isolation. The standalone conversation module adapts that behavior while preserving the richer receipt-first Agent instructions.

These upstream patches have different original parent revisions; do not assume they apply as a sequential patch series to this snapshot. The reference integration was not rebuilt or GPU-qualified for the standalone release.
