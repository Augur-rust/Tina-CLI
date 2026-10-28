# Contributing

Open an issue describing the reproducible behavior and expected result, or submit a focused pull request. Never include API keys, private prompts or local conversation archives.

Run `scripts/Verify.ps1` before submitting. Keep actual model measurements separate from scripted CPU replay. Changes to context admission should include exact-boundary and output-reservation tests; changes to tool execution should preserve actual terminal receipts and denied-operation behavior.

New contributions are provided under the repository's MIT OR Apache-2.0 license. Preserve upstream notices and record material changes to extracted files in `provenance/source-manifest.json`.
