# Tidal Luna splash assets

These frames belong to the original interactive UI module. The standalone CPU
verification commands do not launch that UI; use the model-free demo for a preview.

The private 4K source artwork is not part of the public repository. The three
runtime derivatives in this directory are deterministic Sixel frames embedded
in `cq-infer` at compile time:

- `luna_splash.sixel`: 640x360 Sixel frame.
- `luna_splash_960.sixel`: 960x540 Sixel frame.
- `luna_splash_1280.sixel`: 1280x720 Sixel frame.

The CLI has exactly two startup presentations:

- Windows Terminal and known Sixel-capable Linux terminals use the closest
  embedded Sixel resolution.
- VS Code's integrated terminal and unknown terminals use the code-native TINA
  wordmark. No image asset is required for this path.

Automatic selection is the default. `CQ_SPLASH_STYLE=tina` and
`CQ_SPLASH_STYLE=sixel` are diagnostic overrides; VS Code and undersized
terminals remain on TINA even when Sixel is requested. `NO_COLOR` renders TINA
without ANSI color sequences.

The private artwork source and its regeneration workflow are not distributed.
The included runtime frames retain their recorded upstream provenance.
