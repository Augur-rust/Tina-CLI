# Terminal acceptance — 2026-10-28

Windows x64, actual terminal raw mode, loopback HTTP fixture service.
The fixture supplied model response frames; all listed tools and socket operations were real.

| Case | Observation | Result |
| --- | --- | --- |
| Packaged first run | Start-Tina.cmd from the extracted ZIP opened setup, saved endpoint/workspace settings and entered the live terminal | PASS |
| Approved write | UI showed the write preview; typing APPLY created the requested file with its exact sentinel content | PASS |
| Tool continuation | The HTTP service received the actual tool receipt before issuing its final response | PASS |
| Cancel generation | Ctrl-C returned to the prompt and the HTTP peer observed disconnection | PASS |
| Next session | `/new` opened a new archive, then `/exit` completed cleanly | PASS |
| Cancel before first token | Transport regression test observed the server-side socket close before receiving any model token | PASS |

These checks establish terminal input, approvals, transport cancellation and receipt flow. They do not measure inference quality, GPU speed or native-engine cancellation behavior.

Real-model acceptance was attempted with the locally installed Augur package. Its native adapter exited with Windows OS error 5 (access denied) before opening its listener. The engine did not load the model. Root cause is not established. This is recorded as BLOCKED, not a pass or a model-quality failure. No private keys, launcher state or user conversations are included in public evidence.
