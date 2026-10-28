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

An initial attempt to start the local Augur package failed with Windows OS error 5 before its listener opened. The user then authorized real acceptance against DeepSeek using an environment credential. That real API acceptance passed: seven scripted cases plus four terminal cases (approved write/continuation, off-to-low mode switch with a read tool, cancelled stream without a committed partial answer, and a successful next request). See `alpha2-deepseek-live.json` for the sanitized record. No API keys, private launcher state or raw reasoning transcripts are included in the release.
