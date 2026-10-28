# Security boundaries

Tina's tool host is a separate process. Workspace path checks, request schemas, user approvals, bounded output and process cleanup provide application-level controls. **They are not an operating-system sandbox.** A permitted build or test can execute project code with the current user's OS permissions and network access.

The standalone replay denies all mutations, deletion and command execution. Model/fixture text cannot grant approval. Original interactive modules support request-bound approval flows; do not assume that every operation prompts under a Full Access policy.

Conversation archives and project memory contain untrusted historical text. Keep tool-result content separate from host instructions. Do not commit local histories, API credentials or private source artwork.

Unsafe code is isolated mainly around Windows handle/job-object cleanup, Unix process-group termination and optional native retrieval FFI. Review ownership, lifetime and platform checks at those boundaries when integrating the modules. Process isolation does not make arbitrary native code safe.

For a vulnerability, use GitHub private vulnerability reporting where enabled. Do not include credentials or private conversation contents in public issues.
