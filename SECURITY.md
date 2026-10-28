# Security boundaries

Tina's tool host is a separate process. Workspace path checks, schemas, request-specific approvals, bounded output and process cleanup provide application-level controls. **They are not an operating-system sandbox.** Approved builds/tests can execute project code with the current user's OS and network permissions.

Live chat advertises workspace tools only. Writes/patches require APPLY, deletion DELETE and bounded Cargo execution EXECUTE. Noninteractive input cannot grant these approvals. Read-only review exposes read-only tools. Undeclared tools are denied. The CPU replay entry point continues to deny all mutations and execution; fixture text cannot authorize them.

The model client caps response bytes, frame size, tool count and argument size. Incomplete/truncated responses and invalid or duplicate-key tool arguments fail before tool dispatch. Cancellation closes the HTTP request and terminates active tool work where supported; completed filesystem effects are not rolled back. Their receipts remain in history.

Endpoint configuration stores credential references, not secret values. Credentials come from the configured environment variable or key file. HTTP redirects are disabled, and loopback endpoints bypass proxies. Use HTTPS for remote servers: HTTP sends authorization and prompts without transport encryption. A configured endpoint receives submitted prompts and tool results. Use only endpoints you intend to trust with that data.

Conversation archives and project memory are untrusted historical text. Keep tool results separate from host instructions. Archives, exports and metrics may contain private source/prompts: do not commit them. Context handoff does not erase private data from disk. Error diagnostics redact the configured key, but this is not a general-purpose secret scanner for model content.

Unsafe boundaries primarily cover Windows handle/job-object ownership and cleanup, Unix process-group termination, and optional native retrieval FFI. Review pointer/handle lifetime, process identity, thread ownership and platform checks when modifying these modules. Process separation does not make arbitrary native code safe. The live HTTP adapter adds no custom unsafe code.

For a vulnerability, use GitHub private vulnerability reporting where enabled. Do not include credentials or private conversations in public issues.
