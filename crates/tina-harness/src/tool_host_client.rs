//! Bounded client for the independent `cq-tool-host` process.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const TOOL_HOST_PROTOCOL_VERSION: u16 = 2;
const TOOL_HOST_PROTOCOL_SCHEMA: &str = "tina.tool-host.v2";
const TOOL_HOST_INLINE_OUTPUT_BYTES: usize = 64 * 1024;
const MAX_TOOL_HOST_ERROR_BYTES: usize = 16 * 1024;
// JSON may encode one input byte as `\u00xx`; 512 KiB covers the host's
// 64 KiB inline payload plus all receipt fields at that worst-case expansion.
const MAX_TOOL_HOST_RESPONSE_BYTES: u64 = 512 * 1024;
const BASE_READ_ONLY_AGENT_TOOLS: [&str; 5] = ["git_diff", "git_status", "list", "read", "search"];
const READ_ONLY_AGENT_TOOLS: [&str; 6] = [
    "git_diff",
    "git_status",
    "list",
    "read",
    "search",
    "web_search",
];
const MUTATING_AGENT_TOOLS: [&str; 5] = ["apply_patch", "write_file", "mkdir", "delete", "rename"];
const FULL_ACCESS_MUTATING_AGENT_TOOLS: [&str; 6] = [
    "apply_patch",
    "write_file",
    "mkdir",
    "delete",
    "rename",
    "shell",
];
const EXECUTION_AGENT_TOOLS: [&str; 1] = ["workspace_exec"];
const MAX_APPLY_PATCH_ARGUMENT_BYTES: usize = 192 * 1024;
const MAX_APPLY_PATCH_EDITS: usize = 128;
const MAX_APPLY_PATCH_PREVIEW_EDITS: usize = 8;
const MAX_APPLY_PATCH_PREVIEW_FRAGMENT_BYTES: usize = 512;
const MAX_DELETE_ARGUMENT_BYTES: usize = 16 * 1024;
const MAX_FILE_MUTATION_ARGUMENT_BYTES: usize = 192 * 1024;
const MAX_FILE_MUTATION_PREVIEW_BYTES: usize = 2 * 1024;
const MAX_EXEC_ARGUMENT_BYTES: usize = 64 * 1024;
const MAX_EXEC_ARGS: usize = 64;
const MAX_EXEC_ARG_BYTES_PER_ITEM: usize = 8 * 1024;
const MAX_EXEC_DETAIL_BYTES: usize = 3 * 1024;
const DEFAULT_TOOL_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_EXECUTION_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const TOOL_CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(25);
const TOOL_CANCEL_GRACE: Duration = Duration::from_millis(250);
const TOOL_HOST_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

type ToolHostResponseFrame = Result<Vec<u8>, String>;
type ToolHostResponseReader = (Receiver<ToolHostResponseFrame>, JoinHandle<()>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ToolHostAccess {
    ReadOnly,
    Mutating,
    Execution,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ApplyPatchApprovalSummary {
    pub(crate) path: String,
    pub(crate) expected_sha256: String,
    pub(crate) edit_count: usize,
    pub(crate) removed_bytes: usize,
    pub(crate) added_bytes: usize,
    pub(crate) preview: String,
    pub(crate) preview_truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeleteApprovalSummary {
    pub(crate) path: String,
    pub(crate) recursive: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WriteFileApprovalSummary {
    pub(crate) path: String,
    pub(crate) create: bool,
    pub(crate) expected_sha256: Option<String>,
    pub(crate) content_bytes: usize,
    pub(crate) content_sha256: String,
    pub(crate) preview: String,
    pub(crate) preview_truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FileMutationApprovalSummary {
    pub(crate) operation: &'static str,
    pub(crate) detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkspaceExecApprovalSummary {
    pub(crate) argv: Vec<String>,
    pub(crate) cwd: Option<String>,
    pub(crate) timeout_ms: Option<u64>,
    pub(crate) detail: String,
}

#[derive(Debug)]
pub(crate) enum ToolHostClientError {
    Spawn(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    Protocol(String),
    Cancelled(String),
    DeadlineExceeded(String),
}

impl fmt::Display for ToolHostClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(error) => write!(formatter, "tool host spawn failure: {error}"),
            Self::Io(error) => write!(formatter, "tool host I/O failure: {error}"),
            Self::Json(error) => write!(formatter, "tool host JSON failure: {error}"),
            Self::Protocol(error) => write!(formatter, "tool host protocol failure: {error}"),
            Self::Cancelled(error) => write!(formatter, "tool host request cancelled: {error}"),
            Self::DeadlineExceeded(error) => {
                write!(formatter, "tool host request deadline exceeded: {error}")
            }
        }
    }
}

impl std::error::Error for ToolHostClientError {}

impl From<std::io::Error> for ToolHostClientError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ToolHostClientError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ToolExecution {
    pub(crate) ok: bool,
    pub(crate) exit_code: Option<i32>,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) output_sha256: Option<String>,
    pub(crate) output_truncated: bool,
    pub(crate) artifact_path: Option<String>,
}

impl ToolExecution {
    pub(crate) fn inline_bytes(&self) -> usize {
        self.stdout.len().saturating_add(self.stderr.len())
    }

    #[cfg(test)]
    pub(crate) fn prompt_content(&self) -> String {
        serde_json::json!({
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "output_sha256": self.output_sha256,
            "truncated": self.output_truncated,
            "artifact": self.artifact_path,
        })
        .to_string()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostResponse {
    v: u16,
    id: String,
    ok: bool,
    result: Option<HostResult>,
    error: Option<HostErrorBody>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostResult {
    kind: String,
    text: String,
    bytes: usize,
    sha256: String,
    truncated: bool,
    artifact: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellReceipt {
    shell: String,
    cwd: String,
    exit_code: Option<i32>,
    success: bool,
    timed_out: bool,
    stdout: String,
    stderr: String,
    stdout_bytes: usize,
    stderr_bytes: usize,
    stdout_captured_bytes: usize,
    stderr_captured_bytes: usize,
    stdout_truncated: bool,
    stderr_truncated: bool,
    stdout_inline_truncated: bool,
    stderr_inline_truncated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecReceipt {
    argv: Vec<String>,
    cwd: String,
    exit_code: Option<i32>,
    success: bool,
    timed_out: bool,
    stdout: String,
    stderr: String,
    stdout_bytes: usize,
    stderr_bytes: usize,
    stdout_captured_bytes: usize,
    stderr_captured_bytes: usize,
    stdout_truncated: bool,
    stderr_truncated: bool,
    stdout_inline_truncated: bool,
    stderr_inline_truncated: bool,
    policy: String,
    os_sandbox: bool,
    inherited_environment: bool,
    network_access: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostErrorBody {
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HostCapabilities {
    schema: String,
    version: u16,
    tools: Vec<String>,
    writes: bool,
    network: bool,
    #[serde(default)]
    web_search: bool,
    shell: bool,
    shell_dialect: String,
    cancellation: bool,
    max_deadline_ms: u64,
    workspace_exec_policy: String,
    workspace_exec_os_sandbox: bool,
    workspace_exec_inherited_environment: bool,
    workspace: String,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyPatchArgs {
    path: String,
    expected_sha256: String,
    edits: Vec<ApplyPatchEdit>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyPatchEdit {
    old: String,
    new: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeleteArgs {
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WriteFileArgs {
    path: String,
    content: String,
    #[serde(default)]
    expected_sha256: Option<String>,
    #[serde(default)]
    create: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MkdirArgs {
    path: String,
    #[serde(default)]
    recursive: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RenameArgs {
    from: String,
    to: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceExecArgs {
    argv: Vec<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

pub(crate) struct ToolHostClient {
    child: Child,
    stdin: Option<ChildStdin>,
    responses: Receiver<Result<Vec<u8>, String>>,
    reader_thread: Option<JoinHandle<()>>,
    process_tree: HostProcessTree,
    capabilities: BTreeSet<String>,
    binary: PathBuf,
    workspace: PathBuf,
    full_access: bool,
    shell: String,
    web_search_url: Option<String>,
}

impl ToolHostClient {
    pub(crate) fn spawn(
        binary: &Path,
        workspace: &Path,
        full_access: bool,
        shell: &str,
        web_search_url: Option<&str>,
    ) -> Result<Self, ToolHostClientError> {
        if !binary.is_file() {
            return Err(ToolHostClientError::Spawn(format!(
                "{} is not a file; build cq-tool-host or pass --tool-host",
                binary.display()
            )));
        }
        if !workspace.is_dir() {
            return Err(ToolHostClientError::Spawn(format!(
                "workspace is not a directory: {}",
                workspace.display()
            )));
        }
        let workspace = std::fs::canonicalize(workspace)
            .map_err(|error| ToolHostClientError::Spawn(error.to_string()))?;
        let mut command = Command::new(binary);
        command
            .arg("--workspace")
            .arg(&workspace)
            .arg("--shell")
            .arg(shell)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if full_access {
            command.arg("--full-access");
        }
        if let Some(url) = web_search_url {
            command.arg("--web-search-url").arg(url);
        }
        configure_process_group(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| ToolHostClientError::Spawn(error.to_string()))?;
        let process_tree = HostProcessTree::attach(&child).map_err(|error| {
            let _ = child.kill();
            let _ = child.wait();
            ToolHostClientError::Spawn(error)
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            ToolHostClientError::Spawn("tool host omitted its stdin pipe".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            ToolHostClientError::Spawn("tool host omitted its stdout pipe".to_string())
        })?;
        let (responses, reader_thread) = spawn_response_reader(stdout)?;
        let mut client = Self {
            child,
            stdin: Some(stdin),
            responses,
            reader_thread: Some(reader_thread),
            process_tree,
            capabilities: BTreeSet::new(),
            binary: binary.to_path_buf(),
            workspace,
            full_access,
            shell: shell.to_string(),
            web_search_url: web_search_url.map(str::to_string),
        };
        let handshake = client.request_raw_with_control(
            "host-handshake",
            "capabilities",
            Value::Null,
            HANDSHAKE_TIMEOUT,
            None,
            &mut || Ok(false),
        )?;
        if !handshake.ok {
            return Err(ToolHostClientError::Protocol(handshake.stderr));
        }
        let capabilities: HostCapabilities = serde_json::from_str(&handshake.stdout)?;
        client.capabilities =
            validate_capabilities(&capabilities, &client.workspace, full_access, shell)?;
        Ok(client)
    }

    pub(crate) fn binary(&self) -> &Path {
        &self.binary
    }

    pub(crate) fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub(crate) fn full_access(&self) -> bool {
        self.full_access
    }

    pub(crate) fn restart(&mut self) -> Result<(), ToolHostClientError> {
        let binary = self.binary.clone();
        let workspace = self.workspace.clone();
        let full_access = self.full_access;
        let shell = self.shell.clone();
        let web_search_url = self.web_search_url.clone();
        // Complete the replacement handshake before retiring the old host.
        // A spawn or protocol failure therefore leaves the existing client
        // usable instead of converting a recoverable restart into downtime.
        let replacement = Self::spawn(
            &binary,
            &workspace,
            full_access,
            &shell,
            web_search_url.as_deref(),
        )?;
        self.terminate_process_tree();
        if let Some(reader) = self.reader_thread.take() {
            let _ = reader.join();
        }
        *self = replacement;
        Ok(())
    }

    pub(crate) fn supports(&self, operation: &str) -> bool {
        self.capabilities.contains(operation)
    }

    pub(crate) fn read_only_capabilities(&self) -> Vec<&'static str> {
        READ_ONLY_AGENT_TOOLS
            .into_iter()
            .filter(|operation| self.supports(operation))
            .collect()
    }

    pub(crate) fn mutating_capabilities(&self) -> Vec<&'static str> {
        let operations = if self.full_access {
            &FULL_ACCESS_MUTATING_AGENT_TOOLS[..]
        } else {
            &MUTATING_AGENT_TOOLS[..]
        };
        operations
            .iter()
            .copied()
            .filter(|operation| self.supports(operation))
            .chain(
                EXECUTION_AGENT_TOOLS
                    .into_iter()
                    .filter(|operation| self.supports(operation)),
            )
            .collect()
    }

    pub(crate) fn access(&self, operation: &str) -> Option<ToolHostAccess> {
        if !self.supports(operation) {
            return None;
        }
        operation_access(operation)
    }

    pub(crate) fn execute_read_only(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
    ) -> Result<ToolExecution, ToolHostClientError> {
        self.execute_read_only_cancellable(call_id, operation, args, &mut || Ok(false))
    }

    pub(crate) fn execute_read_only_cancellable(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
        poll_cancel: &mut impl FnMut() -> io::Result<bool>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        self.execute_with_access_controlled(
            call_id,
            operation,
            args,
            ToolHostAccess::ReadOnly,
            DEFAULT_TOOL_REQUEST_TIMEOUT,
            poll_cancel,
        )
    }

    #[cfg(test)]
    pub(crate) fn execute_approved_mutation(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
    ) -> Result<ToolExecution, ToolHostClientError> {
        self.execute_approved_mutation_cancellable(call_id, operation, args, &mut || Ok(false))
    }

    pub(crate) fn execute_approved_mutation_cancellable(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
        poll_cancel: &mut impl FnMut() -> io::Result<bool>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        self.execute_with_access_controlled(
            call_id,
            operation,
            args,
            ToolHostAccess::Mutating,
            DEFAULT_TOOL_REQUEST_TIMEOUT,
            poll_cancel,
        )
    }

    pub(crate) fn execute_approved_execution_cancellable(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
        poll_cancel: &mut impl FnMut() -> io::Result<bool>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        self.execute_with_access_controlled(
            call_id,
            operation,
            args,
            ToolHostAccess::Execution,
            DEFAULT_EXECUTION_REQUEST_TIMEOUT,
            poll_cancel,
        )
    }

    fn execute_with_access_controlled(
        &mut self,
        call_id: &str,
        operation: &str,
        args: Value,
        required_access: ToolHostAccess,
        timeout: Duration,
        poll_cancel: &mut impl FnMut() -> io::Result<bool>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        if self.access(operation) != Some(required_access) {
            return Ok(ToolExecution {
                ok: false,
                exit_code: None,
                stdout: String::new(),
                stderr: format!(
                    "tool {operation:?} is not enabled with {required_access:?} access"
                ),
                output_sha256: None,
                output_truncated: false,
                artifact_path: None,
            });
        }
        let expected_exec = if required_access == ToolHostAccess::Execution {
            let summary = summarize_workspace_exec(&args).map_err(ToolHostClientError::Protocol)?;
            let requested_cwd = summary.cwd.as_deref().unwrap_or(".");
            let expected_cwd = canonical_workspace_exec_cwd(&self.workspace, requested_cwd)?;
            Some((summary.argv, expected_cwd))
        } else {
            None
        };
        self.request_raw_with_control(
            call_id,
            operation,
            args,
            timeout,
            expected_exec.as_ref(),
            poll_cancel,
        )
    }

    fn request_raw_with_control(
        &mut self,
        id: &str,
        operation: &str,
        args: Value,
        timeout: Duration,
        expected_exec: Option<&(Vec<String>, PathBuf)>,
        poll_cancel: &mut impl FnMut() -> io::Result<bool>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        let deadline_unix_ms = unix_millis()?
            .checked_add(timeout.as_millis().try_into().map_err(|_| {
                ToolHostClientError::Protocol("tool request timeout overflowed".to_string())
            })?)
            .ok_or_else(|| {
                ToolHostClientError::Protocol("tool request deadline overflowed".to_string())
            })?;
        let request = serde_json::json!({
            "v": TOOL_HOST_PROTOCOL_VERSION,
            "kind": "request",
            "id": id,
            "op": operation,
            "deadline_unix_ms": deadline_unix_ms,
            "args": args,
        });
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            ToolHostClientError::Protocol("tool host stdin is closed".to_string())
        })?;
        serde_json::to_writer(&mut *stdin, &request)?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;

        let started = Instant::now();
        let response = loop {
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return self.abort_active_request(id, false, expected_exec);
            }
            match self
                .responses
                .recv_timeout(remaining.min(TOOL_CANCEL_POLL_INTERVAL))
            {
                Ok(Ok(response)) => break response,
                Ok(Err(error)) => {
                    return Err(ToolHostClientError::Protocol(error));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ToolHostClientError::Protocol(
                        "tool host response reader closed".to_string(),
                    ));
                }
                Err(RecvTimeoutError::Timeout) => {
                    if poll_cancel()? {
                        return self.abort_active_request(id, true, expected_exec);
                    }
                }
            }
        };
        self.decode_response_bytes(&response, id, expected_exec)
    }

    fn abort_active_request(
        &mut self,
        id: &str,
        cancelled: bool,
        expected_exec: Option<&(Vec<String>, PathBuf)>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        let frame = serde_json::json!({
            "v": TOOL_HOST_PROTOCOL_VERSION,
            "kind": "cancel",
            "id": id,
        });
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = serde_json::to_writer(&mut *stdin, &frame);
            let _ = stdin.write_all(b"\n");
            let _ = stdin.flush();
        }
        // A mutation may have crossed its atomic commit boundary just as the
        // user pressed Ctrl-C. Prefer its real terminal receipt during the
        // short grace period; otherwise reclaim the complete process tree.
        if let Ok(Ok(response)) = self.responses.recv_timeout(TOOL_CANCEL_GRACE) {
            let response: HostResponse = serde_json::from_slice(&response)?;
            if response.id == id {
                return self.decode_response(response, id, expected_exec);
            }
        }
        self.terminate_process_tree();
        if cancelled {
            Err(ToolHostClientError::Cancelled(
                "the active request was cancelled before a terminal receipt".to_string(),
            ))
        } else {
            Err(ToolHostClientError::DeadlineExceeded(
                "the active request exceeded its bounded deadline".to_string(),
            ))
        }
    }

    fn decode_response(
        &self,
        response: HostResponse,
        id: &str,
        expected_exec: Option<&(Vec<String>, PathBuf)>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        if response.v != TOOL_HOST_PROTOCOL_VERSION || response.id != id {
            return Err(ToolHostClientError::Protocol(format!(
                "tool host response identity drifted: version {}, id {:?}",
                response.v, response.id
            )));
        }
        if response.ok {
            let result = response.result.ok_or_else(|| {
                ToolHostClientError::Protocol("successful response omitted result".to_string())
            })?;
            if response.error.is_some() {
                return Err(ToolHostClientError::Protocol(
                    "successful response also carried an error".to_string(),
                ));
            }
            validate_host_result(&result, &self.workspace)?;
            if result.kind == "shell" {
                return decode_shell_execution(result);
            }
            if result.kind == "exec" {
                let (argv, cwd) = expected_exec.ok_or_else(|| {
                    ToolHostClientError::Protocol(
                        "workspace_exec receipt had no request-bound expectation".to_string(),
                    )
                })?;
                return decode_exec_execution(result, argv, cwd);
            }
            Ok(ToolExecution {
                ok: true,
                exit_code: Some(0),
                stdout: result.text,
                stderr: String::new(),
                output_sha256: Some(result.sha256),
                output_truncated: result.truncated,
                artifact_path: result.artifact,
            })
        } else {
            let error = response.error.ok_or_else(|| {
                ToolHostClientError::Protocol("failed response omitted error".to_string())
            })?;
            if response.result.is_some() {
                return Err(ToolHostClientError::Protocol(
                    "failed response also carried a result".to_string(),
                ));
            }
            if error.retryable
                || error.code.is_empty()
                || error.code.len() > 64
                || !error
                    .code
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
                || error.message.len() > MAX_TOOL_HOST_ERROR_BYTES
            {
                return Err(ToolHostClientError::Protocol(
                    "tool host returned an invalid error receipt".to_string(),
                ));
            }
            Ok(ToolExecution {
                ok: false,
                exit_code: None,
                stdout: String::new(),
                stderr: format!("{}: {}", error.code, error.message),
                output_sha256: None,
                output_truncated: false,
                artifact_path: None,
            })
        }
    }

    fn decode_response_bytes(
        &self,
        bytes: &[u8],
        id: &str,
        expected_exec: Option<&(Vec<String>, PathBuf)>,
    ) -> Result<ToolExecution, ToolHostClientError> {
        if bytes.len() as u64 > MAX_TOOL_HOST_RESPONSE_BYTES || bytes.last() != Some(&b'\n') {
            return Err(ToolHostClientError::Protocol(format!(
                "tool host response exceeded {MAX_TOOL_HOST_RESPONSE_BYTES} bytes or lacked newline framing"
            )));
        }
        let response: HostResponse = serde_json::from_slice(bytes)?;
        self.decode_response(response, id, expected_exec)
    }

    pub(crate) fn shutdown(mut self) -> Result<(), ToolHostClientError> {
        drop(self.stdin.take());
        let started = Instant::now();
        let status = loop {
            if let Some(status) = self.child.try_wait()? {
                break status;
            }
            if started.elapsed() >= TOOL_HOST_SHUTDOWN_TIMEOUT {
                self.terminate_process_tree();
                break self.child.wait()?;
            }
            thread::sleep(TOOL_CANCEL_POLL_INTERVAL);
        };
        if let Some(reader) = self.reader_thread.take() {
            let _ = reader.join();
        }
        if status.success() {
            Ok(())
        } else {
            Err(ToolHostClientError::Protocol(format!(
                "tool host exited with {status}"
            )))
        }
    }

    fn terminate_process_tree(&mut self) {
        let _ = self.stdin.take();
        self.process_tree.terminate();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn decode_shell_execution(result: HostResult) -> Result<ToolExecution, ToolHostClientError> {
    if result.truncated || result.artifact.is_some() {
        return Err(ToolHostClientError::Protocol(
            "shell receipt must remain inline and complete".to_string(),
        ));
    }
    let receipt: ShellReceipt = serde_json::from_str(&result.text).map_err(|error| {
        ToolHostClientError::Protocol(format!("invalid shell receipt: {error}"))
    })?;
    if receipt.shell.trim().is_empty()
        || receipt.cwd.trim().is_empty()
        || receipt.stdout_captured_bytes > receipt.stdout_bytes
        || receipt.stderr_captured_bytes > receipt.stderr_bytes
        || (!receipt.stdout_truncated && receipt.stdout_captured_bytes != receipt.stdout_bytes)
        || (!receipt.stderr_truncated && receipt.stderr_captured_bytes != receipt.stderr_bytes)
        || (receipt.stdout_truncated && receipt.stdout_captured_bytes >= receipt.stdout_bytes)
        || (receipt.stderr_truncated && receipt.stderr_captured_bytes >= receipt.stderr_bytes)
        || (receipt.timed_out && receipt.success)
    {
        return Err(ToolHostClientError::Protocol(
            "shell receipt fields are inconsistent".to_string(),
        ));
    }
    Ok(ToolExecution {
        ok: receipt.success,
        exit_code: receipt.exit_code,
        stdout: receipt.stdout,
        stderr: receipt.stderr,
        output_sha256: Some(result.sha256),
        output_truncated: receipt.stdout_truncated
            || receipt.stderr_truncated
            || receipt.stdout_inline_truncated
            || receipt.stderr_inline_truncated,
        artifact_path: None,
    })
}

fn decode_exec_execution(
    result: HostResult,
    expected_argv: &[String],
    expected_cwd: &Path,
) -> Result<ToolExecution, ToolHostClientError> {
    if result.truncated || result.artifact.is_some() {
        return Err(ToolHostClientError::Protocol(
            "workspace_exec receipt must remain inline and complete".to_string(),
        ));
    }
    let receipt: ExecReceipt = serde_json::from_str(&result.text).map_err(|error| {
        ToolHostClientError::Protocol(format!("invalid workspace_exec receipt: {error}"))
    })?;
    if receipt.argv != expected_argv
        || validate_workspace_exec_argv_values(&receipt.argv).is_err()
        || receipt.cwd.trim().is_empty()
        || Path::new(&receipt.cwd) != expected_cwd
        || receipt.policy != "bounded-argv-cwd-v1"
        || receipt.os_sandbox
        || !receipt.inherited_environment
        || receipt.network_access != "unrestricted"
        || receipt.stdout_captured_bytes > receipt.stdout_bytes
        || receipt.stderr_captured_bytes > receipt.stderr_bytes
        || (!receipt.stdout_truncated && receipt.stdout_captured_bytes != receipt.stdout_bytes)
        || (!receipt.stderr_truncated && receipt.stderr_captured_bytes != receipt.stderr_bytes)
        || (receipt.stdout_truncated && receipt.stdout_captured_bytes >= receipt.stdout_bytes)
        || (receipt.stderr_truncated && receipt.stderr_captured_bytes >= receipt.stderr_bytes)
        || (receipt.timed_out && receipt.success)
        || (receipt.success && receipt.exit_code != Some(0))
        || (!receipt.success && !receipt.timed_out && receipt.exit_code == Some(0))
    {
        return Err(ToolHostClientError::Protocol(
            "workspace_exec receipt fields are inconsistent".to_string(),
        ));
    }
    Ok(ToolExecution {
        ok: receipt.success,
        exit_code: receipt.exit_code,
        stdout: receipt.stdout,
        stderr: receipt.stderr,
        output_sha256: Some(result.sha256),
        output_truncated: receipt.stdout_truncated
            || receipt.stderr_truncated
            || receipt.stdout_inline_truncated
            || receipt.stderr_inline_truncated,
        artifact_path: None,
    })
}

fn canonical_workspace_exec_cwd(
    workspace: &Path,
    raw: &str,
) -> Result<PathBuf, ToolHostClientError> {
    if raw.trim().is_empty()
        || Path::new(raw).is_absolute()
        || raw.starts_with('/')
        || raw.starts_with('\\')
        || raw.as_bytes().get(1) == Some(&b':')
        || Path::new(raw)
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(ToolHostClientError::Protocol(
            "workspace_exec cwd must be workspace-relative".to_string(),
        ));
    }
    let candidate = workspace.join(raw);
    let canonical = std::fs::canonicalize(&candidate).map_err(|error| {
        ToolHostClientError::Protocol(format!("workspace_exec cwd is not accessible: {error}"))
    })?;
    if !canonical.starts_with(workspace) || !canonical.is_dir() {
        return Err(ToolHostClientError::Protocol(
            "workspace_exec cwd escaped the workspace or is not a directory".to_string(),
        ));
    }
    Ok(canonical)
}

fn validate_workspace_exec_argv_values(argv: &[String]) -> Result<(), String> {
    if argv.is_empty() || argv.len() > MAX_EXEC_ARGS {
        return Err(format!(
            "workspace_exec argv must contain 1..={MAX_EXEC_ARGS} items"
        ));
    }
    let executable = argv[0].to_ascii_lowercase();
    if executable != "cargo" && executable != "cargo.exe" {
        return Err("workspace_exec currently allows only cargo or cargo.exe".to_string());
    }
    let subcommand = argv.get(1).map(String::as_str).unwrap_or_default();
    if !matches!(
        subcommand,
        "check" | "build" | "test" | "clippy" | "fmt" | "metadata"
    ) {
        return Err(format!(
            "cargo subcommand {subcommand:?} is not allowed by workspace_exec"
        ));
    }
    let mut total_bytes = 0_usize;
    for (index, arg) in argv.iter().enumerate() {
        if arg.is_empty() || arg.as_bytes().contains(&0) || arg.chars().any(char::is_control) {
            return Err(format!("workspace_exec argv item {index} is invalid"));
        }
        if arg.len() > MAX_EXEC_ARG_BYTES_PER_ITEM {
            return Err(format!(
                "workspace_exec argv item {index} exceeds {MAX_EXEC_ARG_BYTES_PER_ITEM} bytes"
            ));
        }
        total_bytes = total_bytes
            .checked_add(arg.len().saturating_add(1))
            .ok_or_else(|| "workspace_exec argv size overflowed".to_string())?;
        if total_bytes > MAX_EXEC_ARGUMENT_BYTES {
            return Err(format!(
                "workspace_exec argv exceeds {MAX_EXEC_ARGUMENT_BYTES} bytes"
            ));
        }
        if index > 0
            && (Path::new(arg).is_absolute()
                || arg.starts_with('/')
                || arg.starts_with('\\')
                || arg.as_bytes().get(1) == Some(&b':')
                || Path::new(arg)
                    .components()
                    .any(|component| matches!(component, Component::ParentDir))
                || arg == "--config"
                || arg.starts_with("--config=")
                || arg == "-C"
                || arg.starts_with("-C")
                || arg == "--manifest-path"
                || arg.starts_with("--manifest-path=")
                || arg == "--root"
                || arg.starts_with("--root=")
                || arg == "--target-dir"
                || arg.starts_with("--target-dir="))
        {
            return Err(format!(
                "workspace_exec argv item {index} changes the execution boundary"
            ));
        }
    }
    Ok(())
}

fn spawn_response_reader(
    stdout: ChildStdout,
) -> Result<ToolHostResponseReader, ToolHostClientError> {
    let (sender, receiver) = mpsc::channel();
    let reader = thread::Builder::new()
        .name("tina-tool-host-reader".to_string())
        .spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let mut response = Vec::new();
                match (&mut stdout)
                    .take(MAX_TOOL_HOST_RESPONSE_BYTES + 1)
                    .read_until(b'\n', &mut response)
                {
                    Ok(0) => break,
                    Ok(_) if response.len() as u64 > MAX_TOOL_HOST_RESPONSE_BYTES => {
                        let _ = sender.send(Err(format!(
                            "tool host response exceeded {MAX_TOOL_HOST_RESPONSE_BYTES} bytes"
                        )));
                        break;
                    }
                    Ok(_) if response.last() != Some(&b'\n') => {
                        let _ = sender
                            .send(Err("tool host response lacked newline framing".to_string()));
                        break;
                    }
                    Ok(_) => {
                        if sender.send(Ok(response)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        })
        .map_err(|error| ToolHostClientError::Spawn(error.to_string()))?;
    Ok((receiver, reader))
}

fn unix_millis() -> Result<u64, ToolHostClientError> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| ToolHostClientError::Protocol(error.to_string()))?
        .as_millis();
    u64::try_from(millis)
        .map_err(|_| ToolHostClientError::Protocol("system time overflowed".to_string()))
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

#[cfg(windows)]
struct OwnedWindowsHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl OwnedWindowsHandle {
    fn new(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<Self, String> {
        if handle.is_null() {
            Err(std::io::Error::last_os_error().to_string())
        } else {
            Ok(Self(handle))
        }
    }

    const fn as_raw(&self) -> windows_sys::Win32::Foundation::HANDLE {
        self.0
    }
}

#[cfg(windows)]
impl Drop for OwnedWindowsHandle {
    fn drop(&mut self) {
        // SAFETY: this wrapper is constructed only from a non-null handle
        // returned by a Win32 creator and owns that handle exclusively.
        // Drop is the sole close site, including all attach error paths.
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
struct HostProcessTree {
    handle: OwnedWindowsHandle,
}

#[cfg(unix)]
struct HostProcessTree {
    process_group: i32,
}

#[cfg(not(any(unix, windows)))]
struct HostProcessTree;

impl HostProcessTree {
    fn attach(child: &Child) -> Result<Self, String> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
                SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            };
            use windows_sys::Win32::System::Threading::OpenProcess;
            // SAFETY: null names request a private kernel object; no pointer
            // arguments are dereferenced by the call.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            let handle = OwnedWindowsHandle::new(handle)?;
            // SAFETY: this Windows job-information structure is a plain C
            // value whose all-zero representation is valid; only its limit
            // flag is changed before the FFI call.
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `limits` remains initialized and borrowed for the whole
            // synchronous call; `handle` is a live job owned by this scope.
            let configured = unsafe {
                SetInformationJobObject(
                    handle.as_raw(),
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION)
                        .cast::<core::ffi::c_void>(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            // SAFETY: `child.id()` identifies the live child created by this
            // client; these rights are limited to job assignment and querying
            // or terminating that process.
            let process = unsafe {
                OpenProcess(
                    windows_sys::Win32::System::Threading::PROCESS_SET_QUOTA
                        | windows_sys::Win32::System::Threading::PROCESS_TERMINATE
                        | windows_sys::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION,
                    0,
                    child.id(),
                )
            };
            let process = OwnedWindowsHandle::new(process)?;
            // SAFETY: both wrappers hold live handles for this synchronous
            // assignment call; no handle is transferred to another owner.
            let assigned = unsafe { AssignProcessToJobObject(handle.as_raw(), process.as_raw()) };
            if assigned == 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            Ok(Self { handle })
        }
        #[cfg(unix)]
        {
            Ok(Self {
                process_group: child.id() as i32,
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = child;
            Ok(Self)
        }
    }

    fn terminate(&self) {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::JobObjects::TerminateJobObject;
            // SAFETY: the job handle is owned by this struct and remains live
            // until its field is dropped after this method returns.
            unsafe {
                let _ = TerminateJobObject(self.handle.as_raw(), 1);
            }
        }
        #[cfg(unix)]
        // SAFETY: `configure_process_group` puts the child in a fresh
        // process group; the negative PID targets only that group.
        unsafe {
            let _ = libc::kill(-self.process_group, libc::SIGKILL);
        }
    }
}

impl Drop for HostProcessTree {
    fn drop(&mut self) {
        self.terminate();
    }
}

fn validate_capabilities(
    capabilities: &HostCapabilities,
    workspace: &Path,
    full_access: bool,
    shell: &str,
) -> Result<BTreeSet<String>, ToolHostClientError> {
    let expected_scope = if full_access { "machine" } else { "workspace" };
    let advertised_scope = capabilities.scope.as_deref().unwrap_or("workspace");
    if capabilities.schema != TOOL_HOST_PROTOCOL_SCHEMA
        || capabilities.version != TOOL_HOST_PROTOCOL_VERSION
        || !capabilities.writes
        || capabilities.network != full_access
        || capabilities.shell != full_access
        || capabilities.shell_dialect != shell
        || !capabilities.cancellation
        || capabilities.max_deadline_ms == 0
        || capabilities.max_deadline_ms > 5 * 60 * 1_000
        || capabilities.workspace_exec_policy != "bounded-argv-cwd-v1"
        || capabilities.workspace_exec_os_sandbox
        || !capabilities.workspace_exec_inherited_environment
        || Path::new(&capabilities.workspace) != workspace
        || advertised_scope != expected_scope
    {
        return Err(ToolHostClientError::Protocol(
            "tool host capability identity or safety flags drifted".to_string(),
        ));
    }
    let advertised = capabilities.tools.iter().cloned().collect::<BTreeSet<_>>();
    let mut expected = BASE_READ_ONLY_AGENT_TOOLS
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    if capabilities.web_search {
        expected.insert("web_search".to_string());
    }
    let mutating = if full_access {
        &FULL_ACCESS_MUTATING_AGENT_TOOLS[..]
    } else {
        &MUTATING_AGENT_TOOLS[..]
    };
    expected.extend(mutating.iter().copied().map(str::to_string));
    expected.extend(EXECUTION_AGENT_TOOLS.iter().copied().map(str::to_string));
    expected.insert("capabilities".to_string());
    if advertised.len() != capabilities.tools.len() || advertised != expected {
        return Err(ToolHostClientError::Protocol(
            "tool host advertised an unexpected or duplicate capability".to_string(),
        ));
    }
    Ok(READ_ONLY_AGENT_TOOLS
        .into_iter()
        .filter(|operation| advertised.contains(*operation))
        .chain(mutating.iter().copied())
        .chain(EXECUTION_AGENT_TOOLS)
        .map(str::to_string)
        .collect())
}

fn validate_host_result(result: &HostResult, workspace: &Path) -> Result<(), ToolHostClientError> {
    if !matches!(result.kind.as_str(), "json" | "text" | "shell" | "exec")
        || result.text.len() > TOOL_HOST_INLINE_OUTPUT_BYTES
        || !valid_sha256(&result.sha256)
    {
        return Err(ToolHostClientError::Protocol(
            "tool host returned an invalid result receipt".to_string(),
        ));
    }
    if result.truncated {
        if result.bytes <= result.text.len() {
            return Err(ToolHostClientError::Protocol(
                "truncated tool result did not report a larger source payload".to_string(),
            ));
        }
        validate_artifact(
            result.artifact.as_deref(),
            &result.sha256,
            result.bytes,
            workspace,
        )?;
    } else if result.bytes != result.text.len()
        || result.artifact.is_some()
        || sha256_hex(result.text.as_bytes()) != result.sha256
    {
        return Err(ToolHostClientError::Protocol(
            "complete tool result receipt did not match its payload".to_string(),
        ));
    }
    Ok(())
}

fn validate_artifact(
    artifact: Option<&str>,
    output_sha256: &str,
    output_bytes: usize,
    workspace: &Path,
) -> Result<(), ToolHostClientError> {
    let artifact = artifact.ok_or_else(|| {
        ToolHostClientError::Protocol("truncated tool result omitted its artifact".to_string())
    })?;
    let path = Path::new(artifact);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir))
        || artifact.replace('\\', "/") != format!(".tina/tool-artifacts/{output_sha256}.txt")
    {
        return Err(ToolHostClientError::Protocol(
            "tool result artifact escaped its content-addressed directory".to_string(),
        ));
    }
    let absolute = workspace.join(path);
    let metadata = std::fs::symlink_metadata(&absolute).map_err(|error| {
        ToolHostClientError::Protocol(format!("tool result artifact is not accessible: {error}"))
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() != output_bytes as u64
    {
        return Err(ToolHostClientError::Protocol(
            "tool result artifact type or byte count drifted".to_string(),
        ));
    }
    let canonical = std::fs::canonicalize(&absolute)?;
    if !canonical.starts_with(workspace) || sha256_file(&canonical)? != output_sha256 {
        return Err(ToolHostClientError::Protocol(
            "tool result artifact content or workspace identity drifted".to_string(),
        ));
    }
    Ok(())
}

fn operation_access(operation: &str) -> Option<ToolHostAccess> {
    if READ_ONLY_AGENT_TOOLS.contains(&operation) {
        Some(ToolHostAccess::ReadOnly)
    } else if EXECUTION_AGENT_TOOLS.contains(&operation) {
        Some(ToolHostAccess::Execution)
    } else if FULL_ACCESS_MUTATING_AGENT_TOOLS.contains(&operation) {
        Some(ToolHostAccess::Mutating)
    } else {
        None
    }
}

pub(crate) fn summarize_apply_patch(
    arguments: &Value,
) -> Result<ApplyPatchApprovalSummary, String> {
    let encoded_bytes = serde_json::to_vec(arguments)
        .map_err(|error| format!("apply_patch arguments are not encodable: {error}"))?
        .len();
    if encoded_bytes > MAX_APPLY_PATCH_ARGUMENT_BYTES {
        return Err(format!(
            "apply_patch arguments are {encoded_bytes} bytes; limit is {MAX_APPLY_PATCH_ARGUMENT_BYTES}"
        ));
    }
    let args: ApplyPatchArgs = serde_json::from_value(arguments.clone())
        .map_err(|error| format!("invalid apply_patch arguments: {error}"))?;
    if args.path.is_empty() || args.path.as_bytes().contains(&0) {
        return Err("apply_patch path is empty or contains NUL".to_string());
    }
    if !valid_sha256(&args.expected_sha256) {
        return Err(
            "apply_patch expected_sha256 must be 64 lowercase hexadecimal characters".to_string(),
        );
    }
    if args.edits.is_empty() || args.edits.len() > MAX_APPLY_PATCH_EDITS {
        return Err(format!(
            "apply_patch requires 1..={MAX_APPLY_PATCH_EDITS} edits"
        ));
    }
    let mut removed_bytes = 0_usize;
    let mut added_bytes = 0_usize;
    for (index, edit) in args.edits.iter().enumerate() {
        if edit.old.is_empty() || edit.old == edit.new {
            return Err(format!(
                "apply_patch edit {index} must replace non-empty text with different text"
            ));
        }
        removed_bytes = removed_bytes
            .checked_add(edit.old.len())
            .ok_or_else(|| "apply_patch removed-byte count overflowed".to_string())?;
        added_bytes = added_bytes
            .checked_add(edit.new.len())
            .ok_or_else(|| "apply_patch added-byte count overflowed".to_string())?;
    }
    let preview_truncated = args.edits.len() > MAX_APPLY_PATCH_PREVIEW_EDITS
        || args.edits.iter().any(|edit| {
            edit.old.len() > MAX_APPLY_PATCH_PREVIEW_FRAGMENT_BYTES
                || edit.new.len() > MAX_APPLY_PATCH_PREVIEW_FRAGMENT_BYTES
        });
    let mut preview = String::new();
    for (index, edit) in args
        .edits
        .iter()
        .take(MAX_APPLY_PATCH_PREVIEW_EDITS)
        .enumerate()
    {
        use std::fmt::Write as _;
        let old = bounded_debug_fragment(&edit.old, MAX_APPLY_PATCH_PREVIEW_FRAGMENT_BYTES);
        let new = bounded_debug_fragment(&edit.new, MAX_APPLY_PATCH_PREVIEW_FRAGMENT_BYTES);
        writeln!(&mut preview, "edit {}", index + 1)
            .expect("writing a patch preview to String cannot fail");
        writeln!(&mut preview, "- {old}").expect("writing a patch preview to String cannot fail");
        writeln!(&mut preview, "+ {new}").expect("writing a patch preview to String cannot fail");
    }
    Ok(ApplyPatchApprovalSummary {
        path: args.path,
        expected_sha256: args.expected_sha256,
        edit_count: args.edits.len(),
        removed_bytes,
        added_bytes,
        preview,
        preview_truncated,
    })
}

pub(crate) fn summarize_write_file(arguments: &Value) -> Result<WriteFileApprovalSummary, String> {
    let encoded_bytes = serde_json::to_vec(arguments)
        .map_err(|error| format!("write_file arguments are not encodable: {error}"))?
        .len();
    if encoded_bytes > MAX_FILE_MUTATION_ARGUMENT_BYTES {
        return Err(format!(
            "write_file arguments are {encoded_bytes} bytes; limit is {MAX_FILE_MUTATION_ARGUMENT_BYTES}"
        ));
    }
    let args: WriteFileArgs = serde_json::from_value(arguments.clone())
        .map_err(|error| format!("invalid write_file arguments: {error}"))?;
    validate_mutation_path("write_file", &args.path)?;
    if let Some(expected) = args.expected_sha256.as_deref() {
        if !valid_sha256(expected) {
            return Err(
                "write_file expected_sha256 must be 64 lowercase hexadecimal characters"
                    .to_string(),
            );
        }
    }
    let preview_end = args
        .content
        .char_indices()
        .map(|(offset, character)| offset + character.len_utf8())
        .take_while(|end| *end <= MAX_FILE_MUTATION_PREVIEW_BYTES)
        .last()
        .unwrap_or(0);
    Ok(WriteFileApprovalSummary {
        path: args.path,
        create: args.create,
        expected_sha256: args.expected_sha256,
        content_bytes: args.content.len(),
        content_sha256: sha256_hex(args.content.as_bytes()),
        preview: args.content[..preview_end].to_string(),
        preview_truncated: preview_end < args.content.len(),
    })
}

pub(crate) fn summarize_mkdir(arguments: &Value) -> Result<FileMutationApprovalSummary, String> {
    let args: MkdirArgs = parse_small_mutation_args("mkdir", arguments)?;
    validate_mutation_path("mkdir", &args.path)?;
    Ok(FileMutationApprovalSummary {
        operation: "mkdir",
        detail: format!("path={:?}; recursive={}", args.path, args.recursive),
    })
}

pub(crate) fn summarize_rename(arguments: &Value) -> Result<FileMutationApprovalSummary, String> {
    let args: RenameArgs = parse_small_mutation_args("rename", arguments)?;
    validate_mutation_path("rename source", &args.from)?;
    validate_mutation_path("rename destination", &args.to)?;
    Ok(FileMutationApprovalSummary {
        operation: "rename",
        detail: format!("from={:?}; to={:?}", args.from, args.to),
    })
}

pub(crate) fn summarize_delete(arguments: &Value) -> Result<DeleteApprovalSummary, String> {
    let encoded_bytes = serde_json::to_vec(arguments)
        .map_err(|error| format!("delete arguments are not encodable: {error}"))?
        .len();
    if encoded_bytes > MAX_DELETE_ARGUMENT_BYTES {
        return Err(format!(
            "delete arguments are {encoded_bytes} bytes; limit is {MAX_DELETE_ARGUMENT_BYTES}"
        ));
    }
    let args: DeleteArgs = serde_json::from_value(arguments.clone())
        .map_err(|error| format!("invalid delete arguments: {error}"))?;
    if args.path.trim().is_empty() || args.path.as_bytes().contains(&0) {
        return Err("delete path is empty or contains NUL".to_string());
    }
    Ok(DeleteApprovalSummary {
        path: args.path,
        recursive: args.recursive,
    })
}

pub(crate) fn summarize_workspace_exec(
    arguments: &Value,
) -> Result<WorkspaceExecApprovalSummary, String> {
    let encoded_bytes = serde_json::to_vec(arguments)
        .map_err(|error| format!("workspace_exec arguments are not encodable: {error}"))?
        .len();
    if encoded_bytes > 64 * 1024 {
        return Err(format!(
            "workspace_exec arguments are {encoded_bytes} bytes; limit is 65536"
        ));
    }
    let args: WorkspaceExecArgs = serde_json::from_value(arguments.clone())
        .map_err(|error| format!("invalid workspace_exec arguments: {error}"))?;
    if args.argv.is_empty() || args.argv.len() > MAX_EXEC_ARGS {
        return Err(format!(
            "workspace_exec argv must contain 1..={MAX_EXEC_ARGS} items"
        ));
    }
    let executable = args.argv[0].to_ascii_lowercase();
    if executable != "cargo" && executable != "cargo.exe" {
        return Err("workspace_exec currently allows only cargo or cargo.exe".to_string());
    }
    let subcommand = args.argv.get(1).map(String::as_str).unwrap_or_default();
    if !matches!(
        subcommand,
        "check" | "build" | "test" | "clippy" | "fmt" | "metadata"
    ) {
        return Err(format!(
            "cargo subcommand {subcommand:?} is not allowed by workspace_exec"
        ));
    }
    for (index, arg) in args.argv.iter().enumerate() {
        if arg.is_empty() || arg.as_bytes().contains(&0) || arg.chars().any(char::is_control) {
            return Err(format!("workspace_exec argv item {index} is invalid"));
        }
        if arg.len() > MAX_EXEC_ARG_BYTES_PER_ITEM {
            return Err(format!(
                "workspace_exec argv item {index} exceeds {MAX_EXEC_ARG_BYTES_PER_ITEM} bytes"
            ));
        }
        if index > 0
            && (Path::new(arg).is_absolute()
                || arg.starts_with('/')
                || arg.starts_with('\\')
                || arg.as_bytes().get(1) == Some(&b':')
                || Path::new(arg)
                    .components()
                    .any(|component| matches!(component, Component::ParentDir))
                || arg == "--config"
                || arg.starts_with("--config=")
                || arg == "-C"
                || arg.starts_with("-C")
                || arg == "--manifest-path"
                || arg.starts_with("--manifest-path=")
                || arg == "--root"
                || arg.starts_with("--root=")
                || arg == "--target-dir"
                || arg.starts_with("--target-dir="))
        {
            return Err(format!(
                "workspace_exec argv item {index} changes the execution boundary"
            ));
        }
    }
    let argument_bytes = args
        .argv
        .iter()
        .try_fold(0_usize, |total, arg| {
            total.checked_add(arg.len().saturating_add(1))
        })
        .ok_or_else(|| "workspace_exec argv size overflowed".to_string())?;
    if argument_bytes > MAX_EXEC_ARGUMENT_BYTES {
        return Err(format!(
            "workspace_exec argv exceeds {MAX_EXEC_ARGUMENT_BYTES} bytes"
        ));
    }
    if let Some(cwd) = args.cwd.as_deref() {
        if cwd.trim().is_empty()
            || cwd.as_bytes().contains(&0)
            || Path::new(cwd).is_absolute()
            || cwd.starts_with('/')
            || cwd.starts_with('\\')
            || cwd.as_bytes().get(1) == Some(&b':')
            || Path::new(cwd)
                .components()
                .any(|component| matches!(component, Component::ParentDir))
        {
            return Err("workspace_exec cwd must be a workspace-relative directory".to_string());
        }
    }
    if args
        .timeout_ms
        .is_some_and(|timeout| timeout == 0 || timeout > 5 * 60 * 1_000)
    {
        return Err("workspace_exec timeout must be 1..=300000 ms".to_string());
    }
    let mut rendered = args
        .argv
        .iter()
        .map(|arg| bounded_debug_fragment(arg, 128))
        .collect::<Vec<_>>()
        .join(" ");
    if rendered.len() > MAX_EXEC_DETAIL_BYTES {
        let end = rendered
            .char_indices()
            .map(|(offset, character)| offset + character.len_utf8())
            .take_while(|end| *end <= MAX_EXEC_DETAIL_BYTES)
            .last()
            .unwrap_or(0);
        rendered.truncate(end);
        rendered.push_str(" ...");
    }
    let cwd = args.cwd.clone();
    let timeout_ms = args.timeout_ms;
    Ok(WorkspaceExecApprovalSummary {
        argv: args.argv,
        cwd: cwd.clone(),
        timeout_ms,
        detail: format!(
            "argv={rendered}; cwd={:?}; timeout_ms={:?}; policy=bounded-argv-cwd-v1; os_sandbox=false; inherited_environment=true; network=unrestricted",
            cwd, timeout_ms
        ),
    })
}

fn parse_small_mutation_args<T: for<'de> Deserialize<'de>>(
    operation: &str,
    arguments: &Value,
) -> Result<T, String> {
    let encoded_bytes = serde_json::to_vec(arguments)
        .map_err(|error| format!("{operation} arguments are not encodable: {error}"))?
        .len();
    if encoded_bytes > MAX_DELETE_ARGUMENT_BYTES {
        return Err(format!(
            "{operation} arguments are {encoded_bytes} bytes; limit is {MAX_DELETE_ARGUMENT_BYTES}"
        ));
    }
    serde_json::from_value(arguments.clone())
        .map_err(|error| format!("invalid {operation} arguments: {error}"))
}

fn validate_mutation_path(operation: &str, path: &str) -> Result<(), String> {
    if path.trim().is_empty() || path.as_bytes().contains(&0) {
        return Err(format!("{operation} path is empty or contains NUL"));
    }
    Ok(())
}

fn bounded_debug_fragment(value: &str, max_bytes: usize) -> String {
    let end = value
        .char_indices()
        .map(|(offset, character)| offset + character.len_utf8())
        .take_while(|end| *end <= max_bytes)
        .last()
        .unwrap_or(0);
    let mut rendered = format!("{:?}", &value[..end]);
    if end < value.len() {
        rendered.push_str(" ...");
    }
    rendered
}

fn sha256_file(path: &Path) -> Result<String, ToolHostClientError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Drop for ToolHostClient {
    fn drop(&mut self) {
        if self.stdin.is_some() {
            self.process_tree.terminate();
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(reader) = self.reader_thread.take() {
            let _ = reader.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn prompt_content_is_structured_and_bounded_by_the_host_receipt() {
        let execution = ToolExecution {
            ok: true,
            exit_code: Some(0),
            stdout: "ok".to_string(),
            stderr: String::new(),
            output_sha256: Some("ab".repeat(32)),
            output_truncated: false,
            artifact_path: None,
        };
        let value: Value = serde_json::from_str(&execution.prompt_content()).unwrap();
        assert_eq!(value["stdout"], "ok");
        assert_eq!(execution.inline_bytes(), 2);
    }

    #[test]
    fn response_frame_covers_worst_case_json_escaping() {
        let payload = "\0".repeat(TOOL_HOST_INLINE_OUTPUT_BYTES);
        let response = serde_json::json!({
            "v": TOOL_HOST_PROTOCOL_VERSION,
            "id": "tool-frame",
            "ok": true,
            "result": {
                "kind": "text",
                "text": payload,
                "bytes": TOOL_HOST_INLINE_OUTPUT_BYTES + 1,
                "sha256": "ab".repeat(32),
                "truncated": true,
                "artifact": format!(".tina/tool-artifacts/{}.txt", "ab".repeat(32)),
            },
            "error": null,
        });
        let encoded = serde_json::to_vec(&response).unwrap();
        assert!(encoded.len() as u64 <= MAX_TOOL_HOST_RESPONSE_BYTES);
    }

    #[test]
    fn result_receipts_fail_closed_on_hash_and_artifact_drift() {
        let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
        let complete = HostResult {
            kind: "text".to_string(),
            text: "ok".to_string(),
            bytes: 2,
            sha256: sha256_hex(b"ok"),
            truncated: false,
            artifact: None,
        };
        validate_host_result(&complete, &workspace).unwrap();

        let mut bad_hash = complete;
        bad_hash.sha256 = "00".repeat(32);
        assert!(validate_host_result(&bad_hash, &workspace).is_err());

        let digest = "ab".repeat(32);
        let escaped = HostResult {
            kind: "text".to_string(),
            text: "part".to_string(),
            bytes: 8,
            sha256: digest.clone(),
            truncated: true,
            artifact: Some("../outside.txt".to_string()),
        };
        assert!(validate_host_result(&escaped, &workspace).is_err());
    }

    #[test]
    fn capability_handshake_exposes_the_frozen_read_and_write_sets() {
        let workspace = std::env::current_dir().unwrap().canonicalize().unwrap();
        let capabilities = HostCapabilities {
            schema: TOOL_HOST_PROTOCOL_SCHEMA.to_string(),
            version: TOOL_HOST_PROTOCOL_VERSION,
            tools: ["capabilities"]
                .into_iter()
                .chain(BASE_READ_ONLY_AGENT_TOOLS)
                .chain(MUTATING_AGENT_TOOLS)
                .chain(EXECUTION_AGENT_TOOLS)
                .map(str::to_string)
                .collect(),
            writes: true,
            network: false,
            web_search: false,
            shell: false,
            shell_dialect: "powershell".to_string(),
            cancellation: true,
            max_deadline_ms: 5 * 60 * 1_000,
            workspace_exec_policy: "bounded-argv-cwd-v1".to_string(),
            workspace_exec_os_sandbox: false,
            workspace_exec_inherited_environment: true,
            workspace: workspace.display().to_string(),
            scope: Some("workspace".to_string()),
        };
        let tools = validate_capabilities(&capabilities, &workspace, false, "powershell").unwrap();
        assert_eq!(
            tools.len(),
            BASE_READ_ONLY_AGENT_TOOLS.len()
                + MUTATING_AGENT_TOOLS.len()
                + EXECUTION_AGENT_TOOLS.len()
        );
        assert!(!tools.contains("capabilities"));
        assert_eq!(operation_access("read"), Some(ToolHostAccess::ReadOnly));
        assert_eq!(
            operation_access("apply_patch"),
            Some(ToolHostAccess::Mutating)
        );
        assert_eq!(
            operation_access("workspace_exec"),
            Some(ToolHostAccess::Execution)
        );

        let mut unsafe_capabilities = capabilities;
        unsafe_capabilities.tools.push("shell".to_string());
        assert!(
            validate_capabilities(&unsafe_capabilities, &workspace, false, "powershell").is_err()
        );

        let mut web_capabilities = unsafe_capabilities.clone();
        web_capabilities.tools.pop();
        web_capabilities.tools.push("web_search".to_string());
        web_capabilities.web_search = true;
        let web_tools =
            validate_capabilities(&web_capabilities, &workspace, false, "powershell").unwrap();
        assert!(web_tools.contains("web_search"));

        let mut full_capabilities = unsafe_capabilities;
        full_capabilities.tools.pop();
        full_capabilities.tools.extend(
            FULL_ACCESS_MUTATING_AGENT_TOOLS
                .iter()
                .skip(MUTATING_AGENT_TOOLS.len())
                .map(|name| (*name).to_string()),
        );
        full_capabilities.scope = Some("machine".to_string());
        full_capabilities.network = true;
        full_capabilities.shell = true;
        assert!(validate_capabilities(&full_capabilities, &workspace, true, "powershell").is_ok());
    }

    #[test]
    fn apply_patch_summary_is_bounded_and_rejects_noop_edits() {
        let digest = "ab".repeat(32);
        let args = serde_json::json!({
            "path": "src/lib.rs",
            "expected_sha256": digest,
            "edits": [{"old": "before", "new": "after"}],
        });
        let summary = summarize_apply_patch(&args).unwrap();
        assert_eq!(summary.path, "src/lib.rs");
        assert_eq!(summary.edit_count, 1);
        assert_eq!(summary.removed_bytes, 6);
        assert_eq!(summary.added_bytes, 5);
        assert!(summary.preview.contains("- \"before\""));
        assert!(summary.preview.contains("+ \"after\""));
        assert!(!summary.preview_truncated);

        let noop = serde_json::json!({
            "path": "src/lib.rs",
            "expected_sha256": "ab".repeat(32),
            "edits": [{"old": "same", "new": "same"}],
        });
        assert!(summarize_apply_patch(&noop).is_err());
        let unknown = serde_json::json!({
            "path": "src/lib.rs",
            "expected_sha256": "ab".repeat(32),
            "edits": [{"old": "a", "new": "b"}],
            "approval": true,
        });
        assert!(summarize_apply_patch(&unknown).is_err());
    }

    #[test]
    fn delete_summary_is_bounded_before_environment_confirmation() {
        let summary = summarize_delete(&serde_json::json!({
            "path": r"C:\\work\\obsolete",
            "recursive": true,
        }))
        .unwrap();
        assert_eq!(summary.path, r"C:\\work\\obsolete");
        assert!(summary.recursive);
        assert!(summarize_delete(&serde_json::json!({"path": ""})).is_err());
        assert!(summarize_delete(
            &serde_json::json!({"path": "file", "recursive": false, "approved": true})
        )
        .is_err());
    }

    #[test]
    fn workspace_exec_summary_matches_the_direct_argv_policy() {
        let summary = summarize_workspace_exec(&serde_json::json!({
            "argv": ["cargo", "test", "--locked", "-p", "cq-infer"],
            "cwd": "crates/cq-infer",
            "timeout_ms": 45_000,
        }))
        .unwrap();
        assert_eq!(summary.argv[0], "cargo");
        assert_eq!(summary.cwd.as_deref(), Some("crates/cq-infer"));
        assert_eq!(summary.timeout_ms, Some(45_000));
        assert!(summary.detail.contains("bounded-argv-cwd-v1"));
        assert!(summary.detail.contains("os_sandbox=false"));
        assert!(summary.detail.len() < 4 * 1024);
        for bad in [
            serde_json::json!({"argv": ["cargo", "run"]}),
            serde_json::json!({"argv": ["cargo", "check", "../outside"]}),
            serde_json::json!({"argv": ["cargo", "check", "--manifest-path", "x/Cargo.toml"]}),
            serde_json::json!({"argv": ["powershell", "-Command", "cargo check"]}),
            serde_json::json!({"argv": ["cargo", "check"], "cwd": "../outside"}),
        ] {
            assert!(summarize_workspace_exec(&bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn file_lifecycle_summaries_are_bounded_and_reject_unknown_fields() {
        let write = summarize_write_file(&serde_json::json!({
            "path": "src/generated.rs",
            "content": "pub const VALUE: u32 = 7;\n",
            "create": true,
        }))
        .unwrap();
        assert_eq!(write.path, "src/generated.rs");
        assert!(write.create);
        assert_eq!(write.content_bytes, 26);
        assert_eq!(
            write.content_sha256,
            sha256_hex(b"pub const VALUE: u32 = 7;\n")
        );
        assert!(write.preview.contains("pub const VALUE"));
        assert!(!write.preview_truncated);
        assert!(summarize_write_file(&serde_json::json!({
            "path": "src/generated.rs",
            "content": "x",
            "create": true,
            "approved": true,
        }))
        .is_err());
        assert!(summarize_write_file(&serde_json::json!({
            "path": "src/generated.rs",
            "content": "x",
            "expected_sha256": "not-a-digest",
        }))
        .is_err());

        let mkdir = summarize_mkdir(&serde_json::json!({
            "path": "generated/deep",
            "recursive": true,
        }))
        .unwrap();
        assert_eq!(mkdir.operation, "mkdir");
        assert!(mkdir.detail.contains("recursive=true"));
        let rename = summarize_rename(&serde_json::json!({
            "from": "generated/old.rs",
            "to": "generated/new.rs",
        }))
        .unwrap();
        assert_eq!(rename.operation, "rename");
        assert!(rename.detail.contains("generated/old.rs"));
        assert!(rename.detail.contains("generated/new.rs"));
        assert!(summarize_mkdir(&serde_json::json!({
            "path": "generated",
            "recursive": true,
            "approval": "APPLY",
        }))
        .is_err());
        assert!(summarize_rename(&serde_json::json!({"from": "", "to": "new"})).is_err());
    }

    #[test]
    fn truncated_artifact_is_rehashed_before_the_receipt_is_accepted() {
        let workspace = std::env::temp_dir().join(format!(
            "cq-tool-artifact-client-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let directory = workspace.join(".tina/tool-artifacts");
        fs::create_dir_all(&directory).unwrap();
        let payload = b"complete artifact";
        let digest = sha256_hex(payload);
        fs::write(directory.join(format!("{digest}.txt")), payload).unwrap();
        let result = HostResult {
            kind: "text".to_string(),
            text: "complete".to_string(),
            bytes: payload.len(),
            sha256: digest.clone(),
            truncated: true,
            artifact: Some(format!(".tina/tool-artifacts/{digest}.txt")),
        };
        validate_host_result(&result, &workspace.canonicalize().unwrap()).unwrap();
        fs::write(directory.join(format!("{digest}.txt")), b"corrupt artifact").unwrap();
        assert!(validate_host_result(&result, &workspace.canonicalize().unwrap()).is_err());
        fs::remove_dir_all(workspace).unwrap();
    }

    #[test]
    fn real_host_client_handshake_and_read_when_binary_is_provided() {
        let Ok(binary) = std::env::var("CQ_TOOL_HOST_TEST_BINARY") else {
            return;
        };
        let workspace = std::env::temp_dir().join(format!(
            "cq-tool-host-client-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&workspace).unwrap();
        fs::write(workspace.join("sample.txt"), "alpha\nbeta\n").unwrap();
        fs::create_dir(workspace.join("src")).unwrap();
        fs::write(
            workspace.join("Cargo.toml"),
            "[package]\nname = \"client_exec_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::write(workspace.join("src/lib.rs"), "pub fn value() -> u8 { 1 }\n").unwrap();

        let mut client =
            ToolHostClient::spawn(Path::new(&binary), &workspace, false, "powershell", None)
                .unwrap();
        assert!(client.supports("read"));
        assert!(client.supports("apply_patch"));
        assert!(client.supports("workspace_exec"));
        assert!(!client.supports("capabilities"));
        assert!(!client.supports("shell"));
        let read = client
            .execute_read_only(
                "read-integration",
                "read",
                serde_json::json!({"path":"sample.txt","start_line":2,"end_line":2}),
            )
            .unwrap();
        assert!(read.ok);
        assert_eq!(read.stdout, "beta");
        let search = client
            .execute_read_only(
                "search-integration",
                "search",
                serde_json::json!({"query":"alpha"}),
            )
            .unwrap();
        assert!(search.ok);
        assert_eq!(search.stdout, "sample.txt:1:alpha\n");
        let denied_write = client
            .execute_read_only("patch-wrong-access", "apply_patch", serde_json::json!({}))
            .unwrap();
        assert!(!denied_write.ok);
        assert_eq!(
            fs::read_to_string(workspace.join("sample.txt")).unwrap(),
            "alpha\nbeta\n"
        );
        let patch = client
            .execute_approved_mutation(
                "patch-integration",
                "apply_patch",
                serde_json::json!({
                    "path": "sample.txt",
                    "expected_sha256": sha256_hex(b"alpha\nbeta\n"),
                    "edits": [{"old": "beta", "new": "gamma"}],
                }),
            )
            .unwrap();
        assert!(patch.ok, "{}", patch.stderr);
        assert_eq!(
            fs::read_to_string(workspace.join("sample.txt")).unwrap(),
            "alpha\ngamma\n"
        );
        let denied_read = client
            .execute_approved_mutation("read-wrong-access", "read", serde_json::json!({}))
            .unwrap();
        assert!(!denied_read.ok);
        let execution = client
            .execute_approved_execution_cancellable(
                "exec-integration",
                "workspace_exec",
                serde_json::json!({
                    "argv": ["cargo", "metadata", "--no-deps", "--format-version=1"],
                    "timeout_ms": 30_000,
                }),
                &mut || Ok(false),
            )
            .unwrap();
        assert!(execution.ok, "{}", execution.stderr);
        assert!(execution.stdout.contains("client_exec_fixture"));
        client.restart().unwrap();
        let read_after_restart = client
            .execute_read_only(
                "read-after-restart",
                "read",
                serde_json::json!({"path":"sample.txt"}),
            )
            .unwrap();
        assert!(read_after_restart.ok);
        assert_eq!(read_after_restart.stdout, "alpha\ngamma\n");
        client.shutdown().unwrap();

        assert!(workspace.starts_with(std::env::temp_dir()));
        fs::remove_dir_all(workspace).unwrap();
    }
}
