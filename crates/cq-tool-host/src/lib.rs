//! Independent, bounded tool host for the Tina Agent loop.
//!
//! The host is deliberately separate from `cq-infer`: it owns workspace I/O
//! and process boundaries, but it has no CUDA, tokenizer, or KV-cache
//! dependency. Requests and responses are one JSON object per line so the
//! parent process can kill/restart the host without corrupting model state.

#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::min;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const PROTOCOL_VERSION: u16 = 2;
pub const PROTOCOL_SCHEMA: &str = "tina.tool-host.v2";
pub const MAX_REQUEST_DEADLINE_MS: u64 = 5 * 60 * 1_000;
pub const DEFAULT_INLINE_OUTPUT_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_READ_BYTES: usize = 4 * 1024 * 1024;
pub const DEFAULT_MAX_LIST_ENTRIES: usize = 2_048;
pub const DEFAULT_MAX_SEARCH_MATCHES: usize = 512;
pub const DEFAULT_MAX_SEARCH_FILE_BYTES: usize = 1_048_576;
pub const DEFAULT_MAX_REQUEST_BYTES: usize = 256 * 1024;
pub const DEFAULT_MAX_LINE_RANGE: usize = 16_384;
pub const DEFAULT_MAX_PATCH_BYTES: usize = 192 * 1024;
pub const DEFAULT_MAX_PATCH_EDITS: usize = 128;
pub const DEFAULT_MAX_SHELL_COMMAND_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_SHELL_OUTPUT_BYTES: usize = 32 * 1024;
pub const DEFAULT_MAX_SHELL_TIMEOUT_MS: u64 = 120 * 1_000;
pub const DEFAULT_MAX_EXEC_ARGUMENT_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_EXEC_ARGS: usize = 64;
pub const DEFAULT_MAX_EXEC_TIMEOUT_MS: u64 = MAX_REQUEST_DEADLINE_MS;
pub const DEFAULT_MAX_WEB_SEARCH_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
pub const DEFAULT_MAX_WEB_SEARCH_RESULTS: usize = 10;
pub const DEFAULT_MAX_WEB_SEARCH_TIMEOUT_MS: u64 = 20 * 1_000;
pub const MAX_ERROR_MESSAGE_BYTES: usize = 16 * 1024;
const MAX_EXEC_ARG_BYTES_PER_ITEM: usize = 8 * 1024;
const MAX_CONFIGURED_REQUEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_WEB_SEARCH_QUERY_BYTES: usize = 4 * 1024;
const MAX_WEB_SEARCH_ENDPOINT_BYTES: usize = 2 * 1024;
const MAX_WEB_SEARCH_URL_BYTES: usize = 8 * 1024;
const MAX_WEB_SEARCH_TITLE_BYTES: usize = 512;
const MAX_WEB_SEARCH_SNIPPET_BYTES: usize = 2 * 1024;
static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolHostScope {
    Workspace,
    Machine,
}

impl ToolHostScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Machine => "machine",
        }
    }
}

/// Shell dialect selected by the parent CLI. Full-access commands are always
/// run through one explicit dialect so the model cannot silently mix Windows
/// and POSIX syntax.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolHostShell {
    PowerShell,
    Cmd,
    Bash,
    Zsh,
    Fish,
    Sh,
}

impl ToolHostShell {
    pub fn parse(value: &str) -> Result<Self, HostError> {
        let value = value.trim().to_ascii_lowercase();
        let name = Path::new(&value)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(&value);
        match name {
            "powershell" | "pwsh" => Ok(Self::PowerShell),
            "cmd" => Ok(Self::Cmd),
            "bash" => Ok(Self::Bash),
            "zsh" => Ok(Self::Zsh),
            "fish" => Ok(Self::Fish),
            "sh" | "dash" => Ok(Self::Sh),
            _ => Err(HostError::InvalidConfig(format!(
                "unsupported shell {value:?}; expected powershell, cmd, bash, zsh, fish, or sh"
            ))),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PowerShell => "powershell",
            Self::Cmd => "cmd",
            Self::Bash => "bash",
            Self::Zsh => "zsh",
            Self::Fish => "fish",
            Self::Sh => "sh",
        }
    }

    fn detect() -> Self {
        if let Ok(value) = std::env::var("TINA_SHELL") {
            if let Ok(shell) = Self::parse(&value) {
                return shell;
            }
        }
        #[cfg(windows)]
        {
            if std::env::var_os("PSModulePath").is_some() {
                Self::PowerShell
            } else {
                Self::Cmd
            }
        }
        #[cfg(not(windows))]
        {
            std::env::var("SHELL")
                .ok()
                .and_then(|value| Self::parse(&value).ok())
                .unwrap_or(Self::Sh)
        }
    }
}

/// Owns the process group/job used for one unrestricted shell call. A shell
/// command may launch grandchildren that inherit stdout/stderr; terminating
/// only the shell would leave those pipes open and block the receipt forever.
#[cfg(windows)]
struct OwnedWindowsHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl OwnedWindowsHandle {
    fn new(handle: windows_sys::Win32::Foundation::HANDLE) -> Result<Self, io::Error> {
        if handle.is_null() {
            Err(io::Error::last_os_error())
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
        // Drop is the sole close site, so every successful construction is
        // closed exactly once, including error paths.
        unsafe {
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
struct ShellProcessTree {
    handle: OwnedWindowsHandle,
}

#[cfg(unix)]
struct ShellProcessTree {
    process_group: i32,
}

#[cfg(not(any(unix, windows)))]
struct ShellProcessTree;

impl ShellProcessTree {
    fn attach(child: &Child) -> Result<Self, HostError> {
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
                SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            };
            use windows_sys::Win32::System::Threading::{
                OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SET_QUOTA,
                PROCESS_TERMINATE,
            };

            // SAFETY: null names request a private kernel object; no pointer
            // arguments are dereferenced by the call.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            let handle = OwnedWindowsHandle::new(handle)
                .map_err(|error| HostError::Tool(error.to_string()))?;
            // SAFETY: the Windows job-information structure is a plain C
            // value whose all-zero representation is valid; only its limit
            // flag is changed before the FFI call.
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `limits` is a valid initialized structure for the
            // duration of the call, and the handle belongs to this process.
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
                return Err(HostError::Tool(io::Error::last_os_error().to_string()));
            }
            // SAFETY: `child.id()` identifies the live child created by this
            // host; the requested rights are limited to job assignment and
            // querying/termination.
            let process = unsafe {
                OpenProcess(
                    PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
                    0,
                    child.id(),
                )
            };
            let process = OwnedWindowsHandle::new(process)
                .map_err(|error| HostError::Tool(error.to_string()))?;
            // SAFETY: both wrappers hold live handles; the pointed-to job and
            // process remain valid for the duration of this synchronous call.
            let assigned = unsafe { AssignProcessToJobObject(handle.as_raw(), process.as_raw()) };
            if assigned == 0 {
                return Err(HostError::Tool(io::Error::last_os_error().to_string()));
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
            // SAFETY: the handle is owned by this struct and remains valid
            // until Drop completes. The call is synchronous and does not
            // transfer ownership of the job handle.
            unsafe {
                let _ = TerminateJobObject(self.handle.as_raw(), 1);
            }
        }
        #[cfg(unix)]
        // SAFETY: spawn_shell places the child in a fresh process group;
        // the negative PID targets only that group, never this host.
        unsafe {
            let _ = libc::kill(-self.process_group, libc::SIGKILL);
        }
    }
}

impl Drop for ShellProcessTree {
    fn drop(&mut self) {
        self.terminate();
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolRequest {
    pub v: u16,
    pub kind: String,
    pub id: String,
    pub op: String,
    pub deadline_unix_ms: u64,
    #[serde(default)]
    pub args: Value,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CancelRequest {
    v: u16,
    kind: String,
    id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolErrorBody {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub retryable: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolResultBody {
    pub kind: String,
    pub text: String,
    pub bytes: usize,
    pub sha256: String,
    pub truncated: bool,
    pub artifact: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolResponse {
    pub v: u16,
    pub id: String,
    pub ok: bool,
    pub result: Option<ToolResultBody>,
    pub error: Option<ToolErrorBody>,
}

impl ToolResponse {
    fn success(id: String, result: ToolResultBody) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: String, code: impl Into<String>, message: impl Into<String>) -> Self {
        let message = truncate_utf8_owned(message.into(), MAX_ERROR_MESSAGE_BYTES);
        Self {
            v: PROTOCOL_VERSION,
            id,
            ok: false,
            result: None,
            error: Some(ToolErrorBody {
                code: code.into(),
                message,
                retryable: false,
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostLimits {
    pub inline_output_bytes: usize,
    pub max_read_bytes: usize,
    pub max_list_entries: usize,
    pub max_search_matches: usize,
    pub max_search_file_bytes: usize,
    pub max_request_bytes: usize,
    pub max_line_range: usize,
    pub max_patch_bytes: usize,
    pub max_patch_edits: usize,
    pub max_shell_command_bytes: usize,
    pub max_shell_output_bytes: usize,
    pub max_shell_timeout_ms: u64,
    pub max_exec_argument_bytes: usize,
    pub max_exec_args: usize,
    pub max_exec_timeout_ms: u64,
}

impl Default for HostLimits {
    fn default() -> Self {
        Self {
            inline_output_bytes: DEFAULT_INLINE_OUTPUT_BYTES,
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
            max_list_entries: DEFAULT_MAX_LIST_ENTRIES,
            max_search_matches: DEFAULT_MAX_SEARCH_MATCHES,
            max_search_file_bytes: DEFAULT_MAX_SEARCH_FILE_BYTES,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            max_line_range: DEFAULT_MAX_LINE_RANGE,
            max_patch_bytes: DEFAULT_MAX_PATCH_BYTES,
            max_patch_edits: DEFAULT_MAX_PATCH_EDITS,
            max_shell_command_bytes: DEFAULT_MAX_SHELL_COMMAND_BYTES,
            max_shell_output_bytes: DEFAULT_MAX_SHELL_OUTPUT_BYTES,
            max_shell_timeout_ms: DEFAULT_MAX_SHELL_TIMEOUT_MS,
            max_exec_argument_bytes: DEFAULT_MAX_EXEC_ARGUMENT_BYTES,
            max_exec_args: DEFAULT_MAX_EXEC_ARGS,
            max_exec_timeout_ms: DEFAULT_MAX_EXEC_TIMEOUT_MS,
        }
    }
}

impl HostLimits {
    fn validate(&self) -> Result<(), HostError> {
        if self.inline_output_bytes == 0
            || self.inline_output_bytes > 16 * 1024 * 1024
            || self.max_read_bytes == 0
            || self.max_read_bytes > 64 * 1024 * 1024
            || self.max_list_entries == 0
            || self.max_search_matches == 0
            || self.max_search_file_bytes == 0
            || self.max_request_bytes < 64
            || self.max_request_bytes > MAX_CONFIGURED_REQUEST_BYTES
            || self.max_line_range == 0
            || self.max_patch_bytes == 0
            || self.max_patch_bytes > self.max_request_bytes
            || self.max_patch_edits == 0
            || self.max_patch_edits > 1_024
            || self.max_shell_command_bytes == 0
            || self.max_shell_command_bytes > 1024 * 1024
            || self.max_shell_output_bytes == 0
            || self.max_shell_output_bytes > 64 * 1024 * 1024
            || self.max_shell_timeout_ms == 0
            || self.max_shell_timeout_ms > MAX_REQUEST_DEADLINE_MS
            || self.max_exec_argument_bytes == 0
            || self.max_exec_argument_bytes > 1024 * 1024
            || self.max_exec_args == 0
            || self.max_exec_args > 256
            || self.max_exec_timeout_ms == 0
            || self.max_exec_timeout_ms > MAX_REQUEST_DEADLINE_MS
        {
            return Err(HostError::InvalidConfig(
                "tool host limits are outside the bounded range".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum HostError {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidConfig(String),
    InvalidRequest(String),
    Protocol(String),
    Deadline(String),
    Workspace(String),
    StaleRead(String),
    PatchConflict(String),
    Tool(String),
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "tool host I/O failure: {e}"),
            Self::Json(e) => write!(f, "tool host JSON failure: {e}"),
            Self::InvalidConfig(e) => write!(f, "tool host configuration: {e}"),
            Self::InvalidRequest(e) => write!(f, "invalid tool request: {e}"),
            Self::Protocol(e) => write!(f, "tool protocol failure: {e}"),
            Self::Deadline(e) => write!(f, "request deadline exceeded: {e}"),
            Self::Workspace(e) => write!(f, "workspace denied: {e}"),
            Self::StaleRead(e) => write!(f, "stale read: {e}"),
            Self::PatchConflict(e) => write!(f, "patch conflict: {e}"),
            Self::Tool(e) => write!(f, "tool failure: {e}"),
        }
    }
}

impl std::error::Error for HostError {}

impl From<io::Error> for HostError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for HostError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone, Debug)]
struct WebSearchProvider {
    endpoint: String,
    endpoint_sha256: String,
}

#[derive(Clone, Debug)]
pub struct ToolHost {
    root: PathBuf,
    artifact_dir: PathBuf,
    limits: HostLimits,
    scope: ToolHostScope,
    shell: ToolHostShell,
    web_search: Option<WebSearchProvider>,
}

impl ToolHost {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, HostError> {
        Self::with_scope_and_limits(root, ToolHostScope::Workspace, HostLimits::default())
    }

    pub fn with_scope(root: impl AsRef<Path>, scope: ToolHostScope) -> Result<Self, HostError> {
        Self::with_scope_and_limits(root, scope, HostLimits::default())
    }

    pub fn with_limits(root: impl AsRef<Path>, limits: HostLimits) -> Result<Self, HostError> {
        Self::with_scope_and_limits(root, ToolHostScope::Workspace, limits)
    }

    pub fn with_scope_and_limits(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        limits: HostLimits,
    ) -> Result<Self, HostError> {
        Self::with_scope_limits_and_shell(root, scope, limits, ToolHostShell::detect())
    }

    pub fn with_scope_and_shell(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        shell: ToolHostShell,
    ) -> Result<Self, HostError> {
        Self::with_scope_limits_and_shell(root, scope, HostLimits::default(), shell)
    }

    pub fn with_scope_and_web_search_url(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        web_search_url: Option<&str>,
    ) -> Result<Self, HostError> {
        Self::with_scope_limits_shell_and_web_search(
            root,
            scope,
            HostLimits::default(),
            ToolHostShell::detect(),
            web_search_url,
        )
    }

    pub fn with_scope_limits_and_shell(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        limits: HostLimits,
        shell: ToolHostShell,
    ) -> Result<Self, HostError> {
        Self::with_scope_limits_shell_and_web_search(root, scope, limits, shell, None)
    }

    pub fn with_scope_shell_and_web_search_url(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        shell: ToolHostShell,
        web_search_url: Option<&str>,
    ) -> Result<Self, HostError> {
        Self::with_scope_limits_shell_and_web_search(
            root,
            scope,
            HostLimits::default(),
            shell,
            web_search_url,
        )
    }

    fn with_scope_limits_shell_and_web_search(
        root: impl AsRef<Path>,
        scope: ToolHostScope,
        limits: HostLimits,
        shell: ToolHostShell,
        web_search_url: Option<&str>,
    ) -> Result<Self, HostError> {
        limits.validate()?;
        let web_search = web_search_url
            .map(validate_web_search_endpoint)
            .transpose()?
            .map(|endpoint| WebSearchProvider {
                endpoint_sha256: sha256_hex(endpoint.as_bytes()),
                endpoint,
            });
        if web_search.is_some() {
            validate_web_search_client()?;
        }
        let root = fs::canonicalize(root.as_ref()).map_err(|error| {
            HostError::Workspace(format!("workspace root is not accessible: {error}"))
        })?;
        if !root.is_dir() {
            return Err(HostError::Workspace(format!(
                "workspace root is not a directory: {}",
                root.display()
            )));
        }
        let tina_dir = root.join(".tina");
        ensure_directory_without_symlink(&tina_dir)?;
        let tina_dir = fs::canonicalize(&tina_dir)?;
        if !tina_dir.starts_with(&root) {
            return Err(HostError::Workspace(
                "Tina state directory escapes the workspace root".to_string(),
            ));
        }
        let artifact_dir = tina_dir.join("tool-artifacts");
        ensure_directory_without_symlink(&artifact_dir)?;
        let artifact_dir = fs::canonicalize(&artifact_dir)?;
        if !artifact_dir.starts_with(&root) {
            return Err(HostError::Workspace(
                "tool artifact directory escapes the workspace root".to_string(),
            ));
        }
        Ok(Self {
            root,
            artifact_dir,
            limits,
            scope,
            shell,
            web_search,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn limits(&self) -> &HostLimits {
        &self.limits
    }

    pub fn scope(&self) -> ToolHostScope {
        self.scope
    }

    pub fn shell_dialect(&self) -> ToolHostShell {
        self.shell
    }

    /// Parse and dispatch one bounded JSONL request. A malformed request gets
    /// a protocol error response instead of terminating the host process.
    pub fn handle_line(&self, line: &str) -> String {
        let response = self.handle_line_result(line);
        serde_json::to_string(&response).unwrap_or_else(|error| {
            serde_json::json!({
                "v": PROTOCOL_VERSION,
                "id": "",
                "ok": false,
                "result": null,
                "error": {"code": "response_encode", "message": error.to_string(), "retryable": false}
            })
            .to_string()
        })
    }

    pub fn handle_line_result(&self, line: &str) -> ToolResponse {
        if line.len() > self.limits.max_request_bytes {
            return ToolResponse::error(
                String::new(),
                "request_too_large",
                format!(
                    "request is {} bytes; limit is {}",
                    line.len(),
                    self.limits.max_request_bytes
                ),
            );
        }
        let envelope: Value = match serde_json::from_str(line) {
            Ok(envelope) => envelope,
            Err(error) => {
                return ToolResponse::error(String::new(), "invalid_json", error.to_string())
            }
        };
        if envelope.get("kind").and_then(Value::as_str) == Some("cancel") {
            let cancel: CancelRequest = match serde_json::from_value(envelope) {
                Ok(cancel) => cancel,
                Err(error) => {
                    return ToolResponse::error(String::new(), "invalid_cancel", error.to_string())
                }
            };
            if cancel.v != PROTOCOL_VERSION
                || cancel.kind != "cancel"
                || !valid_request_id(&cancel.id)
            {
                return ToolResponse::error(
                    cancel.id,
                    "invalid_cancel",
                    "cancel frame version or request identity is invalid",
                );
            }
            return ToolResponse::error(
                cancel.id,
                "cancel_not_active",
                "the target request is no longer active",
            );
        }
        let request: ToolRequest = match serde_json::from_value(envelope) {
            Ok(request) => request,
            Err(error) => {
                return ToolResponse::error(String::new(), "invalid_request", error.to_string())
            }
        };
        let id = request.id.clone();
        if request.v != PROTOCOL_VERSION {
            return ToolResponse::error(
                id,
                "unsupported_version",
                format!("request version {} is not supported", request.v),
            );
        }
        if !valid_request_id(&request.id) {
            return ToolResponse::error(
                request.id,
                "invalid_id",
                "request id must be 1..=128 printable bytes",
            );
        }
        if request.kind != "request" {
            return ToolResponse::error(id, "invalid_request", "request kind must be 'request'");
        }
        if let Err(error) = validate_deadline(request.deadline_unix_ms) {
            return ToolResponse::error(id, error_code(&error), error.to_string());
        }
        match self.dispatch(&request) {
            Ok(body) => ToolResponse::success(id, body),
            Err(error) => ToolResponse::error(id, error_code(&error), error.to_string()),
        }
    }

    fn dispatch(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        match request.op.as_str() {
            "capabilities" => self.capabilities(),
            "read" => self.read(request),
            "list" => self.list(request),
            "search" => self.search(request),
            "web_search" => self.web_search(request),
            "git_status" => self.git(request, false),
            "git_diff" => self.git(request, true),
            "apply_patch" => self.apply_patch(request),
            "write_file" => self.write_file(request),
            "mkdir" => self.mkdir(request),
            "delete" => self.delete(request),
            "rename" => self.rename(request),
            "workspace_exec" => self.workspace_exec(request),
            "shell" => self.shell(request),
            other => Err(HostError::Tool(format!("unsupported operation {other:?}"))),
        }
    }

    fn capabilities(&self) -> Result<ToolResultBody, HostError> {
        self.result_from_text(
            "json",
            serde_json::json!({
                "schema": PROTOCOL_SCHEMA,
                "version": PROTOCOL_VERSION,
                "scope": self.scope.as_str(),
                "tools": self.capability_names(),
                "writes": true,
                "network": self.scope == ToolHostScope::Machine,
                "web_search": self.web_search.is_some(),
                "shell": self.scope == ToolHostScope::Machine,
                "shell_dialect": self.shell.as_str(),
                "cancellation": true,
                "max_deadline_ms": MAX_REQUEST_DEADLINE_MS,
                "workspace_exec_policy": "bounded-argv-cwd-v1",
                "workspace_exec_os_sandbox": false,
                "workspace_exec_inherited_environment": true,
                "workspace": self.root.display().to_string(),
            })
            .to_string(),
        )
    }

    fn capability_names(&self) -> Vec<&'static str> {
        let mut tools = vec![
            "capabilities",
            "read",
            "list",
            "search",
            "git_status",
            "git_diff",
            "apply_patch",
            "write_file",
            "mkdir",
            "delete",
            "rename",
            "workspace_exec",
        ];
        if self.web_search.is_some() {
            tools.push("web_search");
        }
        if self.scope == ToolHostScope::Machine {
            tools.push("shell");
        }
        tools
    }

    fn web_search(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let provider = self.web_search.as_ref().ok_or_else(|| {
            HostError::Tool("web_search is unavailable because no provider is configured".into())
        })?;
        let args: WebSearchArgs = parse_args(&request.args, "web_search")?;
        let query = args.query.trim();
        if query.is_empty()
            || query.len() > MAX_WEB_SEARCH_QUERY_BYTES
            || query.chars().any(char::is_control)
        {
            return Err(HostError::InvalidRequest(format!(
                "web_search query must be 1..={MAX_WEB_SEARCH_QUERY_BYTES} bytes and contain no control characters"
            )));
        }
        let max_results = args.max_results.unwrap_or(DEFAULT_MAX_WEB_SEARCH_RESULTS);
        if max_results == 0 || max_results > DEFAULT_MAX_WEB_SEARCH_RESULTS {
            return Err(HostError::InvalidRequest(format!(
                "web_search max_results must be 1..={DEFAULT_MAX_WEB_SEARCH_RESULTS}"
            )));
        }
        let response = fetch_searxng_json(provider, query, request.deadline_unix_ms)?;
        let receipt = normalize_searxng_response(provider, query, max_results, &response)?;
        self.result_from_text("json", receipt)
    }

    fn read(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: ReadArgs = parse_args(&request.args, "read")?;
        let path = self.resolve_existing(&args.path, false)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() {
            return Err(HostError::Tool(format!(
                "path is not a file: {}",
                args.path
            )));
        }
        if metadata.len() > self.limits.max_read_bytes as u64 {
            return Err(HostError::Tool(format!(
                "file is {} bytes; read limit is {}",
                metadata.len(),
                self.limits.max_read_bytes
            )));
        }
        let mut text = String::new();
        File::open(&path)?
            .read_to_string(&mut text)
            .map_err(|error| {
                HostError::Tool(format!(
                    "file is not valid UTF-8 or could not be read: {error}"
                ))
            })?;
        let start = args.start_line.unwrap_or(1).max(1);
        let end = args.end_line.unwrap_or(usize::MAX);
        if end < start
            || (end != usize::MAX && end.saturating_sub(start) > self.limits.max_line_range)
        {
            return Err(HostError::Tool(format!(
                "line range {start}..={end} exceeds the bounded range"
            )));
        }
        if start != 1 || end != usize::MAX {
            text = text
                .lines()
                .enumerate()
                .filter_map(|(index, line)| {
                    let line_number = index + 1;
                    (line_number >= start && line_number <= end).then_some(line)
                })
                .collect::<Vec<_>>()
                .join("\n");
        }
        self.result_from_text("text", text)
    }

    fn list(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: ListArgs = parse_args(&request.args, "list")?;
        let path = self.resolve_existing(args.path.as_deref().unwrap_or("."), true)?;
        if !path.is_dir() {
            return Err(HostError::Tool(format!(
                "path is not a directory: {}",
                path.display()
            )));
        }
        let limit = args
            .max_entries
            .unwrap_or(self.limits.max_list_entries)
            .min(self.limits.max_list_entries);
        let mut entries = Vec::new();
        if args.recursive.unwrap_or(false) {
            let mut pending = vec![path.clone()];
            while let Some(directory) = pending.pop() {
                let mut children = read_dir_sorted(&directory)?;
                children.reverse();
                for child in children {
                    let child_path = child.path();
                    let metadata = fs::symlink_metadata(&child_path)?;
                    if metadata_is_link_like(&metadata) {
                        continue;
                    }
                    if should_skip_directory(&child_path, &self.root) {
                        continue;
                    }
                    entries.push(self.relative_path(&child_path));
                    if entries.len() >= limit {
                        break;
                    }
                    if metadata.is_dir() {
                        pending.push(child_path);
                    }
                }
                if entries.len() >= limit {
                    break;
                }
            }
        } else {
            for child in read_dir_sorted(&path)?.into_iter().take(limit) {
                let child_path = child.path();
                if metadata_is_link_like(&fs::symlink_metadata(&child_path)?) {
                    continue;
                }
                entries.push(self.relative_path(&child_path));
            }
        }
        entries.sort();
        let mut text = entries.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        self.result_from_text("text", text)
    }

    fn search(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: SearchArgs = parse_args(&request.args, "search")?;
        if args.query.trim().is_empty() || args.query.len() > 256 {
            return Err(HostError::Tool(
                "search query must be 1..=256 bytes".to_string(),
            ));
        }
        let root = self.resolve_existing(args.path.as_deref().unwrap_or("."), true)?;
        if !root.is_dir() {
            return Err(HostError::Tool(
                "search path must be a directory".to_string(),
            ));
        }
        let limit = args
            .max_matches
            .unwrap_or(self.limits.max_search_matches)
            .min(self.limits.max_search_matches);
        let mut files = Vec::new();
        collect_files(
            &root,
            &self.root,
            &mut files,
            limit.saturating_mul(16).max(16),
        )?;
        files.sort();
        let mut matches = Vec::new();
        for path in files {
            if matches.len() >= limit {
                break;
            }
            let metadata = fs::metadata(&path)?;
            if metadata.len() > self.limits.max_search_file_bytes as u64 {
                continue;
            }
            let mut text = String::new();
            if File::open(&path)
                .and_then(|mut file| file.read_to_string(&mut text))
                .is_err()
            {
                continue;
            }
            for (index, line) in text.lines().enumerate() {
                if line.contains(&args.query) {
                    matches.push(format!(
                        "{}:{}:{}",
                        self.relative_path(&path),
                        index + 1,
                        line
                    ));
                    if matches.len() >= limit {
                        break;
                    }
                }
            }
        }
        let mut text = matches.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        self.result_from_text("text", text)
    }

    fn git(&self, request: &ToolRequest, diff: bool) -> Result<ToolResultBody, HostError> {
        let args: GitArgs =
            parse_args(&request.args, if diff { "git_diff" } else { "git_status" })?;
        let git_root = self.resolve_existing(args.path.as_deref().unwrap_or("."), true)?;
        if !git_root.is_dir() {
            return Err(HostError::Tool("git path must be a directory".to_string()));
        }
        let mut command = Command::new("git");
        command.current_dir(&git_root);
        command.arg("--no-pager");
        if diff {
            command.args(["diff", "--no-ext-diff", "--no-color"]);
            if args.cached.unwrap_or(false) {
                command.arg("--cached");
            }
        } else {
            command.args(["status", "--short", "--branch", "--untracked-files=normal"]);
        }
        let output = command.output().map_err(|error| {
            HostError::Tool(format!("failed to run git read-only command: {error}"))
        })?;
        let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
        if !output.stderr.is_empty() {
            text.push_str("\nstderr: ");
            text.push_str(&String::from_utf8_lossy(&output.stderr));
        }
        if !output.status.success() {
            return Err(HostError::Tool(format!(
                "git command exited with {}: {}",
                output.status,
                text.trim()
            )));
        }
        self.result_from_text("text", text)
    }

    fn apply_patch(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: ApplyPatchArgs = parse_args(&request.args, "apply_patch")?;
        let encoded_bytes = serde_json::to_vec(&request.args)?.len();
        if encoded_bytes > self.limits.max_patch_bytes {
            return Err(HostError::InvalidRequest(format!(
                "apply_patch arguments are {encoded_bytes} bytes; limit is {}",
                self.limits.max_patch_bytes
            )));
        }
        if !valid_sha256(&args.expected_sha256) {
            return Err(HostError::InvalidRequest(
                "expected_sha256 must be 64 lowercase hexadecimal characters".to_string(),
            ));
        }
        if args.edits.is_empty() || args.edits.len() > self.limits.max_patch_edits {
            return Err(HostError::InvalidRequest(format!(
                "apply_patch requires 1..={} exact edits",
                self.limits.max_patch_edits
            )));
        }

        let path = self.resolve_patch_target(&args.path)?;
        let metadata = fs::metadata(&path)?;
        if !metadata.is_file() {
            return Err(HostError::Tool(format!(
                "patch target is not a regular file: {}",
                args.path
            )));
        }
        if metadata.len() > self.limits.max_read_bytes as u64 {
            return Err(HostError::Tool(format!(
                "patch target is {} bytes; file limit is {}",
                metadata.len(),
                self.limits.max_read_bytes
            )));
        }

        let original_bytes = fs::read(&path)?;
        let original = std::str::from_utf8(&original_bytes).map_err(|error| {
            HostError::Tool(format!("patch target is not valid UTF-8: {error}"))
        })?;
        let actual_sha256 = sha256_hex(&original_bytes);
        if actual_sha256 != args.expected_sha256 {
            return Err(HostError::StaleRead(format!(
                "{} has SHA-256 {actual_sha256}, expected {}",
                args.path, args.expected_sha256
            )));
        }

        let patched = apply_exact_replacements(original, &args.edits)?;
        if patched.is_empty() {
            return Err(HostError::PatchConflict(
                "apply_patch cannot empty a file; file deletion is a separate destructive operation"
                    .to_string(),
            ));
        }
        if patched.len() > self.limits.max_read_bytes {
            return Err(HostError::PatchConflict(format!(
                "patched file would be {} bytes; file limit is {}",
                patched.len(),
                self.limits.max_read_bytes
            )));
        }

        let temporary = TemporaryReplacement::create(
            path.parent().ok_or_else(|| {
                HostError::Workspace("patch target has no parent directory".to_string())
            })?,
            Some(metadata.permissions()),
            patched.as_bytes(),
        )?;

        // Cancellation/deadline checks must happen before the irreversible
        // atomic replacement. Once commit starts, the host reports its real
        // terminal result rather than pretending the write did not happen.
        validate_deadline(request.deadline_unix_ms)?;

        // Re-resolve and re-read immediately before the atomic replacement so
        // an editor racing the stale-read check cannot be silently overwritten.
        let final_path = self.resolve_patch_target(&args.path).map_err(|error| {
            HostError::StaleRead(format!("patch target changed before commit: {error}"))
        })?;
        if final_path != path {
            return Err(HostError::StaleRead(
                "patch target identity changed before commit".to_string(),
            ));
        }
        let current_bytes = fs::read(&final_path)?;
        if current_bytes != original_bytes || sha256_hex(&current_bytes) != args.expected_sha256 {
            return Err(HostError::StaleRead(
                "patch target changed before atomic commit".to_string(),
            ));
        }
        temporary.commit(&final_path)?;

        let after_sha256 = sha256_hex(patched.as_bytes());
        self.result_from_text(
            "json",
            serde_json::json!({
                "path": self.relative_path(&final_path),
                "before_sha256": args.expected_sha256,
                "after_sha256": after_sha256,
                "before_bytes": original_bytes.len(),
                "after_bytes": patched.len(),
                "edits_applied": args.edits.len(),
            })
            .to_string(),
        )
    }

    fn write_file(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: WriteFileArgs = parse_args(&request.args, "write_file")?;
        if args.content.len() > self.limits.max_read_bytes {
            return Err(HostError::InvalidRequest(format!(
                "write_file content is {} bytes; file limit is {}",
                args.content.len(),
                self.limits.max_read_bytes
            )));
        }
        let path = self.resolve_write_target(&args.path)?;
        if self.is_workspace_root_or_control_path(&path) {
            return Err(HostError::Workspace(
                "write_file cannot modify Tina host control data".to_string(),
            ));
        }
        let existed = path.exists();
        if let Some(expected) = args.expected_sha256.as_deref() {
            if !valid_sha256(expected) {
                return Err(HostError::InvalidRequest(
                    "expected_sha256 must be 64 lowercase hexadecimal characters".to_string(),
                ));
            }
            if !existed {
                return Err(HostError::StaleRead(
                    "write_file expected an existing file but the target is absent".to_string(),
                ));
            }
            let current = fs::read(&path)?;
            let actual = sha256_hex(&current);
            if actual != expected {
                return Err(HostError::StaleRead(format!(
                    "{} has SHA-256 {actual}, expected {expected}",
                    args.path
                )));
            }
        }
        if !existed && !args.create {
            return Err(HostError::Tool(
                "write_file target does not exist; set create=true to create it".to_string(),
            ));
        }
        if existed && !fs::metadata(&path)?.is_file() {
            return Err(HostError::Tool(format!(
                "write_file target is not a regular file: {}",
                args.path
            )));
        }
        let permissions = if existed {
            Some(fs::metadata(&path)?.permissions())
        } else {
            None
        };
        validate_deadline(request.deadline_unix_ms)?;
        let parent = path.parent().ok_or_else(|| {
            HostError::Workspace("write_file target has no parent directory".to_string())
        })?;
        let temporary = TemporaryReplacement::create(parent, permissions, args.content.as_bytes())?;
        if existed {
            temporary.commit(&path)?;
        } else {
            temporary.commit_new(&path)?;
        }
        let digest = sha256_hex(args.content.as_bytes());
        self.result_from_text(
            "json",
            serde_json::json!({
                "path": self.relative_path(&path),
                "created": !existed,
                "bytes": args.content.len(),
                "sha256": digest,
            })
            .to_string(),
        )
    }

    fn mkdir(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: MkdirArgs = parse_args(&request.args, "mkdir")?;
        let path = self.resolve_write_target(&args.path)?;
        if self.is_workspace_root_or_control_path(&path) {
            return Err(HostError::Workspace(
                "mkdir cannot modify Tina host control data".to_string(),
            ));
        }
        if path.exists() {
            return Err(HostError::Tool(format!(
                "directory already exists: {}",
                args.path
            )));
        }
        validate_deadline(request.deadline_unix_ms)?;
        if args.recursive {
            fs::create_dir_all(&path)?;
        } else {
            fs::create_dir(&path)?;
        }
        self.result_from_text(
            "json",
            serde_json::json!({
                "path": self.relative_path(&path),
                "recursive": args.recursive,
            })
            .to_string(),
        )
    }

    fn delete(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: DeleteArgs = parse_args(&request.args, "delete")?;
        let input = self.resolve_input_path(&args.path)?;
        let path = if self.scope == ToolHostScope::Workspace {
            // Workspace deletion must not follow an intermediate link or
            // junction into a path the bounded host does not own.
            self.resolve_existing(&args.path, false)?
        } else {
            input
        };
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            HostError::Workspace(format!("path {:?} is not accessible: {error}", args.path))
        })?;
        if self.is_workspace_root_or_control_path(&path) {
            return Err(HostError::Workspace(
                "delete cannot remove the Agent anchor or Tina host control data".to_string(),
            ));
        }
        validate_deadline(request.deadline_unix_ms)?;
        let is_directory = metadata.is_dir() && !metadata_is_link_like(&metadata);
        if is_directory {
            if !args.recursive {
                return Err(HostError::Tool(
                    "delete of a directory requires recursive=true".to_string(),
                ));
            }
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
        self.result_from_text(
            "json",
            serde_json::json!({
                "path": self.relative_path(&path),
                "directory": is_directory,
                "recursive": args.recursive,
            })
            .to_string(),
        )
    }

    fn rename(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: RenameArgs = parse_args(&request.args, "rename")?;
        let source_input = self.resolve_input_path(&args.from)?;
        if self.is_workspace_root_or_control_path(&source_input) {
            return Err(HostError::Workspace(
                "rename cannot modify Tina host control data".to_string(),
            ));
        }
        let source = self.resolve_existing(&args.from, false)?;
        let destination = self.resolve_write_target(&args.to)?;
        if self.is_workspace_root_or_control_path(&source)
            || self.is_workspace_root_or_control_path(&destination)
        {
            return Err(HostError::Workspace(
                "rename cannot modify Tina host control data".to_string(),
            ));
        }
        if destination.exists() {
            return Err(HostError::Tool(
                "rename destination already exists; delete it explicitly first".to_string(),
            ));
        }
        validate_deadline(request.deadline_unix_ms)?;
        fs::rename(&source, &destination)?;
        self.result_from_text(
            "json",
            serde_json::json!({
                "from": self.relative_path(&source),
                "to": self.relative_path(&destination),
            })
            .to_string(),
        )
    }

    /// Run a bounded, direct-argv validation command from inside the workspace.
    ///
    /// This is an approval-gated development convenience, not an operating
    /// system sandbox: Cargo build scripts and tests inherit the host account,
    /// environment, and network permissions. The receipt states that fact so
    /// callers cannot present this as isolation.
    fn workspace_exec(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        let args: WorkspaceExecArgs = parse_args(&request.args, "workspace_exec")?;
        validate_workspace_exec_argv(
            &args.argv,
            self.limits.max_exec_args,
            self.limits.max_exec_argument_bytes,
        )?;
        let cwd = self.resolve_workspace_exec_cwd(args.cwd.as_deref())?;
        let requested_timeout = args.timeout_ms.unwrap_or(self.limits.max_exec_timeout_ms);
        if requested_timeout == 0 || requested_timeout > self.limits.max_exec_timeout_ms {
            return Err(HostError::InvalidRequest(format!(
                "workspace_exec timeout must be 1..={} ms",
                self.limits.max_exec_timeout_ms
            )));
        }
        validate_deadline(request.deadline_unix_ms)?;
        let timeout_ms =
            requested_timeout.min(request.deadline_unix_ms.saturating_sub(unix_millis()?));
        if timeout_ms == 0 {
            return Err(HostError::Deadline(
                "workspace_exec request has no remaining execution time".to_string(),
            ));
        }

        let mut child = self.spawn_workspace_exec(&args.argv, &cwd)?;
        let process_tree = match ShellProcessTree::attach(&child) {
            Ok(process_tree) => process_tree,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let stdout = child.stdout.take().ok_or_else(|| {
            process_tree.terminate();
            let _ = child.kill();
            HostError::Tool("workspace_exec child omitted stdout capture".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            process_tree.terminate();
            let _ = child.kill();
            HostError::Tool("workspace_exec child omitted stderr capture".to_string())
        })?;
        let output_limit = self.limits.max_shell_output_bytes;
        let stdout_reader = thread::spawn(move || read_process_output(stdout, output_limit));
        let stderr_reader = thread::spawn(move || read_process_output(stderr, output_limit));
        let started = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let mut timed_out = false;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= timeout {
                timed_out = true;
                process_tree.terminate();
                let _ = child.kill();
                break child.wait()?;
            }
            thread::sleep(Duration::from_millis(10));
        };
        process_tree.terminate();
        let stdout_capture = stdout_reader
            .join()
            .map_err(|_| HostError::Tool("workspace_exec stdout reader panicked".to_string()))??;
        let stderr_capture = stderr_reader
            .join()
            .map_err(|_| HostError::Tool("workspace_exec stderr reader panicked".to_string()))??;
        let receipt = ExecReceipt {
            argv: args.argv,
            cwd: cwd.display().to_string(),
            exit_code: status.code(),
            success: status.success() && !timed_out,
            timed_out,
            stdout: String::from_utf8_lossy(&stdout_capture.bytes).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_capture.bytes).into_owned(),
            stdout_bytes: stdout_capture.total_bytes,
            stderr_bytes: stderr_capture.total_bytes,
            stdout_captured_bytes: stdout_capture.bytes.len(),
            stderr_captured_bytes: stderr_capture.bytes.len(),
            stdout_truncated: stdout_capture.truncated,
            stderr_truncated: stderr_capture.truncated,
            stdout_inline_truncated: false,
            stderr_inline_truncated: false,
            policy: "bounded-argv-cwd-v1",
            os_sandbox: false,
            inherited_environment: true,
            network_access: "unrestricted",
        };
        let mut text = serde_json::to_string(&receipt)?;
        if text.len() > self.limits.inline_output_bytes {
            let reduced_limit = self
                .limits
                .max_shell_output_bytes
                .min(self.limits.inline_output_bytes / 8)
                .max(1);
            let stdout_len = receipt.stdout.len();
            let stderr_len = receipt.stderr.len();
            let receipt = ExecReceipt {
                stdout: truncate_utf8_owned(receipt.stdout, reduced_limit),
                stderr: truncate_utf8_owned(receipt.stderr, reduced_limit),
                stdout_inline_truncated: stdout_len > reduced_limit,
                stderr_inline_truncated: stderr_len > reduced_limit,
                ..receipt
            };
            text = serde_json::to_string(&receipt)?;
            if text.len() > self.limits.inline_output_bytes {
                return Err(HostError::Tool(
                    "workspace_exec receipt exceeded the inline protocol limit".to_string(),
                ));
            }
        }
        self.result_from_text("exec", text)
    }

    fn resolve_workspace_exec_cwd(&self, raw: Option<&str>) -> Result<PathBuf, HostError> {
        let raw = raw.unwrap_or(".");
        let input = validate_path(raw, ToolHostScope::Workspace)?;
        let candidate = self.root.join(input);
        reject_symlink_components(&self.root, &candidate)?;
        let canonical = fs::canonicalize(&candidate).map_err(|error| {
            HostError::Workspace(format!(
                "workspace_exec cwd {raw:?} is not accessible: {error}"
            ))
        })?;
        let under_vcs_control = canonical
            .strip_prefix(&self.root)
            .ok()
            .is_some_and(|relative| {
                relative.components().any(|component| {
                    matches!(component, Component::Normal(name) if name.eq_ignore_ascii_case(".git"))
                })
            });
        if !canonical.starts_with(&self.root)
            || self.is_host_control_path(&canonical)
            || under_vcs_control
        {
            return Err(HostError::Workspace(
                "workspace_exec cwd must remain inside the editable workspace".to_string(),
            ));
        }
        if !canonical.is_dir() {
            return Err(HostError::Tool(format!(
                "workspace_exec cwd is not a directory: {}",
                canonical.display()
            )));
        }
        Ok(canonical)
    }

    fn spawn_workspace_exec(&self, argv: &[String], cwd: &Path) -> Result<Child, HostError> {
        let mut process = Command::new(&argv[0]);
        process
            .args(&argv[1..])
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            process.process_group(0);
        }
        process.spawn().map_err(|error| {
            HostError::Tool(format!(
                "could not start workspace_exec program {:?}: {error}",
                argv[0]
            ))
        })
    }

    /// Execute one user-authorized command through the selected host shell.
    /// This operation is deliberately unreachable in workspace scope. In
    /// machine scope the parent has already shown the explicit FULL_ACCESS
    /// warning; the host still bounds command size, wall time, and captured
    /// output so a runaway process cannot fill the JSONL channel.
    fn shell(&self, request: &ToolRequest) -> Result<ToolResultBody, HostError> {
        self.require_machine_scope("shell")?;
        let args: ShellArgs = parse_args(&request.args, "shell")?;
        if args.command.trim().is_empty() || args.command.as_bytes().contains(&0) {
            return Err(HostError::InvalidRequest(
                "shell command is empty or contains NUL".to_string(),
            ));
        }
        if args.command.len() > self.limits.max_shell_command_bytes {
            return Err(HostError::InvalidRequest(format!(
                "shell command is {} bytes; limit is {}",
                args.command.len(),
                self.limits.max_shell_command_bytes
            )));
        }
        let cwd = self.resolve_command_cwd(args.cwd.as_deref())?;
        let requested_timeout = args.timeout_ms.unwrap_or(self.limits.max_shell_timeout_ms);
        if requested_timeout == 0 || requested_timeout > self.limits.max_shell_timeout_ms {
            return Err(HostError::InvalidRequest(format!(
                "shell timeout must be 1..={} ms",
                self.limits.max_shell_timeout_ms
            )));
        }
        validate_deadline(request.deadline_unix_ms)?;
        let now = unix_millis()?;
        let deadline_remaining = request.deadline_unix_ms.saturating_sub(now);
        let timeout_ms = requested_timeout.min(deadline_remaining);
        if timeout_ms == 0 {
            return Err(HostError::Deadline(
                "shell request has no remaining execution time".to_string(),
            ));
        }

        let mut child = self.spawn_shell(&args.command, &cwd)?;
        let process_tree = match ShellProcessTree::attach(&child) {
            Ok(process_tree) => process_tree,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
        };
        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                process_tree.terminate();
                let _ = child.kill();
                let _ = child.wait();
                return Err(HostError::Tool(
                    "shell child omitted stdout capture".to_string(),
                ));
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                process_tree.terminate();
                let _ = child.kill();
                let _ = child.wait();
                return Err(HostError::Tool(
                    "shell child omitted stderr capture".to_string(),
                ));
            }
        };
        let output_limit = self.limits.max_shell_output_bytes;
        let stdout_reader = thread::spawn(move || read_process_output(stdout, output_limit));
        let stderr_reader = thread::spawn(move || read_process_output(stderr, output_limit));
        let started = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let mut timed_out = false;
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() >= timeout {
                timed_out = true;
                process_tree.terminate();
                let _ = child.kill();
                break child.wait()?;
            }
            thread::sleep(Duration::from_millis(10));
        };
        // A command can exit while a background descendant still owns one of
        // the inherited pipes. Reclaim the complete command tree before
        // joining the drainers, otherwise a normal shell exit can hang here.
        process_tree.terminate();
        let stdout_capture = stdout_reader
            .join()
            .map_err(|_| HostError::Tool("shell stdout reader panicked".to_string()))??;
        let stderr_capture = stderr_reader
            .join()
            .map_err(|_| HostError::Tool("shell stderr reader panicked".to_string()))??;
        let stdout_text = String::from_utf8_lossy(&stdout_capture.bytes).into_owned();
        let stderr_text = String::from_utf8_lossy(&stderr_capture.bytes).into_owned();
        let receipt = ShellReceipt {
            shell: self.shell.as_str(),
            cwd: cwd.display().to_string(),
            exit_code: status.code(),
            success: status.success() && !timed_out,
            timed_out,
            stdout: stdout_text,
            stderr: stderr_text,
            stdout_bytes: stdout_capture.total_bytes,
            stderr_bytes: stderr_capture.total_bytes,
            stdout_captured_bytes: stdout_capture.bytes.len(),
            stderr_captured_bytes: stderr_capture.bytes.len(),
            stdout_truncated: stdout_capture.truncated,
            stderr_truncated: stderr_capture.truncated,
            stdout_inline_truncated: false,
            stderr_inline_truncated: false,
        };
        let mut text = serde_json::to_string(&receipt)?;
        // Keep the structured receipt inline even when a shell emits a large
        // amount of Unicode or control characters that expand during JSON
        // encoding. The truncation flags preserve the loss information.
        if text.len() > self.limits.inline_output_bytes {
            let reduced_limit = self
                .limits
                .max_shell_output_bytes
                .min(self.limits.inline_output_bytes / 8)
                .max(1);
            let stdout_len = receipt.stdout.len();
            let stderr_len = receipt.stderr.len();
            let receipt = ShellReceipt {
                stdout: truncate_utf8_owned(receipt.stdout, reduced_limit),
                stderr: truncate_utf8_owned(receipt.stderr, reduced_limit),
                stdout_inline_truncated: receipt.stdout_inline_truncated
                    || stdout_len > reduced_limit,
                stderr_inline_truncated: receipt.stderr_inline_truncated
                    || stderr_len > reduced_limit,
                ..receipt
            };
            text = serde_json::to_string(&receipt)?;
            if text.len() > self.limits.inline_output_bytes {
                return Err(HostError::Tool(
                    "shell receipt exceeded the inline protocol limit".to_string(),
                ));
            }
        }
        self.result_from_text("shell", text)
    }

    fn resolve_command_cwd(&self, raw: Option<&str>) -> Result<PathBuf, HostError> {
        let path = match raw {
            Some(raw) => self.resolve_existing(raw, true)?,
            None => self.root.clone(),
        };
        if self.is_host_control_path(&path) {
            return Err(HostError::Workspace(
                "shell cwd cannot be Tina host control data".to_string(),
            ));
        }
        if !path.is_dir() {
            return Err(HostError::Tool(format!(
                "shell cwd is not a directory: {}",
                path.display()
            )));
        }
        Ok(path)
    }

    fn spawn_shell(&self, command: &str, cwd: &Path) -> Result<Child, HostError> {
        let mut process = match self.shell {
            ToolHostShell::PowerShell => {
                let mut process = Command::new(if cfg!(windows) {
                    "powershell.exe"
                } else {
                    "pwsh"
                });
                process.args([
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                ]);
                process
            }
            ToolHostShell::Cmd => {
                let mut process = Command::new(if cfg!(windows) { "cmd.exe" } else { "cmd" });
                process.args(["/D", "/S", "/C", command]);
                process
            }
            ToolHostShell::Bash => {
                let mut process = Command::new("bash");
                process.args(["-lc", command]);
                process
            }
            ToolHostShell::Zsh => {
                let mut process = Command::new("zsh");
                process.args(["-lc", command]);
                process
            }
            ToolHostShell::Fish => {
                let mut process = Command::new("fish");
                process.args(["-c", command]);
                process
            }
            ToolHostShell::Sh => {
                let mut process = Command::new(if cfg!(windows) { "sh.exe" } else { "sh" });
                process.args(["-c", command]);
                process
            }
        };
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Put every shell invocation in its own group so timeout and
            // Drop can reclaim descendants rather than only the shell.
            process.process_group(0);
        }
        process
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                HostError::Tool(format!("could not start {}: {error}", self.shell.as_str()))
            })
    }

    fn require_machine_scope(&self, operation: &str) -> Result<(), HostError> {
        if self.scope != ToolHostScope::Machine {
            return Err(HostError::Workspace(format!(
                "{operation} requires --full-access"
            )));
        }
        Ok(())
    }

    fn resolve_patch_target(&self, raw: &str) -> Result<PathBuf, HostError> {
        let path = self.resolve_existing(raw, false)?;
        if self.is_host_control_path(&path)
            || (self.scope == ToolHostScope::Workspace
                && path.strip_prefix(&self.root).ok().is_some_and(|relative| {
                    relative.components().any(|component| {
                        let Component::Normal(name) = component else {
                            return false;
                        };
                        name.eq_ignore_ascii_case(".git")
                    })
                }))
        {
            return Err(HostError::Workspace(
                "apply_patch cannot modify host control data".to_string(),
            ));
        }
        #[cfg(windows)]
        if windows_path_has_alternate_data_stream(raw) {
            return Err(HostError::Workspace(
                "Windows alternate data streams are not accepted".to_string(),
            ));
        }
        Ok(path)
    }

    fn resolve_existing(&self, raw: &str, allow_root: bool) -> Result<PathBuf, HostError> {
        let candidate = self.resolve_input_path(raw)?;
        if self.scope == ToolHostScope::Workspace {
            reject_symlink_components(&self.root, &candidate)?;
        }
        let canonical = fs::canonicalize(&candidate).map_err(|error| {
            HostError::Workspace(format!("path {raw:?} is not accessible: {error}"))
        })?;
        if !allow_root && canonical == self.root {
            return Err(HostError::Workspace(
                "workspace root is not a file target".to_string(),
            ));
        }
        if self.scope == ToolHostScope::Workspace && !canonical.starts_with(&self.root) {
            return Err(HostError::Workspace(
                "path escapes the workspace root".to_string(),
            ));
        }
        Ok(canonical)
    }

    fn resolve_input_path(&self, raw: &str) -> Result<PathBuf, HostError> {
        let input = validate_path(raw, self.scope)?;
        if self.scope == ToolHostScope::Machine && input.is_absolute() {
            Ok(input.to_path_buf())
        } else {
            Ok(self.root.join(input))
        }
    }

    fn resolve_write_target(&self, raw: &str) -> Result<PathBuf, HostError> {
        let candidate = self.resolve_input_path(raw)?;
        if candidate == self.root {
            return Err(HostError::Workspace(
                "workspace root is not a file target".to_string(),
            ));
        }
        if self.scope == ToolHostScope::Workspace && !candidate.starts_with(&self.root) {
            return Err(HostError::Workspace(
                "path escapes the workspace root".to_string(),
            ));
        }
        if self.scope == ToolHostScope::Workspace {
            reject_link_prefix_components(&self.root, &candidate)?;
        }
        let path = self.resolve_write_prefix(&candidate, raw)?;
        if self.scope == ToolHostScope::Workspace && !path.starts_with(&self.root) {
            return Err(HostError::Workspace(
                "path escapes the workspace root".to_string(),
            ));
        }
        Ok(path)
    }

    /// Canonicalize the existing prefix of a possibly-new target. This keeps
    /// structured machine writes from following a symlink into `.tina` while
    /// still allowing `mkdir --recursive` to create several missing levels.
    fn resolve_write_prefix(&self, candidate: &Path, raw: &str) -> Result<PathBuf, HostError> {
        let mut missing = Vec::new();
        let mut probe = candidate.to_path_buf();
        loop {
            match fs::symlink_metadata(&probe) {
                Ok(metadata) => {
                    if missing.is_empty() && metadata_is_link_like(&metadata) {
                        return Err(HostError::Workspace(
                            "link or reparse-point write targets are not accepted".to_string(),
                        ));
                    }
                    let canonical = fs::canonicalize(&probe).map_err(|error| {
                        HostError::Workspace(format!("path {raw:?} is not accessible: {error}"))
                    })?;
                    let mut resolved = canonical;
                    for component in missing.iter().rev() {
                        match Path::new(component).components().next() {
                            Some(Component::CurDir) => {}
                            Some(Component::ParentDir) => {
                                if !resolved.pop() {
                                    resolved.push(component);
                                }
                            }
                            Some(Component::Normal(_)) => resolved.push(component),
                            _ => {
                                return Err(HostError::Workspace(format!(
                                    "path {raw:?} contains an unsupported component"
                                )))
                            }
                        }
                    }
                    return Ok(resolved);
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    let name = probe.file_name().ok_or_else(|| {
                        HostError::Workspace("write target has no file name".to_string())
                    })?;
                    missing.push(name.to_os_string());
                    if !probe.pop() {
                        return Err(HostError::Workspace(format!(
                            "parent directory for {raw:?} is not accessible"
                        )));
                    }
                }
                Err(error) => {
                    return Err(HostError::Workspace(format!(
                        "path {raw:?} is not accessible: {error}"
                    )))
                }
            }
        }
    }

    fn is_host_control_path(&self, path: &Path) -> bool {
        path == self.artifact_dir
            || path.starts_with(&self.artifact_dir)
            || path == self.root.join(".tina")
            || path.starts_with(self.root.join(".tina"))
    }

    fn is_workspace_vcs_control_path(&self, path: &Path) -> bool {
        self.scope == ToolHostScope::Workspace
            && path
                .strip_prefix(&self.root)
                .ok()
                .is_some_and(|relative| {
                    relative.components().any(|component| {
                        matches!(component, Component::Normal(name) if name.eq_ignore_ascii_case(".git"))
                    })
                })
    }

    fn is_workspace_root_or_control_path(&self, path: &Path) -> bool {
        if path == self.root
            || self.is_host_control_path(path)
            || self.is_workspace_vcs_control_path(path)
        {
            return true;
        }
        fs::canonicalize(path).is_ok_and(|canonical| {
            canonical == self.root
                || self.is_host_control_path(&canonical)
                || self.is_workspace_vcs_control_path(&canonical)
        })
    }

    fn relative_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    fn result_from_text(&self, kind: &str, text: String) -> Result<ToolResultBody, HostError> {
        let bytes = text.as_bytes();
        let digest = sha256_hex(bytes);
        let truncated = bytes.len() > self.limits.inline_output_bytes;
        let inline_len = utf8_prefix_len(&text, min(bytes.len(), self.limits.inline_output_bytes));
        let inline = text[..inline_len].to_owned();
        let artifact = if truncated {
            let metadata = fs::symlink_metadata(&self.artifact_dir)?;
            let canonical_artifact_dir = fs::canonicalize(&self.artifact_dir)?;
            if metadata_is_link_like(&metadata)
                || !metadata.is_dir()
                || canonical_artifact_dir != self.artifact_dir
                || !canonical_artifact_dir.starts_with(&self.root)
            {
                return Err(HostError::Workspace(
                    "tool artifact directory identity drifted".to_string(),
                ));
            }
            let path = self.artifact_dir.join(format!("{digest}.txt"));
            persist_content_addressed_artifact(&path, bytes)?;
            Some(self.relative_path(&path))
        } else {
            None
        };
        Ok(ToolResultBody {
            kind: kind.to_string(),
            text: inline,
            bytes: bytes.len(),
            sha256: digest,
            truncated,
            artifact,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadArgs {
    path: String,
    start_line: Option<usize>,
    end_line: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ListArgs {
    path: Option<String>,
    recursive: Option<bool>,
    max_entries: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    query: String,
    path: Option<String>,
    max_matches: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WebSearchArgs {
    query: String,
    #[serde(default)]
    max_results: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GitArgs {
    path: Option<String>,
    cached: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyPatchArgs {
    path: String,
    expected_sha256: String,
    edits: Vec<ExactReplacement>,
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
struct DeleteArgs {
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
struct ShellArgs {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
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

#[derive(Debug, Serialize)]
struct ShellReceipt {
    shell: &'static str,
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

#[derive(Debug, Serialize)]
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
    policy: &'static str,
    os_sandbox: bool,
    inherited_environment: bool,
    network_access: &'static str,
}

#[derive(Debug)]
struct ProcessOutput {
    bytes: Vec<u8>,
    total_bytes: usize,
    truncated: bool,
}

fn validate_workspace_exec_argv(
    argv: &[String],
    max_args: usize,
    max_argument_bytes: usize,
) -> Result<(), HostError> {
    if argv.is_empty() {
        return Err(HostError::InvalidRequest(
            "workspace_exec argv must not be empty".to_string(),
        ));
    }
    if argv.len() > max_args {
        return Err(HostError::InvalidRequest(format!(
            "workspace_exec accepts at most {max_args} argv items"
        )));
    }
    let executable = argv[0].to_ascii_lowercase();
    if executable != "cargo" && executable != "cargo.exe" {
        return Err(HostError::InvalidRequest(
            "workspace_exec currently allows only cargo or cargo.exe".to_string(),
        ));
    }
    let subcommand = argv.get(1).map(String::as_str).ok_or_else(|| {
        HostError::InvalidRequest(
            "workspace_exec requires a cargo check/build/test/clippy/fmt/metadata subcommand"
                .to_string(),
        )
    })?;
    if !matches!(
        subcommand,
        "check" | "build" | "test" | "clippy" | "fmt" | "metadata"
    ) {
        return Err(HostError::InvalidRequest(format!(
            "cargo subcommand {subcommand:?} is not allowed by workspace_exec"
        )));
    }
    let mut total_bytes = 0_usize;
    for (index, arg) in argv.iter().enumerate() {
        if arg.is_empty() || arg.as_bytes().contains(&0) || arg.chars().any(char::is_control) {
            return Err(HostError::InvalidRequest(format!(
                "workspace_exec argv item {index} is empty or contains control data"
            )));
        }
        if arg.len() > MAX_EXEC_ARG_BYTES_PER_ITEM {
            return Err(HostError::InvalidRequest(format!(
                "workspace_exec argv item {index} exceeds {MAX_EXEC_ARG_BYTES_PER_ITEM} bytes"
            )));
        }
        total_bytes = total_bytes
            .checked_add(arg.len().saturating_add(1))
            .ok_or_else(|| {
                HostError::InvalidRequest("workspace_exec argv size overflowed".into())
            })?;
        if total_bytes > max_argument_bytes {
            return Err(HostError::InvalidRequest(format!(
                "workspace_exec argv exceeds {max_argument_bytes} bytes"
            )));
        }
        if index > 0
            && (arg == "--config"
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
            return Err(HostError::InvalidRequest(format!(
                "cargo argument {arg:?} can change the workspace or execution environment"
            )));
        }
        if index > 0
            && (Path::new(arg).is_absolute()
                || arg.starts_with('/')
                || arg.starts_with('\\')
                || (arg.as_bytes().get(1) == Some(&b':'))
                || Path::new(arg)
                    .components()
                    .any(|component| matches!(component, Component::ParentDir)))
        {
            return Err(HostError::Workspace(format!(
                "workspace_exec argument {arg:?} contains an absolute or parent path"
            )));
        }
    }
    Ok(())
}

fn fetch_searxng_json(
    provider: &WebSearchProvider,
    query: &str,
    deadline_unix_ms: u64,
) -> Result<Vec<u8>, HostError> {
    validate_deadline(deadline_unix_ms)?;
    let remaining_ms = deadline_unix_ms.saturating_sub(unix_millis()?);
    let timeout_ms = remaining_ms.clamp(1, DEFAULT_MAX_WEB_SEARCH_TIMEOUT_MS);
    let timeout_seconds = format!("{:.3}", timeout_ms as f64 / 1_000.0);
    let mut child = Command::new(curl_program())
        .args([
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--location",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--connect-timeout",
            "5",
            "--max-time",
            &timeout_seconds,
            "--max-filesize",
            &DEFAULT_MAX_WEB_SEARCH_RESPONSE_BYTES.to_string(),
            "--header",
            "Accept: application/json",
            "--get",
            "--data-urlencode",
            &format!("q={query}"),
            "--data",
            "format=json",
            &provider.endpoint,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            HostError::Tool(format!("cannot start curl web-search client: {error}"))
        })?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| HostError::Tool("curl omitted its stdout pipe".to_string()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| HostError::Tool("curl omitted its stderr pipe".to_string()))?;
    let stdout_reader =
        thread::spawn(move || read_process_output(stdout, DEFAULT_MAX_WEB_SEARCH_RESPONSE_BYTES));
    let stderr_reader = thread::spawn(move || read_process_output(stderr, MAX_ERROR_MESSAGE_BYTES));
    let started = Instant::now();
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait()? {
            break (status, false);
        }
        if started.elapsed() >= Duration::from_millis(timeout_ms)
            || unix_millis()? >= deadline_unix_ms
        {
            let _ = child.kill();
            break (child.wait()?, true);
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| HostError::Tool("curl stdout reader panicked".to_string()))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| HostError::Tool("curl stderr reader panicked".to_string()))??;
    if timed_out {
        return Err(HostError::Deadline(
            "bounded web-search request timed out".to_string(),
        ));
    }
    if stdout.truncated {
        return Err(HostError::Tool(format!(
            "web-search response exceeded {DEFAULT_MAX_WEB_SEARCH_RESPONSE_BYTES} bytes"
        )));
    }
    if !status.success() {
        let detail = bounded_external_text(
            &String::from_utf8_lossy(&stderr.bytes),
            MAX_ERROR_MESSAGE_BYTES,
        );
        return Err(HostError::Tool(format!(
            "web-search provider returned {status}: {}",
            detail.trim()
        )));
    }
    Ok(stdout.bytes)
}

fn normalize_searxng_response(
    provider: &WebSearchProvider,
    query: &str,
    max_results: usize,
    response: &[u8],
) -> Result<String, HostError> {
    let document: Value = serde_json::from_slice(response)
        .map_err(|error| HostError::Tool(format!("web-search response is not JSON: {error}")))?;
    let raw_results = document
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| HostError::Tool("web-search JSON omitted its results array".to_string()))?;
    let mut urls = BTreeSet::new();
    let mut results = Vec::new();
    for raw in raw_results {
        let Some(url) = raw.get("url").and_then(Value::as_str) else {
            continue;
        };
        if !valid_web_result_url(url) || !urls.insert(url.to_string()) {
            continue;
        }
        let title = bounded_external_text(
            raw.get("title").and_then(Value::as_str).unwrap_or(""),
            MAX_WEB_SEARCH_TITLE_BYTES,
        );
        let snippet = bounded_external_text(
            raw.get("content").and_then(Value::as_str).unwrap_or(""),
            MAX_WEB_SEARCH_SNIPPET_BYTES,
        );
        results.push(serde_json::json!({
            "url": url,
            "title": title,
            "snippet": snippet,
        }));
        if results.len() == max_results {
            break;
        }
    }
    Ok(serde_json::json!({
        "schema": "tina.web-search.receipt.v1",
        "provider": "searxng-json",
        "provider_endpoint_sha256": provider.endpoint_sha256,
        "query": query,
        "result_count": results.len(),
        "untrusted_external_data": true,
        "results": results,
    })
    .to_string())
}

fn validate_web_search_endpoint(raw: &str) -> Result<String, HostError> {
    let endpoint = raw.trim();
    if endpoint.is_empty()
        || endpoint.len() > MAX_WEB_SEARCH_ENDPOINT_BYTES
        || endpoint.chars().any(char::is_control)
        || endpoint.contains(char::is_whitespace)
        || endpoint.contains('#')
    {
        return Err(HostError::InvalidConfig(
            "web-search URL must be a bounded HTTP(S) URL without whitespace or fragments"
                .to_string(),
        ));
    }
    let authority = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .and_then(|rest| rest.split(['/', '?']).next())
        .filter(|authority| !authority.is_empty() && !authority.contains('@'))
        .ok_or_else(|| {
            HostError::InvalidConfig(
                "web-search URL must use http:// or https:// with a host and no credentials"
                    .to_string(),
            )
        })?;
    let host = if let Some(ipv6) = authority.strip_prefix('[') {
        ipv6.split_once(']').map(|(host, _)| host).unwrap_or("")
    } else {
        authority.split(':').next().unwrap_or("")
    };
    if host.is_empty() {
        return Err(HostError::InvalidConfig(
            "web-search URL host is empty".to_string(),
        ));
    }
    Ok(endpoint.to_string())
}

fn validate_web_search_client() -> Result<(), HostError> {
    let status = Command::new(curl_program())
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| {
            HostError::InvalidConfig(format!(
                "web_search requires a directly executable curl client: {error}"
            ))
        })?;
    if !status.success() {
        return Err(HostError::InvalidConfig(format!(
            "curl capability probe failed with {status}"
        )));
    }
    Ok(())
}

fn curl_program() -> &'static str {
    if cfg!(windows) {
        "curl.exe"
    } else {
        "curl"
    }
}

fn valid_web_result_url(url: &str) -> bool {
    if url.is_empty()
        || url.len() > MAX_WEB_SEARCH_URL_BYTES
        || url.chars().any(char::is_control)
        || url.contains(char::is_whitespace)
    {
        return false;
    }
    let Some(authority) = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .and_then(|rest| rest.split(['/', '?', '#']).next())
        .filter(|authority| !authority.is_empty() && !authority.contains('@'))
    else {
        return false;
    };
    if let Some(ipv6) = authority.strip_prefix('[') {
        ipv6.split_once(']')
            .is_some_and(|(host, _)| !host.is_empty())
    } else {
        authority
            .split(':')
            .next()
            .is_some_and(|host| !host.is_empty())
    }
}

fn bounded_external_text(value: &str, max_bytes: usize) -> String {
    let mut normalized = String::with_capacity(value.len().min(max_bytes));
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_whitespace() || character.is_control() {
            pending_space = !normalized.is_empty();
            continue;
        }
        if pending_space {
            if normalized.len() + 1 > max_bytes {
                break;
            }
            normalized.push(' ');
            pending_space = false;
        }
        if normalized.len() + character.len_utf8() > max_bytes {
            break;
        }
        normalized.push(character);
    }
    normalized
}

fn read_process_output(mut reader: impl Read, limit: usize) -> io::Result<ProcessOutput> {
    let mut bytes = Vec::with_capacity(limit.min(8 * 1024));
    let mut total_bytes = 0_usize;
    let mut truncated = false;
    let mut buffer = [0_u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total_bytes = total_bytes.saturating_add(read);
        if bytes.len() < limit {
            let keep = (limit - bytes.len()).min(read);
            bytes.extend_from_slice(&buffer[..keep]);
            if keep < read {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    Ok(ProcessOutput {
        bytes,
        total_bytes,
        truncated,
    })
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExactReplacement {
    old: String,
    new: String,
}

#[derive(Debug)]
struct LocatedReplacement<'a> {
    start: usize,
    end: usize,
    new: &'a str,
}

fn parse_args<T: for<'de> Deserialize<'de>>(args: &Value, operation: &str) -> Result<T, HostError> {
    serde_json::from_value(args.clone()).map_err(|error| {
        HostError::InvalidRequest(format!("invalid {operation} arguments: {error}"))
    })
}

fn valid_request_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 128 && id.bytes().all(|byte| (0x20..=0x7e).contains(&byte))
}

fn error_code(error: &HostError) -> &'static str {
    match error {
        HostError::Workspace(_) => "workspace_denied",
        HostError::StaleRead(_) => "stale_read",
        HostError::PatchConflict(_) => "patch_conflict",
        HostError::InvalidRequest(_) => "invalid_arguments",
        HostError::Tool(_) => "tool_error",
        HostError::Protocol(_) => "protocol_error",
        HostError::Deadline(_) => "deadline_exceeded",
        HostError::InvalidConfig(_) => "invalid_config",
        HostError::Io(_) | HostError::Json(_) => "host_error",
    }
}

fn validate_deadline(deadline_unix_ms: u64) -> Result<(), HostError> {
    let now = unix_millis().map_err(HostError::Io)?;
    if deadline_unix_ms <= now {
        return Err(HostError::Deadline(
            "the request deadline has already passed".to_string(),
        ));
    }
    if deadline_unix_ms.saturating_sub(now) > MAX_REQUEST_DEADLINE_MS {
        return Err(HostError::InvalidRequest(format!(
            "request deadline exceeds the {MAX_REQUEST_DEADLINE_MS} ms host limit"
        )));
    }
    Ok(())
}

fn unix_millis() -> io::Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_millis();
    u64::try_from(millis).map_err(|_| io::Error::other("system time exceeds u64 milliseconds"))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_path(raw: &str, scope: ToolHostScope) -> Result<&Path, HostError> {
    if raw.is_empty() || raw.as_bytes().contains(&0) {
        return Err(HostError::Workspace(
            "path is empty or contains NUL".to_string(),
        ));
    }
    let input = Path::new(raw);
    if scope == ToolHostScope::Workspace && input.is_absolute() {
        return Err(HostError::Workspace(
            "absolute paths are not accepted; use a workspace-relative path".to_string(),
        ));
    }
    if scope == ToolHostScope::Workspace
        && input
            .components()
            .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(HostError::Workspace(
            "parent path components are not accepted".to_string(),
        ));
    }
    #[cfg(windows)]
    if scope == ToolHostScope::Machine && (raw.starts_with(r"\\.\") || raw.starts_with(r"\\?\")) {
        return Err(HostError::Workspace(
            "Windows device namespace paths are not accepted".to_string(),
        ));
    }
    Ok(input)
}

#[cfg(windows)]
fn windows_path_has_alternate_data_stream(raw: &str) -> bool {
    raw.char_indices()
        .filter(|(_, character)| *character == ':')
        .any(|(index, _)| index != 1 || !raw.as_bytes()[0].is_ascii_alphabetic())
}

fn ensure_directory_without_symlink(path: &Path) -> Result<(), HostError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata_is_link_like(&metadata) || !metadata.is_dir() {
                return Err(HostError::Workspace(format!(
                    "host directory is not a real directory: {}",
                    path.display()
                )));
            }
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(path)?;
            Ok(())
        }
        Err(error) => Err(HostError::Io(error)),
    }
}

fn metadata_is_link_like(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    {
        false
    }
}

fn reject_link_prefix_components(root: &Path, candidate: &Path) -> Result<(), HostError> {
    let relative = candidate.strip_prefix(root).map_err(|_| {
        HostError::Workspace("path does not belong to the workspace root".to_string())
    })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => current.push(name),
            _ => {
                return Err(HostError::Workspace(
                    "path contains an unsupported component".to_string(),
                ))
            }
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata_is_link_like(&metadata) => {
                return Err(HostError::Workspace(format!(
                    "link or reparse-point components are not accepted: {}",
                    current.display()
                )))
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(HostError::Workspace(format!(
                    "path component {} is not accessible: {error}",
                    current.display()
                )))
            }
        }
    }
    Ok(())
}

fn reject_symlink_components(root: &Path, candidate: &Path) -> Result<(), HostError> {
    let relative = candidate.strip_prefix(root).map_err(|_| {
        HostError::Workspace("path does not belong to the workspace root".to_string())
    })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        match component {
            Component::CurDir => continue,
            Component::Normal(name) => current.push(name),
            _ => {
                return Err(HostError::Workspace(
                    "path contains an unsupported component".to_string(),
                ))
            }
        }
        let metadata = fs::symlink_metadata(&current).map_err(|error| {
            HostError::Workspace(format!(
                "path component {} is not accessible: {error}",
                current.display()
            ))
        })?;
        if metadata_is_link_like(&metadata) {
            return Err(HostError::Workspace(format!(
                "link or reparse-point components are not accepted: {}",
                current.display()
            )));
        }
    }
    Ok(())
}

fn apply_exact_replacements(
    original: &str,
    edits: &[ExactReplacement],
) -> Result<String, HostError> {
    let mut located = Vec::with_capacity(edits.len());
    for (index, edit) in edits.iter().enumerate() {
        if edit.old.is_empty() {
            return Err(HostError::InvalidRequest(format!(
                "edit {index} has an empty old string"
            )));
        }
        if edit.old == edit.new {
            return Err(HostError::InvalidRequest(format!(
                "edit {index} does not change its matched text"
            )));
        }
        let mut matches = original.match_indices(&edit.old);
        let (start, _) = matches.next().ok_or_else(|| {
            HostError::PatchConflict(format!("edit {index} old text was not found"))
        })?;
        if matches.next().is_some() {
            return Err(HostError::PatchConflict(format!(
                "edit {index} old text is ambiguous; include more exact context"
            )));
        }
        located.push(LocatedReplacement {
            start,
            end: start + edit.old.len(),
            new: &edit.new,
        });
    }
    located.sort_by_key(|edit| edit.start);
    for pair in located.windows(2) {
        if pair[0].end > pair[1].start {
            return Err(HostError::PatchConflict(
                "exact edits overlap in the original file".to_string(),
            ));
        }
    }
    let covered_bytes = located.iter().try_fold(0_usize, |total, edit| {
        total.checked_add(edit.end - edit.start)
    });
    if covered_bytes == Some(original.len()) {
        return Err(HostError::PatchConflict(
            "apply_patch cannot replace the complete file; full overwrite requires the destructive-operation gate"
                .to_string(),
        ));
    }

    let removed_bytes = covered_bytes
        .ok_or_else(|| HostError::InvalidRequest("edit byte count overflowed".to_string()))?;
    let added_bytes = located
        .iter()
        .try_fold(0_usize, |total, edit| total.checked_add(edit.new.len()));
    let capacity = original
        .len()
        .checked_sub(removed_bytes)
        .and_then(|value| value.checked_add(added_bytes?))
        .ok_or_else(|| HostError::InvalidRequest("patched byte count overflowed".to_string()))?;
    let mut patched = String::with_capacity(capacity);
    let mut cursor = 0;
    for edit in located {
        patched.push_str(&original[cursor..edit.start]);
        patched.push_str(edit.new);
        cursor = edit.end;
    }
    patched.push_str(&original[cursor..]);
    if patched == original {
        return Err(HostError::PatchConflict(
            "apply_patch produced no file change".to_string(),
        ));
    }
    Ok(patched)
}

fn persist_content_addressed_artifact(path: &Path, bytes: &[u8]) -> Result<(), HostError> {
    match OpenOptions::new().create_new(true).write(true).open(path) {
        Ok(mut file) => {
            if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
                drop(file);
                let _ = fs::remove_file(path);
                return Err(HostError::Io(error));
            }
            Ok(())
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::symlink_metadata(path)?;
            if metadata_is_link_like(&metadata) || !metadata.is_file() {
                return Err(HostError::Workspace(
                    "tool artifact path is not an existing regular file".to_string(),
                ));
            }
            let existing = fs::read(path)?;
            if existing != bytes {
                return Err(HostError::Protocol(
                    "content-addressed tool artifact does not match its digest".to_string(),
                ));
            }
            Ok(())
        }
        Err(error) => Err(HostError::Io(error)),
    }
}

struct TemporaryReplacement {
    path: PathBuf,
}

impl TemporaryReplacement {
    fn create(
        directory: &Path,
        permissions: Option<fs::Permissions>,
        bytes: &[u8],
    ) -> Result<Self, HostError> {
        for _ in 0..32 {
            let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(".tina-apply-{}-{sequence}.tmp", std::process::id()));
            match OpenOptions::new().create_new(true).write(true).open(&path) {
                Ok(mut file) => {
                    let mut write_result = file.write_all(bytes);
                    if write_result.is_ok() {
                        if let Some(permissions) = permissions.as_ref() {
                            write_result = file.set_permissions(permissions.clone());
                        }
                    }
                    if write_result.is_ok() {
                        write_result = file.sync_all();
                    }
                    drop(file);
                    if let Err(error) = write_result {
                        let _ = fs::remove_file(&path);
                        return Err(HostError::Io(error));
                    }
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(HostError::Io(error)),
            }
        }
        Err(HostError::Tool(
            "could not reserve a bounded temporary patch file".to_string(),
        ))
    }

    fn commit(mut self, destination: &Path) -> Result<(), HostError> {
        replace_existing(&self.path, destination)?;
        self.path.clear();
        Ok(())
    }

    fn commit_new(mut self, destination: &Path) -> Result<(), HostError> {
        fs::rename(&self.path, destination)?;
        self.path.clear();
        Ok(())
    }
}

impl Drop for TemporaryReplacement {
    fn drop(&mut self) {
        if !self.path.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(windows)]
fn replace_existing(temporary: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    // SAFETY: this declaration matches the Kernel32 `ReplaceFileW` ABI. The
    // call site below supplies NUL-terminated UTF-16 buffers and null optional
    // pointers exactly as required by the Windows contract.
    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn ReplaceFileW(
            replaced_file_name: *const u16,
            replacement_file_name: *const u16,
            backup_file_name: *const u16,
            replace_flags: u32,
            exclude: *mut core::ffi::c_void,
            reserved: *mut core::ffi::c_void,
        ) -> i32;
    }

    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let temporary = temporary
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // SAFETY: both paths are owned, NUL-terminated UTF-16 buffers that remain
    // live for the call. Destination is an existing regular file and the
    // replacement was closed after fsync in the same directory.
    let replaced = unsafe {
        ReplaceFileW(
            destination.as_ptr(),
            temporary.as_ptr(),
            core::ptr::null(),
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        )
    };
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_existing(temporary: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(temporary, destination)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn read_dir_sorted(path: &Path) -> Result<Vec<fs::DirEntry>, HostError> {
    let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    Ok(entries)
}

fn should_skip_directory(path: &Path, workspace: &Path) -> bool {
    path != workspace
        && matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some(".git") | Some("target") | Some(".tina")
        )
}

fn collect_files(
    root: &Path,
    workspace: &Path,
    files: &mut Vec<PathBuf>,
    cap: usize,
) -> Result<(), HostError> {
    if files.len() >= cap {
        return Ok(());
    }
    for entry in read_dir_sorted(root)? {
        if files.len() >= cap {
            break;
        }
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        if metadata_is_link_like(&metadata) {
            continue;
        }
        if metadata.is_dir() {
            if should_skip_directory(&path, workspace) {
                continue;
            }
            collect_files(&path, workspace, files, cap)?;
        } else if metadata.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

/// Run the host as a JSONL server over stdin/stdout.
pub fn run_jsonl<R: BufRead, W: Write>(
    host: &ToolHost,
    mut reader: R,
    mut writer: W,
) -> io::Result<()> {
    let mut frame = Vec::with_capacity(host.limits.max_request_bytes.min(64 * 1024));
    loop {
        frame.clear();
        let frame_limit = host.limits.max_request_bytes.saturating_add(2) as u64;
        let bytes_read = (&mut reader)
            .take(frame_limit)
            .read_until(b'\n', &mut frame)?;
        if bytes_read == 0 {
            break;
        }

        let newline_terminated = frame.last() == Some(&b'\n');
        if newline_terminated {
            frame.pop();
            if frame.last() == Some(&b'\r') {
                frame.pop();
            }
        }
        if frame.len() > host.limits.max_request_bytes {
            if !newline_terminated {
                discard_until_newline(&mut reader)?;
            }
            write_response(
                &mut writer,
                &ToolResponse::error(
                    String::new(),
                    "request_too_large",
                    format!(
                        "request exceeds the {}-byte frame limit",
                        host.limits.max_request_bytes
                    ),
                ),
            )?;
            continue;
        }

        let line = match std::str::from_utf8(&frame) {
            Ok(line) => line,
            Err(error) => {
                write_response(
                    &mut writer,
                    &ToolResponse::error(String::new(), "invalid_utf8", error.to_string()),
                )?;
                if !newline_terminated {
                    break;
                }
                continue;
            }
        };
        if !line.trim().is_empty() {
            if is_cancel_frame(line) {
                // Cancellation is a control frame, not a second response for
                // the request it targets. The parent either receives the
                // original terminal receipt during its grace period or kills
                // this process tree.
                let _ = host.handle_line_result(line);
            } else {
                writer.write_all(host.handle_line(line).as_bytes())?;
                writer.write_all(b"\n")?;
                writer.flush()?;
            }
        }
        if !newline_terminated {
            break;
        }
    }
    Ok(())
}

fn is_cancel_frame(line: &str) -> bool {
    serde_json::from_str::<Value>(line)
        .ok()
        .and_then(|value| {
            value
                .get("kind")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .as_deref()
        == Some("cancel")
}

fn write_response(writer: &mut impl Write, response: &ToolResponse) -> io::Result<()> {
    serde_json::to_writer(&mut *writer, response).map_err(io::Error::other)?;
    writer.write_all(b"\n")?;
    writer.flush()
}

fn discard_until_newline(reader: &mut impl BufRead) -> io::Result<()> {
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(());
        }
        if let Some(index) = available.iter().position(|byte| *byte == b'\n') {
            reader.consume(index + 1);
            return Ok(());
        }
        let consumed = available.len();
        reader.consume(consumed);
    }
}

fn utf8_prefix_len(value: &str, maximum_bytes: usize) -> usize {
    let mut length = maximum_bytes.min(value.len());
    while length > 0 && !value.is_char_boundary(length) {
        length -= 1;
    }
    length
}

fn truncate_utf8_owned(mut value: String, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value;
    }
    let length = utf8_prefix_len(&value, maximum_bytes);
    value.truncate(length);
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempRoot(PathBuf);

    impl TempRoot {
        fn new() -> Self {
            let suffix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("cq-tool-host-{suffix}"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn host() -> (TempRoot, ToolHost) {
        let root = TempRoot::new();
        fs::write(root.0.join("one.txt"), "alpha\nbeta\n").unwrap();
        fs::create_dir_all(root.0.join("src")).unwrap();
        fs::write(root.0.join("src/two.rs"), "fn beta() {}\n").unwrap();
        let host = ToolHost::new(&root.0).unwrap();
        (root, host)
    }

    fn request(id: &str, op: &str, args: Value) -> String {
        serde_json::json!({
            "v": PROTOCOL_VERSION,
            "kind": "request",
            "id": id,
            "op": op,
            "deadline_unix_ms": unix_millis().unwrap() + 60_000,
            "args": args,
        })
        .to_string()
    }

    fn patch_args(path: &str, before: &str, old: &str, new: &str) -> Value {
        serde_json::json!({
            "path": path,
            "expected_sha256": before,
            "edits": [{"old": old, "new": new}],
        })
    }

    #[test]
    fn read_is_workspace_relative_and_bounded() {
        let (_root, host) = host();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "r1",
            "read",
            serde_json::json!({"path":"one.txt"}),
        )))
        .unwrap();
        assert!(response.ok);
        assert_eq!(response.result.unwrap().text, "alpha\nbeta\n");

        let denied: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "r2",
            "read",
            serde_json::json!({"path":"../one.txt"}),
        )))
        .unwrap();
        assert!(!denied.ok);
        assert_eq!(denied.error.unwrap().code, "workspace_denied");
    }

    #[test]
    fn capabilities_identify_the_bounded_write_surface() {
        let (_root, host) = host();
        let response: ToolResponse =
            serde_json::from_str(&host.handle_line(&request("c1", "capabilities", Value::Null)))
                .unwrap();
        let result: Value = serde_json::from_str(&response.result.unwrap().text).unwrap();
        assert_eq!(result["writes"], true);
        assert_eq!(result["network"], false);
        assert_eq!(
            result["tools"],
            serde_json::json!([
                "capabilities",
                "read",
                "list",
                "search",
                "git_status",
                "git_diff",
                "apply_patch",
                "write_file",
                "mkdir",
                "delete",
                "rename",
                "workspace_exec"
            ])
        );
    }

    #[test]
    fn workspace_exec_returns_an_explicit_terminal_receipt() {
        let root = TempRoot::new();
        fs::create_dir_all(root.0.join("src")).unwrap();
        fs::write(
            root.0.join("Cargo.toml"),
            "[package]\nname = \"receipt_fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .unwrap();
        fs::write(root.0.join("src/lib.rs"), "pub fn value() -> u8 { 1 }\n").unwrap();
        let host = ToolHost::new(&root.0).unwrap();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "exec-receipt",
            "workspace_exec",
            serde_json::json!({
                "argv": ["cargo", "metadata", "--no-deps", "--format-version=1"],
                "timeout_ms": 30_000,
            }),
        )))
        .unwrap();
        assert!(response.ok, "workspace_exec failed: {:?}", response.error);
        let result = response.result.unwrap();
        assert_eq!(result.kind, "exec");
        let receipt: Value = serde_json::from_str(&result.text).unwrap();
        assert_eq!(receipt["success"], true, "receipt: {receipt}");
        assert_eq!(receipt["exit_code"], 0);
        assert_eq!(receipt["timed_out"], false);
        assert_eq!(receipt["policy"], "bounded-argv-cwd-v1");
        assert_eq!(receipt["os_sandbox"], false);
        assert_eq!(receipt["inherited_environment"], true);
        assert_eq!(receipt["network_access"], "unrestricted");
        assert_eq!(
            receipt["argv"][0].as_str().unwrap().to_ascii_lowercase(),
            "cargo"
        );
        assert_eq!(
            receipt["cwd"],
            root.0.canonicalize().unwrap().display().to_string()
        );
    }

    #[test]
    fn workspace_exec_rejects_shell_escape_and_boundary_flags_before_spawn() {
        let (_root, host) = host();
        for (id, argv) in [
            ("exec-shell", vec!["powershell", "-Command", "cargo check"]),
            ("exec-run", vec!["cargo", "run"]),
            ("exec-parent", vec!["cargo", "check", "../outside"]),
            (
                "exec-manifest",
                vec!["cargo", "check", "--manifest-path", "other/Cargo.toml"],
            ),
        ] {
            let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
                id,
                "workspace_exec",
                serde_json::json!({"argv": argv}),
            )))
            .unwrap();
            assert!(!response.ok, "unexpected success for {id}");
            assert!(matches!(
                response.error.unwrap().code.as_str(),
                "invalid_arguments" | "workspace_denied"
            ));
        }
    }

    #[test]
    fn web_search_capability_requires_an_explicit_valid_provider() {
        let root = TempRoot::new();
        let host = ToolHost::with_scope_shell_and_web_search_url(
            &root.0,
            ToolHostScope::Workspace,
            ToolHostShell::PowerShell,
            Some("https://search.example.test/search"),
        )
        .unwrap();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "web-capabilities",
            "capabilities",
            Value::Null,
        )))
        .unwrap();
        let capabilities: Value = serde_json::from_str(&response.result.unwrap().text).unwrap();
        assert_eq!(capabilities["network"], false);
        assert_eq!(capabilities["web_search"], true);
        assert!(capabilities["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool == "web_search"));

        for invalid in [
            "ftp://search.example.test/",
            "https://user:secret@search.example.test/",
            "https://search.example.test/#fragment",
            "https://:8080/search",
            "https:///search",
        ] {
            assert!(ToolHost::with_scope_shell_and_web_search_url(
                &root.0,
                ToolHostScope::Workspace,
                ToolHostShell::PowerShell,
                Some(invalid),
            )
            .is_err());
        }
    }

    #[test]
    fn searxng_receipt_is_bounded_deduplicated_and_marked_untrusted() {
        let endpoint = validate_web_search_endpoint("https://search.example.test/search").unwrap();
        let provider = WebSearchProvider {
            endpoint_sha256: sha256_hex(endpoint.as_bytes()),
            endpoint,
        };
        let response = serde_json::to_vec(&serde_json::json!({
            "results": [
                {
                    "url": "https://docs.example.test/one",
                    "title": " First\nresult ",
                    "content": "external\u{1b}[31m text"
                },
                {
                    "url": "https://docs.example.test/one",
                    "title": "duplicate",
                    "content": "duplicate"
                },
                {
                    "url": "javascript:alert(1)",
                    "title": "bad scheme",
                    "content": "ignored"
                },
                {
                    "url": "http://docs.example.test/two",
                    "title": "Second",
                    "content": "valid"
                }
            ]
        }))
        .unwrap();
        let receipt: Value = serde_json::from_str(
            &normalize_searxng_response(&provider, "rust agent", 2, &response).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["schema"], "tina.web-search.receipt.v1");
        assert_eq!(receipt["untrusted_external_data"], true);
        assert_eq!(receipt["result_count"], 2);
        assert_eq!(receipt["results"][0]["title"], "First result");
        assert_eq!(receipt["results"][0]["snippet"], "external [31m text");
        assert_eq!(
            receipt["provider_endpoint_sha256"],
            sha256_hex(b"https://search.example.test/search")
        );
    }

    #[test]
    fn web_search_executes_a_real_bounded_loopback_provider_request() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let server = thread::spawn(move || {
            let started = Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        assert!(started.elapsed() < Duration::from_secs(5));
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("loopback provider accept failed: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 2 * 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).unwrap();
                assert!(read > 0);
                request.extend_from_slice(&buffer[..read]);
                assert!(request.len() <= 16 * 1024);
            }
            let body = serde_json::json!({
                "results": [{
                    "url": "https://docs.example.test/rust",
                    "title": "Rust docs",
                    "content": "bounded loopback result"
                }]
            })
            .to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
            String::from_utf8(request).unwrap()
        });
        let root = TempRoot::new();
        let endpoint = format!("http://{address}/search");
        let host = ToolHost::with_scope_shell_and_web_search_url(
            &root.0,
            ToolHostScope::Workspace,
            ToolHostShell::PowerShell,
            Some(&endpoint),
        )
        .unwrap();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "web-loopback",
            "web_search",
            serde_json::json!({"query":"rust agent", "max_results":3}),
        )))
        .unwrap();
        assert!(response.ok, "{response:?}");
        let receipt: Value = serde_json::from_str(&response.result.unwrap().text).unwrap();
        assert_eq!(receipt["result_count"], 1);
        assert_eq!(
            receipt["results"][0]["url"],
            "https://docs.example.test/rust"
        );
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /search?"));
        assert!(request.contains("q=rust"));
        assert!(request.contains("format=json"));
    }

    #[test]
    fn workspace_scope_supports_the_bounded_file_lifecycle() {
        let (root, host) = host();
        let mkdir: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "mkdir-workspace",
            "mkdir",
            serde_json::json!({"path":"generated/deep", "recursive":true}),
        )))
        .unwrap();
        assert!(mkdir.ok, "{mkdir:?}");

        let write: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "write-workspace",
            "write_file",
            serde_json::json!({"path":"generated/deep/new.txt", "content":"first\n", "create":true}),
        )))
        .unwrap();
        assert!(write.ok, "{write:?}");
        let created = root.0.join("generated/deep/new.txt");
        assert_eq!(fs::read_to_string(&created).unwrap(), "first\n");

        let replace: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "replace-workspace",
            "write_file",
            serde_json::json!({
                "path":"generated/deep/new.txt",
                "content":"second\n",
                "expected_sha256":sha256_hex(b"first\n")
            }),
        )))
        .unwrap();
        assert!(replace.ok, "{replace:?}");

        let rename: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "rename-workspace",
            "rename",
            serde_json::json!({"from":"generated/deep/new.txt", "to":"generated/renamed.txt"}),
        )))
        .unwrap();
        assert!(rename.ok, "{rename:?}");
        let renamed = root.0.join("generated/renamed.txt");
        assert_eq!(fs::read_to_string(&renamed).unwrap(), "second\n");

        let delete_file: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "delete-file-workspace",
            "delete",
            serde_json::json!({"path":"generated/renamed.txt"}),
        )))
        .unwrap();
        assert!(delete_file.ok, "{delete_file:?}");
        assert!(!renamed.exists());

        let delete_tree: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "delete-tree-workspace",
            "delete",
            serde_json::json!({"path":"generated", "recursive":true}),
        )))
        .unwrap();
        assert!(delete_tree.ok, "{delete_tree:?}");
        assert!(!root.0.join("generated").exists());
    }

    #[test]
    fn workspace_mutations_reject_escape_root_and_control_paths() {
        let (root, host) = host();
        fs::create_dir(root.0.join(".git")).unwrap();
        fs::write(root.0.join(".git/config"), "protected\n").unwrap();

        for (id, operation, args) in [
            (
                "write-parent",
                "write_file",
                serde_json::json!({"path":"../escape.txt", "content":"no\n", "create":true}),
            ),
            (
                "mkdir-control",
                "mkdir",
                serde_json::json!({"path":".tina/hidden"}),
            ),
            (
                "write-git",
                "write_file",
                serde_json::json!({
                    "path":".git/config",
                    "content":"changed\n",
                    "expected_sha256":sha256_hex(b"protected\n")
                }),
            ),
            (
                "rename-git",
                "rename",
                serde_json::json!({"from":".git/config", "to":"config-away"}),
            ),
            (
                "delete-root",
                "delete",
                serde_json::json!({"path":".", "recursive":true}),
            ),
        ] {
            let response: ToolResponse =
                serde_json::from_str(&host.handle_line(&request(id, operation, args))).unwrap();
            assert!(!response.ok, "{id} unexpectedly succeeded");
            assert_eq!(response.error.unwrap().code, "workspace_denied", "{id}");
        }
        assert_eq!(
            fs::read_to_string(root.0.join(".git/config")).unwrap(),
            "protected\n"
        );
        assert!(root.0.is_dir());
    }

    #[test]
    fn workspace_mutations_reject_link_components() {
        let (root, host) = host();
        let outside = TempRoot::new();
        fs::write(outside.0.join("outside.txt"), "outside\n").unwrap();
        let internal_target = root.0.join("internal-target");
        fs::create_dir(&internal_target).unwrap();
        fs::write(internal_target.join("inside.txt"), "inside\n").unwrap();
        let link = root.0.join("linked");
        let internal_link = root.0.join("internal-linked");
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside.0, &link).unwrap();
            std::os::unix::fs::symlink(&internal_target, &internal_link).unwrap();
        }
        #[cfg(windows)]
        {
            if std::os::windows::fs::symlink_dir(&outside.0, &link).is_err()
                || std::os::windows::fs::symlink_dir(&internal_target, &internal_link).is_err()
            {
                return;
            }
        }

        for (id, operation, args) in [
            (
                "write-link",
                "write_file",
                serde_json::json!({
                    "path":"linked/outside.txt",
                    "content":"changed\n",
                    "expected_sha256":sha256_hex(b"outside\n")
                }),
            ),
            (
                "write-internal-link",
                "write_file",
                serde_json::json!({
                    "path":"internal-linked/inside.txt",
                    "content":"changed\n",
                    "expected_sha256":sha256_hex(b"inside\n")
                }),
            ),
            (
                "delete-link",
                "delete",
                serde_json::json!({"path":"linked/outside.txt"}),
            ),
            (
                "rename-link",
                "rename",
                serde_json::json!({"from":"one.txt", "to":"linked/renamed.txt"}),
            ),
        ] {
            let response: ToolResponse =
                serde_json::from_str(&host.handle_line(&request(id, operation, args))).unwrap();
            assert!(!response.ok, "{id} unexpectedly succeeded");
            assert_eq!(response.error.unwrap().code, "workspace_denied", "{id}");
        }
        assert_eq!(
            fs::read_to_string(outside.0.join("outside.txt")).unwrap(),
            "outside\n"
        );
        assert_eq!(
            fs::read_to_string(internal_target.join("inside.txt")).unwrap(),
            "inside\n"
        );
        assert!(root.0.join("one.txt").is_file());
    }

    #[test]
    fn full_machine_scope_requires_explicit_mode_and_supports_file_lifecycle() {
        let anchor = TempRoot::new();
        let workspace = anchor.0.join("workspace");
        let outside = anchor.0.join("outside.txt");
        fs::create_dir_all(&workspace).unwrap();
        fs::write(&outside, "outside\n").unwrap();
        let host = ToolHost::with_scope(&workspace, ToolHostScope::Machine).unwrap();

        let capabilities: Value = serde_json::from_str(
            &serde_json::from_str::<ToolResponse>(&host.handle_line(&request(
                "c-machine",
                "capabilities",
                Value::Null,
            )))
            .unwrap()
            .result
            .unwrap()
            .text,
        )
        .unwrap();
        assert_eq!(capabilities["scope"], "machine");
        for operation in ["write_file", "mkdir", "delete", "rename", "shell"] {
            assert!(capabilities["tools"]
                .as_array()
                .unwrap()
                .iter()
                .any(|tool| tool == operation));
        }
        assert_eq!(capabilities["network"], true);
        assert_eq!(capabilities["shell"], true);

        let (shell, command) = if cfg!(windows) {
            (ToolHostShell::Cmd, "echo tina-shell-ok")
        } else {
            (ToolHostShell::Sh, "printf tina-shell-ok")
        };
        let host =
            ToolHost::with_scope_and_shell(&workspace, ToolHostScope::Machine, shell).unwrap();
        let shell_response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "shell-machine",
            "shell",
            serde_json::json!({"command": command, "timeout_ms": 5_000}),
        )))
        .unwrap();
        assert!(shell_response.ok, "{shell_response:?}");
        let shell_receipt: Value =
            serde_json::from_str(&shell_response.result.unwrap().text).unwrap();
        assert_eq!(shell_receipt["success"], true);
        assert!(shell_receipt["stdout"]
            .as_str()
            .unwrap()
            .contains("tina-shell-ok"));

        let read: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "read-machine",
            "read",
            serde_json::json!({"path": outside}),
        )))
        .unwrap();
        assert!(read.ok, "{read:?}");
        assert_eq!(read.result.unwrap().text, "outside\n");

        let nested = anchor.0.join("created/deep");
        let mkdir: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "mkdir-machine",
            "mkdir",
            serde_json::json!({"path": nested, "recursive": true}),
        )))
        .unwrap();
        assert!(mkdir.ok, "{mkdir:?}");
        let created = nested.join("file.txt");
        let write: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "write-machine",
            "write_file",
            serde_json::json!({"path": created, "content":"new\n", "create":true}),
        )))
        .unwrap();
        assert!(write.ok, "{write:?}");
        assert_eq!(fs::read_to_string(&created).unwrap(), "new\n");

        let renamed = anchor.0.join("renamed.txt");
        let rename: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "rename-machine",
            "rename",
            serde_json::json!({"from": created, "to": renamed}),
        )))
        .unwrap();
        assert!(rename.ok, "{rename:?}");
        assert!(!created.exists());
        assert_eq!(fs::read_to_string(&renamed).unwrap(), "new\n");

        let delete: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "delete-machine",
            "delete",
            serde_json::json!({"path": renamed}),
        )))
        .unwrap();
        assert!(delete.ok, "{delete:?}");
        assert!(!renamed.exists());
    }

    #[test]
    fn full_machine_scope_cannot_alias_the_anchor_or_tina_control_data() {
        let anchor = TempRoot::new();
        let workspace = anchor.0.join("workspace");
        fs::create_dir_all(workspace.join("alias")).unwrap();
        let host = ToolHost::with_scope(&workspace, ToolHostScope::Machine).unwrap();

        let root_alias = workspace.join("alias").join("..");
        let delete_root: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "delete-root-alias",
            "delete",
            serde_json::json!({"path": root_alias, "recursive": true}),
        )))
        .unwrap();
        assert!(!delete_root.ok);
        assert_eq!(delete_root.error.unwrap().code, "workspace_denied");
        assert!(workspace.is_dir());

        let tina_escape = workspace
            .join("not-created")
            .join("..")
            .join(".tina/escape.txt");
        let write_control: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "write-control-alias",
            "write_file",
            serde_json::json!({"path": tina_escape, "content": "escape\n", "create": true}),
        )))
        .unwrap();
        assert!(!write_control.ok);
        assert_eq!(write_control.error.unwrap().code, "workspace_denied");
        assert!(!workspace.join(".tina/escape.txt").exists());
    }

    #[test]
    fn shell_reclaims_background_descendants_after_the_parent_exits() {
        let anchor = TempRoot::new();
        let workspace = anchor.0.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let (shell, command) = if cfg!(windows) {
            (
                ToolHostShell::PowerShell,
                "Start-Process -FilePath cmd.exe -ArgumentList '/d','/c','ping -n 30 127.0.0.1 >nul' -NoNewWindow; Write-Output parent-done",
            )
        } else {
            (ToolHostShell::Sh, "(sleep 30) & printf parent-done")
        };
        let host =
            ToolHost::with_scope_and_shell(&workspace, ToolHostScope::Machine, shell).unwrap();
        let started = Instant::now();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "shell-background",
            "shell",
            serde_json::json!({"command": command, "timeout_ms": 5_000}),
        )))
        .unwrap();
        assert!(response.ok, "{response:?}");
        assert!(started.elapsed() < Duration::from_secs(5));
        let receipt: Value = serde_json::from_str(&response.result.unwrap().text).unwrap();
        assert_eq!(receipt["timed_out"], false);
        assert_eq!(receipt["success"], true);
        assert!(receipt["stdout"].as_str().unwrap().contains("parent-done"));
    }

    #[test]
    fn apply_patch_requires_the_current_hash_and_atomically_replaces_exact_text() {
        let (root, host) = host();
        let before = sha256_hex(b"alpha\nbeta\n");
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p1",
            "apply_patch",
            patch_args("one.txt", &before, "beta", "gamma"),
        )))
        .unwrap();
        assert!(response.ok, "{:?}", response.error);
        assert_eq!(
            fs::read_to_string(root.0.join("one.txt")).unwrap(),
            "alpha\ngamma\n"
        );
        let receipt: Value = serde_json::from_str(&response.result.unwrap().text).unwrap();
        assert_eq!(receipt["before_sha256"], before);
        assert_eq!(receipt["after_sha256"], sha256_hex(b"alpha\ngamma\n"));
        assert_eq!(receipt["edits_applied"], 1);
        assert!(!fs::read_dir(&root.0)
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with(".tina-apply-")));
    }

    #[test]
    fn stale_or_ambiguous_patch_leaves_the_original_unchanged() {
        let (root, host) = host();
        let stale: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p1",
            "apply_patch",
            patch_args("one.txt", &"00".repeat(32), "beta", "gamma"),
        )))
        .unwrap();
        assert!(!stale.ok);
        assert_eq!(stale.error.unwrap().code, "stale_read");

        fs::write(root.0.join("repeat.txt"), "same same\n").unwrap();
        let before = sha256_hex(b"same same\n");
        let ambiguous: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p2",
            "apply_patch",
            patch_args("repeat.txt", &before, "same", "other"),
        )))
        .unwrap();
        assert!(!ambiguous.ok);
        assert_eq!(ambiguous.error.unwrap().code, "patch_conflict");
        assert_eq!(fs::read(root.0.join("repeat.txt")).unwrap(), b"same same\n");
        assert_eq!(fs::read(root.0.join("one.txt")).unwrap(), b"alpha\nbeta\n");
    }

    #[test]
    fn apply_patch_rejects_full_overwrite_binary_and_control_data() {
        let (root, host) = host();
        let before = sha256_hex(b"alpha\nbeta\n");
        let overwrite: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p1",
            "apply_patch",
            patch_args("one.txt", &before, "alpha\nbeta\n", "replacement\n"),
        )))
        .unwrap();
        assert_eq!(overwrite.error.unwrap().code, "patch_conflict");

        fs::write(root.0.join("binary.bin"), [0xff, 0xfe, 0xfd]).unwrap();
        let binary: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p2",
            "apply_patch",
            patch_args("binary.bin", &sha256_hex(&[0xff, 0xfe, 0xfd]), "x", "y"),
        )))
        .unwrap();
        assert_eq!(binary.error.unwrap().code, "tool_error");

        fs::write(root.0.join(".tina/host-state.txt"), "state\n").unwrap();
        let control: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "p3",
            "apply_patch",
            patch_args(
                ".tina/host-state.txt",
                &sha256_hex(b"state\n"),
                "state",
                "changed",
            ),
        )))
        .unwrap();
        assert_eq!(control.error.unwrap().code, "workspace_denied");
        assert_eq!(fs::read(root.0.join("one.txt")).unwrap(), b"alpha\nbeta\n");
    }

    #[test]
    fn search_and_list_are_deterministic() {
        let (_root, host) = host();
        let list: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "l1",
            "list",
            serde_json::json!({"recursive":true}),
        )))
        .unwrap();
        assert_eq!(list.result.unwrap().text, "one.txt\nsrc\nsrc/two.rs\n");
        let search: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "s1",
            "search",
            serde_json::json!({"query":"beta"}),
        )))
        .unwrap();
        assert_eq!(
            search.result.unwrap().text,
            "one.txt:2:beta\nsrc/two.rs:1:fn beta() {}\n"
        );
    }

    #[test]
    fn output_limit_creates_artifact_and_digest() {
        let root = TempRoot::new();
        fs::write(root.0.join("large.txt"), "0123456789").unwrap();
        let host = ToolHost::with_limits(
            &root.0,
            HostLimits {
                inline_output_bytes: 4,
                ..HostLimits::default()
            },
        )
        .unwrap();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "r1",
            "read",
            serde_json::json!({"path":"large.txt"}),
        )))
        .unwrap();
        let result = response.result.unwrap();
        assert!(result.truncated);
        assert_eq!(result.text, "0123");
        assert!(result
            .artifact
            .as_ref()
            .unwrap()
            .starts_with(".tina/tool-artifacts/"));
        assert_eq!(result.sha256, sha256_hex(b"0123456789"));
        let artifact = root.0.join(result.artifact.unwrap());
        assert_eq!(fs::read(artifact).unwrap(), b"0123456789");
    }

    #[test]
    fn output_limit_never_splits_utf8() {
        let root = TempRoot::new();
        fs::write(root.0.join("utf8.txt"), "ab月c").unwrap();
        let host = ToolHost::with_limits(
            &root.0,
            HostLimits {
                inline_output_bytes: 4,
                ..HostLimits::default()
            },
        )
        .unwrap();
        let response: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "r1",
            "read",
            serde_json::json!({"path":"utf8.txt"}),
        )))
        .unwrap();
        let result = response.result.unwrap();
        assert_eq!(result.text, "ab");
        assert!(result.truncated);
        assert_eq!(result.bytes, "ab月c".len());
    }

    #[test]
    fn jsonl_reader_rejects_an_oversized_frame_then_resynchronizes() {
        let root = TempRoot::new();
        let host = ToolHost::with_limits(
            &root.0,
            HostLimits {
                max_request_bytes: 256,
                max_patch_bytes: 64,
                ..HostLimits::default()
            },
        )
        .unwrap();
        let input = format!(
            "{}\n{}\n",
            "x".repeat(300),
            request("c", "capabilities", Value::Null)
        );
        let mut output = Vec::new();
        run_jsonl(&host, Cursor::new(input.into_bytes()), &mut output).unwrap();
        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<ToolResponse>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_eq!(
            responses[0].error.as_ref().unwrap().code,
            "request_too_large"
        );
        assert!(responses[1].ok);
        assert_eq!(responses[1].id, "c");
    }

    #[test]
    fn protocol_errors_are_utf8_safe_and_bounded() {
        let response = ToolResponse::error(
            "r".to_string(),
            "bounded",
            "月".repeat(MAX_ERROR_MESSAGE_BYTES),
        );
        let message = &response.error.unwrap().message;
        assert!(message.len() <= MAX_ERROR_MESSAGE_BYTES);
        assert!(message.is_char_boundary(message.len()));
    }

    #[test]
    fn malformed_and_unknown_requests_fail_closed() {
        let (_root, host) = host();
        let malformed: ToolResponse = serde_json::from_str(&host.handle_line("{")).unwrap();
        assert_eq!(malformed.error.unwrap().code, "invalid_json");
        let unknown: ToolResponse = serde_json::from_str(&host.handle_line(&request(
            "u1",
            "shell",
            serde_json::json!({"command":"echo no"}),
        )))
        .unwrap();
        assert_eq!(unknown.error.unwrap().code, "workspace_denied");
    }
}
