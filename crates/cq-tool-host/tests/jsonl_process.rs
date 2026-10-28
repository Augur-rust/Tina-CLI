use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use cq_tool_host::ToolResponse;
use sha2::{Digest, Sha256};

struct TempWorkspace(PathBuf);

impl TempWorkspace {
    fn new() -> Self {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cq-tool-host-process-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(path.join("src")).unwrap();
        fs::write(path.join("src/lib.rs"), "pub fn answer() -> u32 { 42 }\n").unwrap();
        Self(path)
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        assert!(self.0.starts_with(std::env::temp_dir()));
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn exchange(
    stdin: &mut impl Write,
    stdout: &mut impl BufRead,
    request: serde_json::Value,
) -> ToolResponse {
    serde_json::to_writer(&mut *stdin, &request).unwrap();
    stdin.write_all(b"\n").unwrap();
    stdin.flush().unwrap();
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    assert!(line.ends_with('\n'));
    serde_json::from_str(&line).unwrap()
}

fn request_frame(id: &str, op: &str, args: serde_json::Value) -> serde_json::Value {
    let deadline_unix_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
        + 60_000;
    serde_json::json!({
        "v": 2,
        "kind": "request",
        "id": id,
        "op": op,
        "deadline_unix_ms": deadline_unix_ms,
        "args": args,
    })
}

#[test]
fn binary_handshake_read_patch_and_search_share_one_bounded_jsonl_session() {
    let workspace = TempWorkspace::new();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cq-tool-host"))
        .arg("--workspace")
        .arg(&workspace.0)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let capabilities = exchange(
        &mut stdin,
        &mut stdout,
        request_frame("hello", "capabilities", serde_json::Value::Null),
    );
    assert!(capabilities.ok);
    let body: serde_json::Value = serde_json::from_str(&capabilities.result.unwrap().text).unwrap();
    assert_eq!(body["schema"], "tina.tool-host.v2");
    assert_eq!(body["writes"], true);
    assert_eq!(body["network"], false);
    assert_eq!(
        body["tools"],
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

    let read = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "read-1",
            "read",
            serde_json::json!({"path":"src/lib.rs","start_line":1,"end_line":1}),
        ),
    );
    assert!(read.ok);
    assert_eq!(read.result.unwrap().text, "pub fn answer() -> u32 { 42 }");

    let before = format!("{:x}", Sha256::digest(b"pub fn answer() -> u32 { 42 }\n"));
    let patch = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "patch-1",
            "apply_patch",
            serde_json::json!({
                "path":"src/lib.rs",
                "expected_sha256":before,
                "edits":[{"old":"42", "new":"43"}]
            }),
        ),
    );
    assert!(patch.ok, "{:?}", patch.error);
    assert_eq!(
        fs::read_to_string(workspace.0.join("src/lib.rs")).unwrap(),
        "pub fn answer() -> u32 { 43 }\n"
    );

    let write = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "write-workspace",
            "write_file",
            serde_json::json!({"path":"created.txt", "content":"created\n", "create":true}),
        ),
    );
    assert!(write.ok, "{:?}", write.error);
    let rename = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "rename-workspace",
            "rename",
            serde_json::json!({"from":"created.txt", "to":"renamed.txt"}),
        ),
    );
    assert!(rename.ok, "{:?}", rename.error);
    let delete = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "delete-workspace",
            "delete",
            serde_json::json!({"path":"renamed.txt"}),
        ),
    );
    assert!(delete.ok, "{:?}", delete.error);
    assert!(!workspace.0.join("renamed.txt").exists());

    let search = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "search-1",
            "search",
            serde_json::json!({"path":"src","query":"43","max_matches":4}),
        ),
    );
    assert!(search.ok);
    assert_eq!(
        search.result.unwrap().text,
        "src/lib.rs:1:pub fn answer() -> u32 { 43 }\n"
    );

    drop(stdin);
    let status = child.wait().unwrap();
    if !status.success() {
        let mut stderr = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        panic!("cq-tool-host failed: {status}: {stderr}");
    }
}

#[test]
fn full_access_binary_advertises_and_executes_the_machine_surface() {
    let workspace = TempWorkspace::new();
    let shell = if cfg!(windows) { "cmd" } else { "sh" };
    let command = if cfg!(windows) {
        "echo binary-full-access"
    } else {
        "printf binary-full-access"
    };
    let mut child = Command::new(env!("CARGO_BIN_EXE_cq-tool-host"))
        .arg("--workspace")
        .arg(&workspace.0)
        .arg("--full-access")
        .arg("--shell")
        .arg(shell)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());

    let capabilities = exchange(
        &mut stdin,
        &mut stdout,
        request_frame("hello-full", "capabilities", serde_json::Value::Null),
    );
    assert!(capabilities.ok);
    let body: serde_json::Value = serde_json::from_str(&capabilities.result.unwrap().text).unwrap();
    assert_eq!(body["scope"], "machine");
    assert_eq!(body["network"], true);
    assert_eq!(body["shell"], true);
    for operation in [
        "write_file",
        "mkdir",
        "delete",
        "rename",
        "workspace_exec",
        "shell",
    ] {
        assert!(body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool == operation));
    }

    let created = workspace.0.join("created.txt");
    let write = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "write-full",
            "write_file",
            serde_json::json!({"path": created, "content": "full\n", "create": true}),
        ),
    );
    assert!(write.ok, "{:?}", write.error);
    assert_eq!(fs::read_to_string(&created).unwrap(), "full\n");

    let shell_result = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "shell-full",
            "shell",
            serde_json::json!({"command": command, "timeout_ms": 5_000}),
        ),
    );
    assert!(shell_result.ok, "{:?}", shell_result.error);
    let receipt: serde_json::Value =
        serde_json::from_str(&shell_result.result.unwrap().text).unwrap();
    assert_eq!(receipt["success"], true);
    assert!(receipt["stdout"]
        .as_str()
        .unwrap()
        .contains("binary-full-access"));

    let delete = exchange(
        &mut stdin,
        &mut stdout,
        request_frame(
            "delete-full",
            "delete",
            serde_json::json!({"path": created}),
        ),
    );
    assert!(delete.ok, "{:?}", delete.error);
    assert!(!created.exists());

    drop(stdin);
    let status = child.wait().unwrap();
    if !status.success() {
        let mut stderr = String::new();
        child
            .stderr
            .take()
            .unwrap()
            .read_to_string(&mut stderr)
            .unwrap();
        panic!("cq-tool-host failed: {status}: {stderr}");
    }
}

#[test]
fn configured_web_search_is_advertised_without_full_machine_network_access() {
    let workspace = TempWorkspace::new();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cq-tool-host"))
        .arg("--workspace")
        .arg(&workspace.0)
        .arg("--web-search-url")
        .arg("https://search.example.test/search")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let capabilities = exchange(
        &mut stdin,
        &mut stdout,
        request_frame("hello-web", "capabilities", serde_json::Value::Null),
    );
    assert!(capabilities.ok);
    let body: serde_json::Value = serde_json::from_str(&capabilities.result.unwrap().text).unwrap();
    assert_eq!(body["scope"], "workspace");
    assert_eq!(body["network"], false);
    assert_eq!(body["shell"], false);
    assert_eq!(body["web_search"], true);
    assert!(body["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool == "web_search"));

    drop(stdin);
    assert!(child.wait().unwrap().success());
}
