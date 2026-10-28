use super::*;
use crate::replay::{run, Scenario};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Sandbox {
    root: PathBuf,
    workspace: PathBuf,
    archive: PathBuf,
}
impl Sandbox {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tina-loop-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        let workspace = root.join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::write(
            workspace.join("sample.txt"),
            "alpha\nreceipt-first harness\n",
        )
        .unwrap();
        let archive = root.join("archive");
        Self {
            root,
            workspace,
            archive,
        }
    }
    fn host() -> PathBuf {
        std::env::var_os("TINA_TEST_TOOL_HOST")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_exe()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .join(if cfg!(windows) {
                        "cq-tool-host.exe"
                    } else {
                        "cq-tool-host"
                    })
            })
    }
}
impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}
fn call(name: &str, args: serde_json::Value) -> String {
    let mut s = format!("checking\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"{name}\">\n");
    for (k, v) in args.as_object().unwrap() {
        s.push_str(&format!(
            "<｜DSML｜parameter name=\"{k}\" string=\"{}\">{}</｜DSML｜parameter>\n",
            v.is_string(),
            v.as_str()
                .map(str::to_string)
                .unwrap_or_else(|| v.to_string())
        ));
    }
    s.push_str("</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>");
    s
}
fn scenario(completion: String) -> Scenario {
    Scenario {
        version: 1,
        user: "Inspect the sample file".into(),
        completions: vec![
            completion,
            "Done after inspecting the receipt.<｜end▁of▁sentence｜>".into(),
        ],
        review: false,
    }
}
#[test]
fn replay_read_receipt_round_trip_and_resume() {
    let s = Sandbox::new();
    let scenario = scenario(call("read", json!({"path":"sample.txt"})));
    let a = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(a.tools_succeeded, 1);
    assert_eq!(a.steps, 2);
    assert!(!a.model_inference);
    assert!(a.trace.iter().any(|e| e["stdout"]
        .as_str()
        .is_some_and(|x| x.contains("receipt-first harness"))));
    let b = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        true,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert!(b.resumed);
    assert_eq!(a.conversation_id, b.conversation_id);
    assert!(b.epoch > a.epoch);
}
#[test]
fn replay_denies_write_even_when_fixture_requests_it() {
    let s = Sandbox::new();
    let scenario = scenario(call(
        "write_file",
        json!({"path":"new.txt","content":"unauthorized","create":true}),
    ));
    let r = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(r.tools_denied, 1);
    assert!(!s.workspace.join("new.txt").exists());
}
#[test]
fn replay_review_rejects_write_before_approval_callback() {
    let s = Sandbox::new();
    let mut scenario = scenario(call(
        "write_file",
        json!({"path":"new.txt","content":"bad","create":true}),
    ));
    scenario.review = true;
    let r = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| panic!("review must not ask for approval"),
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(r.tools_denied, 1);
    assert!(!s.workspace.join("new.txt").exists());
}
#[test]
fn replay_path_escape_returns_failure_without_reading_outside() {
    let s = Sandbox::new();
    fs::write(s.root.join("outside.txt"), "outside sentinel").unwrap();
    let scenario = scenario(call("read", json!({"path":"../outside.txt"})));
    let r = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(r.tools_failed, 1);
    assert!(!serde_json::to_string(&r)
        .unwrap()
        .contains("outside sentinel"));
}
#[test]
fn replay_incomplete_completion_never_executes_tool() {
    let s = Sandbox::new();
    let mut scenario = scenario(call(
        "write_file",
        json!({"path":"new.txt","content":"bad","create":true}),
    ));
    scenario.completions[0] = scenario.completions[0].replace(tina_protocol::M114_EOS, "");
    let result = run(
        &scenario,
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| panic!("parse before approval"),
        &mut || Ok(false),
    );
    assert!(result.unwrap_err().to_string().contains("EOS"));
    assert!(!s.workspace.join("new.txt").exists());
}
#[test]
fn replay_unknown_tool_is_denied() {
    let s = Sandbox::new();
    let r = run(
        &scenario(call("spawn_agent", json!({"task":"invent tool"}))),
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(r.tools_denied, 1);
}
#[test]
fn replay_step_limit_stops_without_false_completion() {
    let s = Sandbox::new();
    let limits = tool_transaction::AgentBudget {
        max_steps: 1,
        ..Default::default()
    };
    let r = run(
        &scenario(call("read", json!({"path":"sample.txt"}))),
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        limits,
        &mut |_, _| false,
        &mut || Ok(false),
    );
    assert!(r.unwrap_err().to_string().contains("Steps"));
}
#[test]
fn replay_zero_call_budget_denies_before_dispatch() {
    let s = Sandbox::new();
    let limits = tool_transaction::AgentBudget {
        max_tool_calls: 0,
        ..Default::default()
    };
    let r = run(
        &scenario(call("read", json!({"path":"sample.txt"}))),
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        limits,
        &mut |_, _| false,
        &mut || Ok(false),
    )
    .unwrap();
    assert_eq!(r.tools_denied, 1);
    assert_eq!(r.tools_succeeded, 0);
}
#[test]
fn replay_post_effect_budget_failure_keeps_actual_write_status() {
    let s = Sandbox::new();
    let limits = tool_transaction::AgentBudget {
        max_result_bytes: 0,
        ..Default::default()
    };
    let r = run(
        &scenario(call(
            "write_file",
            json!({"path":"new.txt","content":"approved test only","create":true}),
        )),
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        limits,
        &mut |_, _| true,
        &mut || Ok(false),
    );
    assert!(r.is_err());
    assert_eq!(
        fs::read_to_string(s.workspace.join("new.txt")).unwrap(),
        "approved test only"
    );
    let (archive, _) = conversation_memory::ConversationArchive::open(
        &s.archive,
        conversation_memory::ResumeSelection::Latest,
    )
    .unwrap();
    assert!(archive
        .agent_events()
        .iter()
        .any(|e| serde_json::to_value(e).unwrap()["body"]["event"]["status"] == "succeeded"));
}
#[test]
fn replay_cancellation_prevents_tool_execution() {
    let s = Sandbox::new();
    let r = run(
        &scenario(call("read", json!({"path":"sample.txt"}))),
        &s.workspace,
        &s.archive,
        &Sandbox::host(),
        false,
        Default::default(),
        &mut |_, _| false,
        &mut || Ok(true),
    );
    assert!(r.unwrap_err().to_string().contains("cancelled"));
}

#[test]
fn large_prompt_preserves_user_content_above_old_byte_ceiling() {
    let s = Sandbox::new();
    let (mut mem, _) = conversation_memory::ActiveConversationMemory::open(
        &s.archive,
        conversation_memory::ResumeSelection::New,
        Default::default(),
    )
    .unwrap();
    let content = "x".repeat(1024 * 1024 + 128);
    mem.append_user(content.clone()).unwrap();
    let project = project_memory::ProjectMemory::open(&s.workspace);
    let messages = replay::prompt_messages(&mut mem, &project, &s.workspace, false).unwrap();
    assert!(serde_json::to_vec(&messages).unwrap().len() > 1024 * 1024);
    assert!(messages.iter().any(|message| message["role"] == "user"
        && message["content"]
            .as_str()
            .is_some_and(|text| text.ends_with(&content))));
}
