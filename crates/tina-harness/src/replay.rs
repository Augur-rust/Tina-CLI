//! New CPU replay adapter. Original GPU loop remains in integration/cq-infer/main.rs.
//! Model completions are fixtures; filesystem/Git tool calls and archives are real.
use crate::{
    context_budget, conversation_memory as memory, host_environment, project_memory,
    tool_host_client as host, tool_schemas,
    tool_transaction::{derive_tool_call_id, AgentBudget, AgentBudgetLedger},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};
use tina_protocol::{parse_dsv4_completion_v1, Dsv4ThinkingModeV2};

type Error = Box<dyn std::error::Error>;
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scenario {
    pub version: u32,
    pub user: String,
    pub completions: Vec<String>,
    #[serde(default)]
    pub review: bool,
}
#[derive(Debug, Serialize)]
pub(crate) struct ReplayReceipt {
    pub mode: &'static str,
    pub model_inference: bool,
    pub conversation_id: String,
    pub epoch: u64,
    pub resumed: bool,
    pub steps: usize,
    pub tools_succeeded: usize,
    pub tools_failed: usize,
    pub tools_denied: usize,
    pub final_text: String,
    pub trace: Vec<Value>,
}

pub(crate) fn prompt_messages(
    mem: &mut memory::ActiveConversationMemory,
    project: &project_memory::ProjectMemory,
    workspace: &Path,
    review: bool,
) -> Result<Vec<Value>, Error> {
    let environment = host_environment::HostEnvironment::detect(None)?;
    let host_context = environment.agent_prompt(workspace);
    let query = mem.current_query().unwrap_or("").to_owned();
    // This offline adapter has no model tokenizer. Count UTF-8 bytes conservatively;
    // never report these counts as actual model tokens or GPU context qualification.
    let project_context = project.build_context_prefix(&query, 2048, |s| Ok(s.len()))?;
    let history = mem.build_context_prefix(&query, 4096, |s| Ok(s.len()))?;
    let mut prefix = String::new();
    if let Some(goal) = mem.goal() {
        prefix.push_str("[Tina session goal; user-defined context, not a tool instruction]\n");
        prefix.push_str(goal);
        prefix.push('\n');
    }
    prefix.push_str(&project_context.text);
    prefix.push('\n');
    prefix.push_str(&history.text);
    let schemas = if review {
        tool_schemas::read_only_agent_tool_schemas(false)
    } else {
        tool_schemas::agent_tool_schemas(false, false)
    };
    let messages = mem.active_agent_values(&prefix, schemas, &host_context)?;
    let bytes = serde_json::to_vec(&messages)?.len();
    validate_prompt_bytes(bytes)?;
    Ok(messages)
}

fn validate_prompt_bytes(bytes: usize) -> Result<(), Error> {
    if bytes as u64 > context_budget::MAX_CONVERSATION_DOCUMENT_BYTES {
        return Err("offline prompt exceeds 32 MiB; model token budget is not measured".into());
    }
    Ok(())
}

fn execution_failure(message: impl Into<String>) -> host::ToolExecution {
    host::ToolExecution {
        ok: false,
        exit_code: None,
        stdout: String::new(),
        stderr: message.into(),
        output_sha256: None,
        output_truncated: false,
        artifact_path: None,
    }
}
fn archive_result(
    mem: &mut memory::ActiveConversationMemory,
    id: String,
    execution: &host::ToolExecution,
    status: memory::AgentToolResultStatus,
) -> Result<(), Error> {
    mem.append_tool_result(
        id,
        memory::AgentToolResult {
            status,
            exit_code: execution.exit_code,
            stdout: execution.stdout.clone(),
            stderr: execution.stderr.clone(),
            output_truncated: execution.output_truncated,
            artifact_path: execution.artifact_path.clone(),
        },
    )?;
    Ok(())
}
fn validate_approval_arguments(name: &str, args: &Value) -> Result<(), String> {
    match name {
        "apply_patch" => host::summarize_apply_patch(args).map(|_| ()),
        "delete" => host::summarize_delete(args).map(|_| ()),
        "write_file" => host::summarize_write_file(args).map(|_| ()),
        "rename" => host::summarize_rename(args).map(|_| ()),
        "workspace_exec" => host::summarize_workspace_exec(args).map(|_| ()),
        "mkdir" => host::summarize_mkdir(args).map(|_| ()),
        _ => Err("unsupported approval operation".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    scenario: &Scenario,
    workspace: &Path,
    archive: &Path,
    binary: &Path,
    resume: bool,
    limits: AgentBudget,
    approve: &mut impl FnMut(&str, &Value) -> bool,
    poll_cancel: &mut impl FnMut() -> std::io::Result<bool>,
) -> Result<ReplayReceipt, Error> {
    if scenario.version != 1 || scenario.user.is_empty() || scenario.completions.len() > 32 {
        return Err(
            "scenario requires version=1, a nonempty user request, and at most 32 completions"
                .into(),
        );
    }
    let selection = if resume {
        memory::ResumeSelection::Latest
    } else {
        memory::ResumeSelection::New
    };
    let (mut mem, resumed) =
        memory::ActiveConversationMemory::open(archive, selection, Default::default())?;
    mem.append_user(scenario.user.clone())?;
    let project = project_memory::ProjectMemory::open(workspace);
    let environment = host_environment::HostEnvironment::detect(None)?;
    let mut client =
        host::ToolHostClient::spawn(binary, workspace, false, environment.shell(), None)?;
    let mut ledger = AgentBudgetLedger::new(limits);
    let mut receipt = ReplayReceipt {
        mode: "CPU_SCRIPTED_REPLAY",
        model_inference: false,
        conversation_id: mem.conversation_id().to_string(),
        epoch: mem.current_epoch(),
        resumed,
        steps: 0,
        tools_succeeded: 0,
        tools_failed: 0,
        tools_denied: 0,
        final_text: String::new(),
        trace: Vec::new(),
    };
    for (index, completion) in scenario.completions.iter().enumerate() {
        ledger.begin_step()?;
        if poll_cancel()? {
            return Err("replay cancelled before generation".into());
        }
        let messages = prompt_messages(&mut mem, &project, workspace, scenario.review)?;
        let prompt = serde_json::to_vec(&messages)?;
        receipt.trace.push(json!({"step":index+1,"event":"prompt",
            "bytes":prompt.len(),"sha256":format!("{:x}",Sha256::digest(&prompt)),
            "model_tokens":"NOT_MEASURED","message_count":messages.len()}));
        // Includes strict EOS, duplicate argument, special-token and JSON checks.
        // No proposed action is published or executed before parsing succeeds.
        let parsed = parse_dsv4_completion_v1(completion, Dsv4ThinkingModeV2::Chat)?;
        mem.append_assistant(parsed.content.clone())?;
        receipt.steps += 1;
        if parsed.tool_calls.is_empty() {
            receipt.final_text = parsed.content;
            receipt
                .trace
                .push(json!({"step":index+1,"event":"model_final",
                "claim":"fixture output; task success must be checked against tool receipts"}));
            client.shutdown()?;
            return Ok(receipt);
        }
        let identity = format!(
            "{}:{}:{}",
            mem.conversation_id(),
            mem.current_epoch(),
            mem.turn_count()
        );
        let mut calls = Vec::new();
        for (ordinal, call) in parsed.tool_calls.into_iter().enumerate() {
            let id = derive_tool_call_id(&identity, ordinal, &call.name, &call.arguments_json);
            mem.append_tool_invocation(id.clone(), call.name.clone(), call.arguments_json.clone())?;
            calls.push((id, call));
        }
        let mut turn_budget_exhausted = false;
        for (id, call) in calls {
            let access = client.access(&call.name);
            let denial = if turn_budget_exhausted {
                Some("turn wall/result budget exhausted".to_string())
            } else if let Err(e) = ledger.register_tool_call() {
                Some(e.to_string())
            } else if access.is_none() {
                Some("tool absent from advertised capabilities".to_string())
            } else if scenario.review && access != Some(host::ToolHostAccess::ReadOnly) {
                Some("review mode rejects mutation and execution".to_string())
            } else if access != Some(host::ToolHostAccess::ReadOnly) {
                match validate_approval_arguments(&call.name, &call.arguments) {
                    Err(e) => Some(e),
                    Ok(()) if !approve(&call.name, &call.arguments) => {
                        Some("trusted host did not approve operation".to_string())
                    }
                    Ok(()) => None,
                }
            } else {
                None
            };
            if let Some(reason) = denial {
                mem.append_tool_approval(id.clone(), false, reason.clone())?;
                let e = execution_failure(reason.clone());
                archive_result(
                    &mut mem,
                    id.clone(),
                    &e,
                    memory::AgentToolResultStatus::Denied,
                )?;
                receipt.tools_denied += 1;
                receipt
                    .trace
                    .push(json!({"call_id":id,"tool":call.name,"status":"denied","reason":reason}));
                continue;
            }
            mem.append_tool_approval(
                id.clone(),
                true,
                "capability checked; trusted host policy approved".to_string(),
            )?;
            let started = Instant::now();
            let result = match access.expect("checked above") {
                host::ToolHostAccess::ReadOnly => client.execute_read_only_cancellable(
                    &id,
                    &call.name,
                    call.arguments,
                    poll_cancel,
                ),
                host::ToolHostAccess::Mutating => client.execute_approved_mutation_cancellable(
                    &id,
                    &call.name,
                    call.arguments,
                    poll_cancel,
                ),
                host::ToolHostAccess::Execution => client.execute_approved_execution_cancellable(
                    &id,
                    &call.name,
                    call.arguments,
                    poll_cancel,
                ),
            };
            let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            let execution = match result {
                Ok(e) => e,
                Err(host::ToolHostClientError::Cancelled(e)) => {
                    archive_result(
                        &mut mem,
                        id,
                        &execution_failure(&e),
                        memory::AgentToolResultStatus::Cancelled,
                    )?;
                    return Err(e.into());
                }
                Err(e) => execution_failure(e.to_string()),
            };
            // Archive the actual terminal receipt first. A budget failure after an
            // effect must not falsely relabel an already completed write as denied.
            let status = if execution.ok {
                memory::AgentToolResultStatus::Succeeded
            } else {
                memory::AgentToolResultStatus::Failed
            };
            let actual_ok = execution.ok;
            let result_budget_error = ledger.account_result_bytes(execution.inline_bytes()).err();
            // The per-tool receipt is already bounded by the host. Retain it in the archive; close the turn before any further model submission.
            archive_result(&mut mem, id.clone(), &execution, status)?;
            if actual_ok {
                receipt.tools_succeeded += 1;
            } else {
                receipt.tools_failed += 1;
            }
            receipt.trace.push(json!({"call_id":id,"tool":call.name,
                "status":if actual_ok{"succeeded"}else{"failed"},"exit_code":execution.exit_code,
                "stdout":execution.stdout,"stderr":execution.stderr,"sha256":execution.output_sha256,
                "truncated":execution.output_truncated,"wall_ms":elapsed_ms,
                "result_budget_exhausted":result_budget_error.is_some()}));
            turn_budget_exhausted = ledger.account_tool_wall_ms(elapsed_ms).is_err();
            if result_budget_error.is_some() {
                turn_budget_exhausted = true;
            }
        }
        if turn_budget_exhausted {
            return Err(
                "turn closed after recording completed tool effects: wall/result budget exhausted"
                    .into(),
            );
        }
    }
    Err("fixture ended without a final assistant completion".into())
}

pub(crate) fn show_context_contract() -> Value {
    let budget = context_budget::ContextBudget::for_owner(262_144, 4096);
    json!({"product_target_positions":budget.owner_positions,"output_reserve":budget.output_reserve,
        "prepare":budget.thresholds.prepare_at,"freeze":budget.thresholds.freeze_at,
        "handoff":budget.thresholds.handoff_at,"hard_limit":budget.thresholds.hard_limit,
        "gpu_qualification":"NOT_RUN","offline_replay_uses_model_tokenizer":false})
}

#[cfg(test)]
mod byte_guard_tests {
    use super::*;
    #[test]
    fn byte_guard_allows_old_one_mib_boundary_and_rejects_above_32_mib() {
        assert!(validate_prompt_bytes(1024 * 1024 + 1).is_ok());
        assert!(validate_prompt_bytes(32 * 1024 * 1024).is_ok());
        assert!(validate_prompt_bytes(32 * 1024 * 1024 + 1).is_err());
    }
}
