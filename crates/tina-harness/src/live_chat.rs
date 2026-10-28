//! Live Agent loop: terminal -> model stream -> approved tool -> immutable receipt.
use crate::{
    context_budget::ContextBudget,
    conversation_memory as memory, host_environment,
    live_config::{self, LiveConfig},
    model_client::{ModelClient, PieceKind, ToolCall},
    project_memory::ProjectMemory,
    tina_cli::{self, TinaCli},
    todo_cli, tool_host_client as host, tool_schemas,
    tool_transaction::{derive_tool_call_id, AgentBudget, AgentBudgetLedger},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    io::{self, IsTerminal, Write},
    path::PathBuf,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const HELP:&str="Tina live commands / 交互命令\n/help /帮助       /new /新建       /sessions /会话       /resume [ID]\n/context /上下文  /status /状态    /compact /压缩        /retry /重试\n/tools /工具      /agent-restart   /environment          /diff /review [focus]\n/thinking off|low|medium|high|xhigh   /goal set TEXT|clear\n/remember TEXT    /memory          /project-memory [remember TEXT|forget ID]\n/todo [add|start|done|reopen|remove]  /export markdown|jsonl PATH\n/language zh|en   /id NAME         /exit\nCtrl-C cancels the active HTTP request or tool; partial model output is not committed.\nWrites require APPLY, deletion DELETE, command execution EXECUTE. No automatic approvals.";
struct Session {
    config: LiveConfig,
    ui: TinaCli,
    model: ModelClient,
    mem: memory::ActiveConversationMemory,
    project: ProjectMemory,
    host: host::ToolHostClient,
    workspace: PathBuf,
    context_tokens: usize,
    context_source: &'static str,
    observed_prompt: Option<PromptObservation>,
    max_steps: usize,
    chat_only: bool,
    read_only: bool,
    last_metrics: Value,
}
struct PromptObservation {
    tokens: usize,
    bytes: usize,
    message_count: usize,
    prefix_digest: [u8; 32],
}
fn prompt_digest(messages: &[Value], tools: &[Value]) -> Result<[u8; 32]> {
    Ok(Sha256::digest(serde_json::to_vec(
        &json!({"messages":messages,"tools":tools}),
    )?)
    .into())
}
impl PromptObservation {
    fn capture(tokens: usize, messages: &[Value], tools: &[Value], bytes: usize) -> Result<Self> {
        Ok(Self {
            tokens,
            bytes,
            message_count: messages.len(),
            prefix_digest: prompt_digest(messages, tools)?,
        })
    }
    fn estimate(&self, messages: &[Value], tools: &[Value], bytes: usize) -> Result<Option<usize>> {
        if bytes < self.bytes
            || messages.len() < self.message_count
            || prompt_digest(&messages[..self.message_count], tools)? != self.prefix_digest
        {
            return Ok(None);
        }
        Ok(Some(
            self.tokens
                .saturating_add(bytes - self.bytes)
                .saturating_add(256),
        ))
    }
}
fn fail(message: impl Into<String>) -> host::ToolExecution {
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
fn store_result(
    mem: &mut memory::ActiveConversationMemory,
    id: String,
    e: &host::ToolExecution,
    status: memory::AgentToolResultStatus,
) -> Result<()> {
    mem.append_tool_result(
        id,
        memory::AgentToolResult {
            status,
            exit_code: e.exit_code,
            stdout: e.stdout.clone(),
            stderr: e.stderr.clone(),
            output_truncated: e.output_truncated,
            artifact_path: e.artifact_path.clone(),
        },
    )?;
    Ok(())
}
fn approve(ui: &TinaCli, name: &str, args: &Value) -> Result<bool> {
    Ok(match name {
        "apply_patch" => {
            let s = host::summarize_apply_patch(args)?;
            ui.confirm_apply_patch(tina_cli::ApplyPatchConfirmation {
                target: &s.path,
                expected_sha256: &s.expected_sha256,
                edit_count: s.edit_count,
                removed_bytes: s.removed_bytes,
                added_bytes: s.added_bytes,
                preview: &s.preview,
                preview_truncated: s.preview_truncated,
            })?
        }
        "write_file" => {
            let s = host::summarize_write_file(args)?;
            ui.confirm_write_file(tina_cli::WriteFileConfirmation {
                target: &s.path,
                create: s.create,
                expected_sha256: s.expected_sha256.as_deref(),
                content_bytes: s.content_bytes,
                content_sha256: &s.content_sha256,
                preview: &s.preview,
                preview_truncated: s.preview_truncated,
            })?
        }
        "delete" => {
            let s = host::summarize_delete(args)?;
            ui.confirm_delete(&format!("{} (recursive={})", s.path, s.recursive))?
        }
        "mkdir" | "rename" => {
            let s = if name == "mkdir" {
                host::summarize_mkdir(args)?
            } else {
                host::summarize_rename(args)?
            };
            ui.confirm_workspace_mutation(s.operation, &s.detail)?
        }
        "workspace_exec" => {
            let s = host::summarize_workspace_exec(args)?;
            ui.confirm_workspace_exec(&s.detail)?
        }
        _ => false,
    })
}
impl Session {
    fn budget(&self) -> ContextBudget {
        ContextBudget::for_owner(self.config.context_positions, self.config.max_output_tokens)
    }
    fn reset_usage(&mut self) {
        self.context_tokens = 0;
        self.context_source = "estimated";
        self.observed_prompt = None;
    }
    fn messages(&mut self) -> Result<(Vec<Value>, Vec<Value>, usize)> {
        let query = self.mem.current_query().unwrap_or("").to_string();
        let project = self
            .project
            .build_context_prefix(&query, 2048, |s| Ok(s.len()))?;
        let history = self
            .mem
            .build_context_prefix(&query, 4096, |s| Ok(s.len()))?;
        let mut prefix = String::new();
        if let Some(goal) = self.mem.goal() {
            prefix.push_str("[Tina session goal; user-defined context]\n");
            prefix.push_str(goal);
            prefix.push('\n');
        }
        prefix.push_str(&project.text);
        prefix.push('\n');
        prefix.push_str(&history.text);
        let tools = if self.chat_only {
            Vec::new()
        } else if self.read_only {
            tool_schemas::read_only_agent_tool_schemas(false)
        } else {
            tool_schemas::agent_tool_schemas(false, false)
        };
        let raw = if self.chat_only {
            self.mem.active_values(&prefix)?
        } else {
            let env = host_environment::HostEnvironment::detect(None)?;
            self.mem.active_agent_values(
                &prefix,
                tools.clone(),
                &env.agent_prompt(&self.workspace),
            )?
        };
        let mut messages: Vec<Value> = Vec::new();
        for mut m in raw {
            m.as_object_mut()
                .ok_or("Invalid archived message")?
                .remove("tools");
            if self.config.is_deepseek() && m["role"] == "assistant" {
                // Host status turns / pre-integration archives have no provider reasoning.
                m.as_object_mut()
                    .ok_or("Invalid archived assistant")?
                    .entry("reasoning_content")
                    .or_insert_with(|| json!(""));
            } else if !self.config.is_deepseek() {
                m.as_object_mut()
                    .ok_or("Invalid archived message")?
                    .remove("reasoning_content");
            }
            // Preserve consecutive user input after cancellation without sending an invalid role sequence.
            if let Some(previous) = messages.last_mut() {
                if previous["role"] == "user" && m["role"] == "user" {
                    let joined = format!(
                        "{}\n\n{}",
                        previous["content"].as_str().unwrap_or(""),
                        m["content"].as_str().unwrap_or("")
                    );
                    previous["content"] = joined.into();
                    continue;
                }
            }
            messages.push(m);
        }
        let bytes = serde_json::to_vec(&json!({"messages":messages,"tools":tools}))?.len();
        if bytes > 32 * 1024 * 1024 {
            return Err("Prompt exceeds the 32 MiB document guard".into());
        }
        Ok((messages, tools, bytes))
    }
    fn estimate(&self, messages: &[Value], tools: &[Value], bytes: usize) -> Result<usize> {
        // Byte accounting is conservative, not a tokenizer. Reuse server usage only
        // after verifying the entire previous message prefix and tool schema.
        if let Some(observed) = &self.observed_prompt {
            if let Some(estimate) = observed.estimate(messages, tools, bytes)? {
                return Ok(estimate);
            }
        }
        Ok(bytes.saturating_add(256))
    }
    fn prepare(&mut self) -> Result<Value> {
        let (mut messages, mut tools, mut bytes) = self.messages()?;
        let mut estimate = self.estimate(&messages, &tools, bytes)?;
        let budget = self.budget();
        self.mem.observe_pressure(budget.pressure(estimate));
        if estimate >= budget.thresholds.handoff_at {
            if let Some(h) = self.mem.compact_for_handoff_retaining(Vec::new(), 1)? {
                self.ui.notice(&format!(
                    "Context handoff / 上下文交接: epoch {} -> {}; archived {} messages",
                    h.source_epoch_id, h.target_epoch_id, h.compacted_turns
                ))?;
                self.reset_usage();
                (messages, tools, bytes) = self.messages()?;
                estimate = self.estimate(&messages, &tools, bytes)?;
            }
        }
        if budget.admit_input(estimate).is_err() {
            return Err(format!("Estimated prompt size {estimate} exceeds usable input {}. History was preserved. Use a smaller input, a larger configured context, or reduce --max-tokens. This estimate is not a model tokenizer.",budget.thresholds.hard_limit).into());
        }
        self.context_tokens = estimate;
        self.context_source = "estimated";
        let mut request = json!({"model":self.config.model,"messages":messages,"stream":true,"stream_options":{"include_usage":true},"max_tokens":self.config.max_output_tokens});
        if !tools.is_empty() {
            request["tools"] = tools.into();
            request["tool_choice"] = "auto".into();
            request["parallel_tool_calls"] = false.into();
        }
        self.config.apply_reasoning(&mut request);
        Ok(request)
    }
    fn execute_tools(
        &mut self,
        calls: Vec<ToolCall>,
        ledger: &mut AgentBudgetLedger,
        round: usize,
    ) -> Result<()> {
        let turn_key = format!(
            "{}:{}:{}:{round}",
            self.mem.conversation_id(),
            self.mem.current_epoch(),
            self.mem.turn_count()
        );
        let mut interrupted = false;
        let mut exhausted = false;
        for (ordinal, call) in calls.into_iter().enumerate() {
            let id = derive_tool_call_id(&turn_key, ordinal, &call.name, &call.arguments);
            let args: Value = serde_json::from_str(&call.arguments)?;
            self.mem
                .append_tool_invocation(id.clone(), call.name.clone(), call.arguments)?;
            let access = self.host.access(&call.name);
            let denial = if interrupted {
                Some("Batch cancelled before dispatch".into())
            } else if exhausted || ledger.register_tool_call().is_err() {
                exhausted = true;
                Some("Tool budget exhausted before dispatch".into())
            } else if self.chat_only || access.is_none() {
                Some("Model requested an unadvertised tool".into())
            } else if self.read_only && access != Some(host::ToolHostAccess::ReadOnly) {
                Some("Read-only mode denies this operation".into())
            } else if access != Some(host::ToolHostAccess::ReadOnly) {
                match approve(&self.ui, &call.name, &args) {
                    Ok(true) => None,
                    Ok(false) => Some("User did not approve this operation".into()),
                    Err(e) => Some(e.to_string()),
                }
            } else {
                None
            };
            if let Some(reason) = denial {
                self.mem
                    .append_tool_approval(id.clone(), false, reason.clone())?;
                store_result(
                    &mut self.mem,
                    id,
                    &fail(&reason),
                    memory::AgentToolResultStatus::Denied,
                )?;
                self.ui
                    .tool_notice(&format!("{}: DENIED — {reason}", call.name))?;
                continue;
            }
            self.mem.append_tool_approval(
                id.clone(),
                true,
                "Advertised capability; read-only or explicitly approved by the user".into(),
            )?;
            self.ui.tool_notice(&format!("{}: running", call.name))?;
            let start = Instant::now();
            let result = {
                let _guard = self.ui.begin_generation()?;
                let ui = &self.ui;
                let mut cancel = || ui.poll_generation_cancel();
                match access.unwrap() {
                    host::ToolHostAccess::ReadOnly => {
                        self.host
                            .execute_read_only_cancellable(&id, &call.name, args, &mut cancel)
                    }
                    host::ToolHostAccess::Mutating => self
                        .host
                        .execute_approved_mutation_cancellable(&id, &call.name, args, &mut cancel),
                    host::ToolHostAccess::Execution => self
                        .host
                        .execute_approved_execution_cancellable(&id, &call.name, args, &mut cancel),
                }
            };
            let (receipt, status) = match result {
                Ok(r) => {
                    let status = if r.ok {
                        memory::AgentToolResultStatus::Succeeded
                    } else {
                        memory::AgentToolResultStatus::Failed
                    };
                    (r, status)
                }
                Err(host::ToolHostClientError::Cancelled(e)) => {
                    interrupted = true;
                    (fail(e), memory::AgentToolResultStatus::Cancelled)
                }
                Err(e) => (fail(e.to_string()), memory::AgentToolResultStatus::Failed),
            };
            store_result(&mut self.mem, id, &receipt, status)?;
            exhausted |= ledger.account_result_bytes(receipt.inline_bytes()).is_err();
            exhausted |= ledger
                .account_tool_wall_ms(start.elapsed().as_millis().min(u64::MAX as u128) as u64)
                .is_err();
            let preview = if receipt.ok {
                &receipt.stdout
            } else {
                &receipt.stderr
            };
            self.ui.tool_notice(&format!(
                "{}: {:?}\n{}",
                call.name,
                status,
                preview.chars().take(1800).collect::<String>()
            ))?;
        }
        self.ui.end_tool_batch()?;
        if interrupted {
            self.host.restart()?;
            return Err("Tool batch cancelled. Terminal receipts were retained; unstarted calls were denied.".into());
        }
        if exhausted {
            return Err(
                "Tool budget exhausted. Completed effects and their receipts were retained.".into(),
            );
        }
        Ok(())
    }
    fn turn(&mut self, text: String) -> Result<()> {
        self.mem.append_user(text)?;
        let start = Instant::now();
        let mut first_piece = None;
        let mut generated = 0usize;
        let mut rounds = 0usize;
        let limits = AgentBudget {
            max_steps: self.max_steps,
            ..Default::default()
        };
        let mut ledger = AgentBudgetLedger::new(limits);
        loop {
            ledger.begin_step()?;
            rounds += 1;
            let request = self.prepare()?;
            let prompt_bytes=serde_json::to_vec(&json!({"messages":request["messages"],"tools":request.get("tools").cloned().unwrap_or(json!([]))}))?.len();
            let round_start = Instant::now();
            self.ui.begin_assistant()?;
            let result = {
                let _guard = self.ui.begin_generation()?;
                let ui = &self.ui;
                let mut thinking_open = false;
                let mut piece = |kind, s: &str| -> std::result::Result<(), String> {
                    if first_piece.is_none() {
                        first_piece = Some(start.elapsed().as_secs_f64());
                    }
                    match kind {
                        PieceKind::Reasoning => {
                            if !thinking_open {
                                ui.stream_assistant("[thinking]\n")
                                    .map_err(|e| e.to_string())?;
                                thinking_open = true;
                            }
                        }
                        PieceKind::Content => {
                            if thinking_open {
                                ui.stream_assistant("\n[/thinking]\n")
                                    .map_err(|e| e.to_string())?;
                                thinking_open = false;
                            }
                        }
                    }
                    ui.stream_assistant(s).map_err(|e| e.to_string())
                };
                self.model.stream(&request, &mut piece, &mut || {
                    ui.poll_generation_cancel().map_err(|e| e.to_string())
                })
            };
            let completion = match result {
                Ok(r) => r,
                Err(e) => {
                    // Retry a capacity rejection only after a real context reduction; never replay tool effects.
                    if (e.contains("context_or_template_limit")
                        || e.contains("context_length_exceeded"))
                        && rounds < self.max_steps
                        && self
                            .mem
                            .compact_for_handoff_retaining(Vec::new(), 1)?
                            .is_some()
                    {
                        self.reset_usage();
                        self.ui.end_error(
                            "Server context limit: archived history and retrying a reduced prompt",
                        )?;
                        continue;
                    }
                    return Err(e.into());
                }
            };
            generated = generated.saturating_add(completion.completion_tokens);
            if completion.prompt_tokens > 0 {
                self.context_tokens = completion
                    .prompt_tokens
                    .saturating_add(completion.completion_tokens);
                self.context_source = "server usage";
                self.observed_prompt = Some(PromptObservation::capture(
                    completion.prompt_tokens,
                    request["messages"]
                        .as_array()
                        .ok_or("Missing model messages")?,
                    request
                        .get("tools")
                        .and_then(Value::as_array)
                        .map_or(&[], Vec::as_slice),
                    prompt_bytes,
                )?);
            }
            self.ui.end_assistant(
                completion.completion_tokens,
                round_start.elapsed().as_secs_f64(),
                self.context_tokens,
                self.budget().thresholds.hard_limit,
            )?;
            let reasoning = if self.config.is_deepseek() || !completion.reasoning.is_empty() {
                Some(completion.reasoning)
            } else {
                None
            };
            self.mem
                .append_assistant_with_reasoning(completion.content, reasoning)?;
            if completion.calls.is_empty() {
                self.last_metrics = json!({"schema":"tina.live-turn.v1","conversation_id":self.mem.conversation_id(),"epoch":self.mem.current_epoch(),"model":self.config.model,"rounds":rounds,"completion_tokens":generated,"wall_seconds":start.elapsed().as_secs_f64(),"first_piece_seconds":first_piece,"context_positions":self.context_tokens,"context_source":self.context_source});
                let path = self.mem.archive_root().join("last-live-metrics.json");
                std::fs::write(path, serde_json::to_vec_pretty(&self.last_metrics)?)?;
                return Ok(());
            }
            self.execute_tools(completion.calls, &mut ledger, rounds)?;
        }
    }
    fn run_turn(&mut self, text: String) -> Result<bool> {
        match self.turn(text) {
            Ok(()) => Ok(true),
            Err(e) => {
                let message = self.model.redact(&e.to_string());
                self.ui.end_error(&message)?;
                // An explicit host status closes the dialogue without treating streamed fragments as a model answer.
                self.mem.append_assistant(format!("[Tina host status: generation did not complete. Partial model output was discarded. Completed tool receipts remain in the archive. {message}]"))?;
                Ok(false)
            }
        }
    }
    fn command(&mut self, command: &str) -> Result<Option<String>> {
        let cmd = command.trim();
        if tina_cli::is_help_command(cmd) {
            self.ui.notice(HELP)?;
            return Ok(None);
        }
        if let Some(c) = tina_cli::language_command(cmd) {
            match c {
                tina_cli::LanguageCommand::Show => self.ui.notice(&self.ui.language_text())?,
                tina_cli::LanguageCommand::Set(v) => {
                    self.ui.set_language(&v)?;
                    self.ui.notice(&self.ui.set_language_text())?;
                }
            }
            return Ok(None);
        }
        if let Some(c) = tina_cli::assistant_id_command(cmd) {
            match c {
                tina_cli::AssistantIdCommand::Show => {
                    self.ui.notice(&self.ui.assistant_id_text())?
                }
                tina_cli::AssistantIdCommand::Set(v) => {
                    self.ui.set_assistant_id(&v)?;
                    self.ui.notice(&self.ui.set_assistant_id_text())?;
                }
            }
            return Ok(None);
        }
        if let Some(c) = tina_cli::goal_command(cmd) {
            match c? {
                tina_cli::GoalCommand::Show => {}
                tina_cli::GoalCommand::Set(s) => self.mem.set_goal(Some(s))?,
                tina_cli::GoalCommand::Clear => self.mem.set_goal(None)?,
            }
            self.reset_usage();
            self.ui.notice(&self.ui.goal_text(self.mem.goal()))?;
            return Ok(None);
        }
        if cmd == "/thinking"
            || cmd.starts_with("/thinking ")
            || cmd == "/思考"
            || cmd.starts_with("/思考 ")
        {
            if let Some((_, value)) = cmd.split_once(' ') {
                let v = value.trim();
                if !["off", "low", "medium", "high", "xhigh", "max"].contains(&v) {
                    return Err("Thinking: off|low|medium|high|xhigh".into());
                }
                self.config.reasoning_effort = if v == "max" { "xhigh" } else { v }.into();
                self.reset_usage();
            }
            self.ui
                .notice(&format!("Thinking: {}", self.config.reasoning_effort))?;
            return Ok(None);
        }
        if let Some(c) = tina_cli::session_command(cmd) {
            match c {
                tina_cli::SessionCommand::New => {
                    self.mem = self.mem.fresh_conversation()?;
                    self.reset_usage();
                    self.ui.notice(&self.ui.new_session_detail_text(
                        self.mem.conversation_id(),
                        self.mem.current_epoch(),
                    ))?;
                }
                tina_cli::SessionCommand::Compact => {
                    let h = self.mem.compact_for_handoff_retaining(Vec::new(), 1)?;
                    self.reset_usage();
                    self.ui.notice(if h.is_some() {
                        "History archived; reduced active context is ready / 已归档并压缩活动上下文"
                    } else {
                        "No older complete turn to compact / 暂无可压缩的旧轮次"
                    })?;
                }
                tina_cli::SessionCommand::Retry => {
                    let s = self.mem.retry_last_user_turn()?;
                    self.reset_usage();
                    return Ok(Some(s));
                }
                tina_cli::SessionCommand::Sessions => self.ui.notice(
                    &self
                        .ui
                        .sessions_text(&self.mem.list_sessions()?, self.mem.conversation_id()),
                )?,
                tina_cli::SessionCommand::AgentRestart => {
                    self.host.restart()?;
                    self.ui.notice("Tool host restarted / 工具进程已重启")?;
                }
                tina_cli::SessionCommand::Tools => self.ui.notice(&self.ui.tools_text(
                    self.host.binary(),
                    &self.workspace,
                    false,
                    &self.host.read_only_capabilities(),
                    &self.host.mutating_capabilities(),
                ))?,
                tina_cli::SessionCommand::Environment => self.ui.notice_untrusted(
                    &host_environment::HostEnvironment::detect(None)?
                        .status_text(Some(&self.workspace)),
                )?,
                tina_cli::SessionCommand::Agent => self.ui.notice(&format!(
                    "Agent={} · read_only={} · approvals=per operation",
                    !self.chat_only, self.read_only
                ))?,
                tina_cli::SessionCommand::Context | tina_cli::SessionCommand::Status => {
                    let b = self.budget();
                    self.ui.notice_untrusted(&format!("Tina {} · {}\nModel: {}\nWorkspace: {}\nSession: {} · epoch {}\nContext: {} / {} ({}) · output reserve {} · handoff {}\nRemote KV/prefix cache: not measured by Tina\nLast turn: {}",env!("CARGO_PKG_VERSION"),self.config.base_url,self.config.model,self.workspace.display(),self.mem.conversation_id(),self.mem.current_epoch(),self.context_tokens,b.thresholds.hard_limit,self.context_source,b.output_reserve,b.thresholds.handoff_at,self.last_metrics))?;
                }
            }
            return Ok(None);
        }
        if let Some(c) = tina_cli::resume_command(cmd) {
            let selection = match c? {
                tina_cli::ResumeTarget::Latest => memory::ResumeSelection::Latest,
                tina_cli::ResumeTarget::Conversation(id) => {
                    memory::ResumeSelection::Conversation(id)
                }
            };
            let (m, resumed) = self.mem.reopen(selection)?;
            self.mem = m;
            self.reset_usage();
            self.ui.notice(&self.ui.conversation_opened_text(
                resumed,
                self.mem.conversation_id(),
                self.mem.current_epoch(),
                self.mem.turn_count(),
            ))?;
            return Ok(None);
        }
        if tina_cli::is_memory_command(cmd) {
            self.ui.notice(&self.ui.memory_status_text(
                self.mem.conversation_id(),
                self.mem.current_epoch(),
                self.mem.turn_count(),
                self.mem.active_message_count(),
                self.mem.unfinished_todo_count(),
                self.mem.archive_root(),
            ))?;
            return Ok(None);
        }
        if let Some(text) = tina_cli::remember_command(cmd) {
            let text = text.ok_or("Usage: /remember TEXT")?;
            let id = self.mem.pin(text.into())?;
            self.reset_usage();
            self.ui.notice(&self.ui.pinned_fact_text(&id))?;
            return Ok(None);
        }
        if let Some(c) = tina_cli::project_memory_command(cmd) {
            match c? {
                tina_cli::ProjectMemoryCommand::Status => {
                    let s = self.project.status();
                    self.ui.notice(&format!(
                        "Project memory: {} · {} entries",
                        s.state.as_str(),
                        s.entries.len()
                    ))?;
                }
                tina_cli::ProjectMemoryCommand::Remember(text) => {
                    let r = self.project.remember(&text)?;
                    self.ui.notice(&format!(
                        "Saved project fact {} · {}",
                        r.entry_id, r.commit_oid
                    ))?;
                }
                tina_cli::ProjectMemoryCommand::Forget(id) => {
                    if self.ui.confirm_delete(&format!("project fact {id}"))? {
                        self.project.forget(&id, "DELETE")?;
                        self.ui.notice("Project fact removed")?;
                    }
                }
            }
            self.reset_usage();
            return Ok(None);
        }
        if todo_cli::is_todo_command(cmd) {
            use todo_cli::TodoCommand as T;
            match todo_cli::parse_todo_command(cmd)? {
                T::Add(s) => {
                    self.mem.add_todo(s)?;
                }
                T::Start(id) => {
                    self.mem
                        .set_todo_status(&id, memory::TodoStatus::InProgress)?;
                }
                T::Done(id) => {
                    self.mem
                        .set_todo_status(&id, memory::TodoStatus::Completed)?;
                }
                T::Reopen(id) => {
                    self.mem.set_todo_status(&id, memory::TodoStatus::Pending)?;
                }
                T::Remove(id) => {
                    if self.ui.confirm_delete(&format!("todo {id}"))? {
                        self.mem.remove_todo(&id)?;
                    }
                }
                T::ClearCompleted => {
                    if self.ui.confirm_delete("completed todos")? {
                        self.mem.clear_completed_todos()?;
                    }
                }
                T::Help => self.ui.notice(todo_cli::TODO_USAGE)?,
                T::List => {}
            }
            self.ui.notice_untrusted(&todo_cli::render_todos_localized(
                &self.mem.todos(),
                self.ui.is_chinese(),
            ))?;
            self.reset_usage();
            return Ok(None);
        }
        if let Some(c) = tina_cli::export_command(cmd) {
            let c = c?;
            let path = memory::ActiveConversationMemory::resolve_export_destination(&c.path)?;
            let overwrite = path.exists() && self.ui.confirm_overwrite(&c.path)?;
            let f = match c.format {
                tina_cli::ExportFormat::Jsonl => memory::ConversationExportFormat::Jsonl,
                tina_cli::ExportFormat::Markdown => memory::ConversationExportFormat::Markdown,
            };
            self.mem.export_to(&path, f, overwrite)?;
            self.ui.notice(&format!("Exported {}", path.display()))?;
            return Ok(None);
        }
        if let Some(c) = tina_cli::workspace_command(cmd) {
            match c? {
                tina_cli::WorkspaceCommand::Diff => {
                    for (ordinal, args) in
                        [json!({}), json!({"cached":false}), json!({"cached":true})]
                            .into_iter()
                            .enumerate()
                    {
                        let name = if ordinal == 0 {
                            "git_status"
                        } else {
                            "git_diff"
                        };
                        let result = self.host.execute_read_only_cancellable(
                            &format!("ui-diff-{ordinal}"),
                            name,
                            args,
                            &mut || Ok(false),
                        )?;
                        self.ui.notice_untrusted(&format!(
                            "ok={}\n{}\n{}",
                            result.ok, result.stdout, result.stderr
                        ))?;
                    }
                }
                tina_cli::WorkspaceCommand::Review { focus } => {
                    return Ok(Some(self.ui.review_prompt(focus.as_deref())));
                }
            }
            return Ok(None);
        }
        self.ui.notice(&self.ui.unknown_command_text(cmd))?;
        Ok(None)
    }
}
pub(crate) fn run(args: &[String]) -> Result<()> {
    let config = live_config::load(args)?;
    let workspace = crate::option(args, "--workspace")
        .map(PathBuf::from)
        .or_else(|| config.workspace_dir.clone())
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()?;
    if !workspace.is_dir() {
        return Err("Workspace must be a directory".into());
    }
    let archive = crate::option(args, "--archive")
        .map(PathBuf::from)
        .unwrap_or_else(|| live_config::archive_root(&workspace, &config));
    let selection = if let Some(id) = crate::option(args, "--resume-conversation") {
        memory::ResumeSelection::Conversation(id)
    } else if args.iter().any(|a| a == "--resume") {
        memory::ResumeSelection::Latest
    } else {
        memory::ResumeSelection::New
    };
    let (mem, resumed) =
        memory::ActiveConversationMemory::open(&archive, selection, Default::default())?;
    let mut ui = TinaCli::with_config(tina_cli::UiConfig::resolve(Some(&config.language), None)?);
    if io::stdin().is_terminal()
        && io::stdout().is_terminal()
        && !args.iter().any(|a| a == "--no-splash")
    {
        crate::splash::show()?;
    }
    ui.start_live(
        &config.model,
        &config.base_url,
        config.context_positions,
        &config.reasoning_effort,
    )?;
    let host = host::ToolHostClient::spawn(
        &crate::tool_host(args)?,
        &workspace,
        false,
        host_environment::HostEnvironment::detect(None)?.shell(),
        None,
    )?;
    ui.notice(&ui.conversation_opened_text(
        resumed,
        mem.conversation_id(),
        mem.current_epoch(),
        mem.turn_count(),
    ))?;
    let model = ModelClient::new(config.clone())?;
    let max_steps = crate::option(args, "--max-steps")
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    if !(1..=128).contains(&max_steps) {
        return Err("--max-steps must be 1..128".into());
    }
    let mut s = Session {
        config,
        ui,
        model,
        mem,
        project: ProjectMemory::open(&workspace),
        host,
        workspace,
        context_tokens: 0,
        context_source: "estimated",
        observed_prompt: None,
        max_steps,
        chat_only: args.iter().any(|a| a == "--chat-only"),
        read_only: args.iter().any(|a| a == "--read-only"),
        last_metrics: Value::Null,
    };
    if let Some(text) = crate::option(args, "--request") {
        let ok = s.run_turn(text)?;
        s.host.shutdown()?;
        return if ok {
            Ok(())
        } else {
            Err("Turn did not complete; see the recorded host status".into())
        };
    }
    loop {
        let (input, text) = s.ui.read_message(
            s.context_tokens,
            s.budget().thresholds.hard_limit,
            !s.chat_only,
            &s.config.reasoning_effort,
        )?;
        match input {
            tina_cli::ChatInput::Exit => break,
            tina_cli::ChatInput::Submit => {
                if !text.trim().is_empty() {
                    s.ui.show_user(&text)?;
                    s.run_turn(text)?;
                }
            }
            tina_cli::ChatInput::Command(cmd) => {
                let review = matches!(
                    tina_cli::workspace_command(&cmd),
                    Some(Ok(tina_cli::WorkspaceCommand::Review { .. }))
                );
                match s.command(&cmd) {
                    Ok(Some(text)) => {
                        let previous = s.read_only;
                        if review {
                            s.read_only = true;
                        }
                        let result = s.run_turn(text);
                        s.read_only = previous;
                        result?;
                    }
                    Ok(None) => {}
                    Err(e) => s.ui.end_error(&e.to_string())?,
                }
            }
        }
    }
    s.host.shutdown()?;
    io::stdout().flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn server_usage_only_anchors_an_unchanged_prefix_and_tools() {
        let messages = vec![json!({"role":"user","content":"old"})];
        let observed = PromptObservation::capture(10, &messages, &[], 100).unwrap();
        let mut grown = messages.clone();
        grown.push(json!({"role":"assistant","content":"answer"}));
        assert_eq!(observed.estimate(&grown, &[], 130).unwrap(), Some(296));
        grown[0]["content"] = json!("new");
        assert_eq!(observed.estimate(&grown, &[], 130).unwrap(), None);
        assert_eq!(
            observed
                .estimate(&messages, &[json!({"name":"new-tool"})], 130)
                .unwrap(),
            None
        );
    }
    #[test]
    fn removed_messages_or_smaller_serialization_cannot_reuse_usage() {
        let messages = vec![json!({"role":"user","content":"question"})];
        let observed = PromptObservation::capture(10, &messages, &[], 100).unwrap();
        assert_eq!(observed.estimate(&[], &[], 110).unwrap(), None);
        assert_eq!(observed.estimate(&messages, &[], 99).unwrap(), None);
    }
}
