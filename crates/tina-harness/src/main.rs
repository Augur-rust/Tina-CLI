//! Standalone CPU validation interface; live model/GPU integration is separate.
#![allow(dead_code)]
mod context_budget;
mod conversation_memory;
mod host_environment;
mod plugin;
mod project_memory;
mod replay;
#[cfg(test)]
mod replay_tests;
mod splash;
mod tina_cli;
mod todo_cli;
mod tool_host_client;
mod tool_schemas;
mod tool_transaction;
use serde_json::json;
use std::{
    io::Read,
    path::{Path, PathBuf},
};

type Error = Box<dyn std::error::Error>;
fn help() {
    println!("Tina CLI CPU verification package 0.1.0-alpha.1\n\nThis binary uses scripted completions, not an LLM. Tool processes and archives are real.\n\nCommands:\n  tina context-check --input-tokens N [--output-tokens N] [--context-positions N]\n  tina doctor --workspace DIR\n  tina prompt --workspace DIR --request TEXT [--format dsml]\n  tina replay --workspace DIR --scenario FILE --archive DIR [--resume] [--tool-host FILE]\n\nReplay never auto-approves writes, deletion, or command execution.\nThe original interactive/GPU integration is included as source under integration/.");
}
fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}
fn required(args: &[String], name: &str) -> Result<String, Error> {
    option(args, name).ok_or_else(|| format!("missing {name}").into())
}
fn tool_host(args: &[String]) -> Result<PathBuf, Error> {
    if let Some(p) = option(args, "--tool-host") {
        return Ok(p.into());
    }
    let p = std::env::current_exe()?
        .parent()
        .ok_or("binary has no parent")?
        .join(if cfg!(windows) {
            "cq-tool-host.exe"
        } else {
            "cq-tool-host"
        });
    Ok(p)
}
fn read_scenario(path: &Path) -> Result<replay::Scenario, Error> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(2 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 2 * 1024 * 1024 {
        return Err("scenario exceeds 2 MiB".into());
    }
    tina_protocol::validate_strict_json(&bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
fn execute() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("--help");
    if ["--help", "-h", "help"].contains(&command) {
        help();
        return Ok(());
    }
    let mut index = 1;
    let mut seen = std::collections::BTreeSet::new();
    while index < args.len() {
        let name = &args[index];
        if !seen.insert(name.clone()) {
            return Err(format!("duplicate option {name}").into());
        }
        if name == "--resume" && command == "replay" {
            index += 1;
            continue;
        }
        let valid = match command {
            "context-check" => ["--input-tokens", "--output-tokens", "--context-positions"]
                .contains(&name.as_str()),
            "doctor" => ["--workspace", "--tool-host"].contains(&name.as_str()),
            "prompt" => ["--workspace", "--request", "--format"].contains(&name.as_str()),
            "replay" => {
                ["--workspace", "--scenario", "--archive", "--tool-host"].contains(&name.as_str())
            }
            _ => false,
        };
        if !valid || index + 1 >= args.len() {
            return Err(format!("invalid or incomplete option {name}").into());
        }
        index += 2;
    }
    if command == "context-check" {
        let input: usize = required(&args, "--input-tokens")?.parse()?;
        let output: usize = option(&args, "--output-tokens")
            .unwrap_or_else(|| "4096".into())
            .parse()?;
        let positions: usize = option(&args, "--context-positions")
            .unwrap_or_else(|| "262144".into())
            .parse()?;
        let budget = context_budget::ContextBudget::for_owner(positions, output);
        let admission = budget.admit_input(input);
        let result = json!({"mode":"CPU_CONTEXT_CHECK", "model_inference":false,
            "token_count_source":"caller", "admitted":admission.is_ok(),
            "owner_positions":positions, "input_tokens":input,
            "requested_output_tokens":output, "reserved_output_tokens":budget.output_reserve,
            "usable_input_limit":budget.thresholds.hard_limit,
            "handoff_at":budget.thresholds.handoff_at,
            "pressure":format!("{:?}", budget.pressure(input)),
            "rejection":admission.as_ref().err().map(|e| json!({"schema":e.schema,
                "kind":e.kind.as_str(), "required_positions":e.required_positions,
                "overflow_positions":e.overflow_positions}))});
        println!("{}", serde_json::to_string_pretty(&result)?);
        admission?;
        return Ok(());
    }
    if let Some(format) = option(&args, "--format") {
        if !["json", "dsml"].contains(&format.as_str()) {
            return Err("--format must be json or dsml".into());
        }
    }
    let workspace = PathBuf::from(required(&args, "--workspace")?).canonicalize()?;
    if !workspace.is_dir() {
        return Err("workspace must be a directory".into());
    }
    match command {
        "doctor" => {
            let environment = host_environment::HostEnvironment::detect(None)?;
            let host = tool_host_client::ToolHostClient::spawn(
                &tool_host(&args)?,
                &workspace,
                false,
                environment.shell(),
                None,
            )?;
            let mut result = json!({"mode":"CPU_VERIFICATION","model_inference":false,
                "tool_host_handshake":"PASS","workspace":workspace,"environment":environment.status_text(Some(&workspace)),
                "tool_schema_count":tool_schemas::agent_tool_schemas(false,false).len(),
                "context_contract":replay::show_context_contract(),
                "mcp":false,"subagents":false,"semantic_provider":"NOT_CONFIGURED"});
            result["archive_and_tool_regressions"] = "run scripts/Verify.ps1".into();
            host.shutdown()?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "prompt" => {
            // Prompt inspection must not create a persistent archive in the user's workspace.
            let temp = std::env::temp_dir().join(format!(
                "tina-prompt-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            ));
            let result = (|| -> Result<(), Error> {
                let (mut mem, _) = conversation_memory::ActiveConversationMemory::open(
                    &temp,
                    conversation_memory::ResumeSelection::New,
                    Default::default(),
                )?;
                mem.append_user(required(&args, "--request")?)?;
                let project = project_memory::ProjectMemory::open(&workspace);
                let messages = replay::prompt_messages(&mut mem, &project, &workspace, false)?;
                let bytes = serde_json::to_vec(&messages)?;
                if option(&args, "--format").as_deref() == Some("dsml") {
                    println!(
                        "{}",
                        tina_protocol::encode_dsv4_messages_v2(
                            &bytes,
                            tina_protocol::Dsv4EncodingOptionsV2::chat()
                        )?
                    );
                } else {
                    println!("{}", serde_json::to_string_pretty(&messages)?);
                }
                Ok(())
            })();
            if temp.is_dir() {
                std::fs::remove_dir_all(&temp)?;
            }
            result?;
        }
        "replay" => {
            let scenario = read_scenario(Path::new(&required(&args, "--scenario")?))?;
            let archive = PathBuf::from(required(&args, "--archive")?);
            // Only the trusted host can grant an approval. Scenario JSON cannot.
            let receipt = replay::run(
                &scenario,
                &workspace,
                &archive,
                &tool_host(&args)?,
                args.iter().any(|a| a == "--resume"),
                Default::default(),
                &mut |_, _| false,
                &mut || Ok(false),
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        _ => return Err(format!("unknown command {command:?}; use --help").into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = execute() {
        eprintln!("Tina: {e}");
        std::process::exit(1);
    }
}
