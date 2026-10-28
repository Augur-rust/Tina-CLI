use std::io::{self, BufReader};
use std::path::PathBuf;

use cq_tool_host::{run_jsonl, ToolHost, ToolHostScope, ToolHostShell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let mut workspace: Option<PathBuf> = None;
    let mut full_access = false;
    let mut shell: Option<ToolHostShell> = None;
    let mut web_search_url: Option<String> = None;
    while let Some(arg) = args.next() {
        if arg == "--workspace" {
            workspace = args.next().map(PathBuf::from);
        } else if arg == "--full-access" {
            full_access = true;
        } else if arg == "--shell" {
            let value = args.next().ok_or("--shell requires a value")?;
            shell = Some(ToolHostShell::parse(&value.to_string_lossy())?);
        } else if arg == "--web-search-url" {
            let value = args.next().ok_or("--web-search-url requires a value")?;
            web_search_url = Some(value.to_string_lossy().into_owned());
        } else if arg == "--help" || arg == "-h" {
            println!("cq-tool-host --workspace <path> [--full-access] [--shell <powershell|cmd|bash|zsh|fish|sh>] [--web-search-url <searxng-url>]\nReads versioned JSONL requests from stdin and writes responses to stdout. --full-access enables machine-wide filesystem, shell, and network access and must only be passed by a warning/confirmation gate. --web-search-url enables only the bounded read-only web_search capability.");
            return Ok(());
        } else {
            return Err(format!("unknown argument: {}", arg.to_string_lossy()).into());
        }
    }
    let workspace = workspace.ok_or("--workspace is required")?;
    let scope = if full_access {
        ToolHostScope::Machine
    } else {
        ToolHostScope::Workspace
    };
    let host = match shell {
        Some(shell) => ToolHost::with_scope_shell_and_web_search_url(
            workspace,
            scope,
            shell,
            web_search_url.as_deref(),
        )?,
        None => {
            ToolHost::with_scope_and_web_search_url(workspace, scope, web_search_url.as_deref())?
        }
    };
    run_jsonl(
        &host,
        BufReader::new(io::stdin().lock()),
        io::stdout().lock(),
    )?;
    Ok(())
}
