//! Local endpoint configuration; credentials are referenced, never serialized.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct LiveConfig {
    pub base_url: String,
    pub model: String,
    pub workspace_dir: Option<PathBuf>,
    pub api_key_env: String,
    pub api_key_file: Option<PathBuf>,
    pub context_positions: usize,
    pub max_output_tokens: usize,
    pub timeout_seconds: u64,
    pub language: String,
    pub reasoning_effort: String,
}
impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:7581/v1".into(),
            model: "quasar-27b".into(),
            workspace_dir: None,
            api_key_env: "TINA_API_KEY".into(),
            api_key_file: None,
            context_positions: 262144,
            max_output_tokens: 4096,
            timeout_seconds: 1800,
            language: "auto".into(),
            reasoning_effort: "high".into(),
        }
    }
}
impl LiveConfig {
    pub fn validate(&self) -> Result<(), String> {
        let url = reqwest::Url::parse(&self.base_url).map_err(|_| "Invalid endpoint URL")?;
        if !["http", "https"].contains(&url.scheme())
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(
                "Use an HTTP(S) base URL without credentials, query parameters or fragments".into(),
            );
        }
        if self.model.trim().is_empty() || self.model.chars().any(char::is_control) {
            return Err("Model ID is required".into());
        }
        if !(1024..=262_144).contains(&self.context_positions)
            || self.max_output_tokens == 0
            || self.max_output_tokens >= self.context_positions
        {
            return Err(
                "Output budget must be positive and smaller than the context window (1024..262144)"
                    .into(),
            );
        }
        if self.timeout_seconds == 0 || self.timeout_seconds > 86400 {
            return Err("Timeout must be 1..86400 seconds".into());
        }
        if !["off", "low", "medium", "high", "xhigh"].contains(&self.reasoning_effort.as_str()) {
            return Err("Thinking must be off, low, medium, high or xhigh".into());
        }
        crate::tina_cli::UiLanguage::parse(&self.language)?;
        Ok(())
    }
    pub fn credential(&self) -> Result<String, String> {
        if !self.api_key_env.is_empty() {
            if let Ok(value) = std::env::var(&self.api_key_env) {
                if !value.trim().is_empty() {
                    return Ok(value.trim().into());
                }
            }
        }
        if let Some(path) = &self.api_key_file {
            let value = std::fs::read_to_string(path)
                .map_err(|_| "Cannot read the configured API key file")?;
            if value.len() > 16384 {
                return Err("API key file is too large".into());
            }
            return Ok(value.trim().into());
        }
        Ok(String::new())
    }
    pub fn endpoint(&self, suffix: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), suffix)
    }
}
pub(crate) fn data_root() -> PathBuf {
    if let Some(root) = std::env::var_os("TINA_HOME") {
        return root.into();
    }
    if let Some(root) = std::env::var_os("LOCALAPPDATA") {
        return PathBuf::from(root).join("TinaCLI");
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join(".local/share/tina-cli")
}
pub(crate) fn config_path(args: &[String]) -> PathBuf {
    crate::option(args, "--config")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("TINA_CONFIG").map(PathBuf::from))
        .unwrap_or_else(|| data_root().join("config.json"))
}
pub(crate) fn load(args: &[String]) -> Result<LiveConfig, String> {
    let path = config_path(args);
    let mut c = if path.is_file() {
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        if bytes.len() > 65536 {
            return Err("Configuration file exceeds 64 KiB".into());
        }
        serde_json::from_slice::<LiveConfig>(&bytes)
            .map_err(|e| format!("Invalid configuration: {e}"))?
    } else {
        LiveConfig::default()
    };
    for (key, env_name) in [("--base-url", "TINA_BASE_URL"), ("--model", "TINA_MODEL")] {
        if let Some(v) = crate::option(args, key).or_else(|| std::env::var(env_name).ok()) {
            if key == "--base-url" {
                c.base_url = v;
            } else {
                c.model = v;
            }
        }
    }
    if let Some(v) = crate::option(args, "--workspace") {
        c.workspace_dir = Some(v.into());
    }
    if let Some(v) = crate::option(args, "--api-key-file") {
        c.api_key_file = Some(v.into());
    }
    if let Some(v) = crate::option(args, "--api-key-env") {
        c.api_key_env = v;
    }
    if let Some(v) = crate::option(args, "--language") {
        c.language = v;
    }
    if let Some(v) = crate::option(args, "--thinking") {
        c.reasoning_effort = v;
    }
    for key in ["--context-positions", "--max-tokens", "--timeout-seconds"] {
        if let Some(v) = crate::option(args, key) {
            let n = v
                .parse::<usize>()
                .map_err(|_| format!("Invalid integer for {key}"))?;
            match key {
                "--context-positions" => c.context_positions = n,
                "--max-tokens" => c.max_output_tokens = n,
                _ => c.timeout_seconds = n as u64,
            }
        }
    }
    c.validate()?;
    Ok(c)
}
fn ask(label: &str, default: &str) -> Result<String, String> {
    print!("{label} [{default}]: ");
    io::stdout().flush().map_err(|e| e.to_string())?;
    let mut s = String::new();
    io::stdin().read_line(&mut s).map_err(|e| e.to_string())?;
    Ok(if s.trim().is_empty() {
        default.into()
    } else {
        s.trim().into()
    })
}
pub(crate) fn configure(args: &[String]) -> Result<(), String> {
    let mut c = load(args)?;
    if args.iter().any(|a| a == "--augur") {
        let local = std::env::var_os("LOCALAPPDATA").ok_or("LOCALAPPDATA is unavailable")?;
        let root = PathBuf::from(local).join("Augur");
        let d: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("config.json"))
                .map_err(|_| "Augur configuration was not found")?,
        )
        .map_err(|_| "Invalid Augur configuration")?;
        let port = d["port"].as_u64().ok_or("Augur port is missing")?;
        if port == 0 || port > 65535 {
            return Err("Invalid Augur port".into());
        }
        c.base_url = format!("http://127.0.0.1:{port}/v1");
        c.context_positions = d["context_tokens"].as_u64().unwrap_or(262144) as usize;
        c.api_key_file = Some(root.join("api-key.txt"));
    } else if args.len() <= 1
        || args.iter().all(|s| {
            s == "configure" || s == "--config" || s == &config_path(args).to_string_lossy()
        })
    {
        if !io::stdin().is_terminal() {
            return Err(
                "Run configure interactively, or supply --base-url URL --model ID (or --augur)"
                    .into(),
            );
        }
        println!("Tina setup / 首次配置 — connect to your running model service.");
        c.base_url = ask("API base URL (include /v1)", &c.base_url)?;
        c.model = ask("Model ID", &c.model)?;
        let workspace = ask(
            "Workspace folder",
            &std::env::current_dir()
                .map_err(|e| e.to_string())?
                .to_string_lossy(),
        )?;
        c.workspace_dir = Some(
            PathBuf::from(workspace)
                .canonicalize()
                .map_err(|_| "Workspace folder does not exist")?,
        );
        let key = ask("API key FILE path; leave blank to use TINA_API_KEY", "")?;
        c.api_key_file = if key.is_empty() {
            None
        } else {
            Some(key.into())
        };
        c.context_positions = ask("Context positions", &c.context_positions.to_string())?
            .parse()
            .map_err(|_| "Invalid context capacity")?;
        c.max_output_tokens = ask("Maximum output tokens", &c.max_output_tokens.to_string())?
            .parse()
            .map_err(|_| "Invalid output budget")?;
        c.language = ask("Language (zh/en/auto)", &c.language)?;
    }
    c.validate()?;
    let path = config_path(args);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    // No API key value is included in this data.
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&c).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "Saved configuration: {}\nStart: tina chat --workspace <your-project>",
        path.display()
    );
    Ok(())
}
pub(crate) fn archive_root(workspace: &Path, c: &LiveConfig) -> PathBuf {
    let mut hash = Sha256::new();
    hash.update(workspace.to_string_lossy().as_bytes());
    hash.update(b"\0");
    hash.update(c.base_url.as_bytes());
    hash.update(b"\0");
    hash.update(c.model.as_bytes());
    data_root()
        .join("conversations")
        .join(format!("{:x}", hash.finalize()))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_credentials_in_url_and_exhausted_budget() {
        let mut c = LiveConfig {
            base_url: "https://secret@example.com/v1".into(),
            ..Default::default()
        };
        assert!(c.validate().is_err());
        c = LiveConfig::default();
        c.max_output_tokens = c.context_positions;
        assert!(c.validate().is_err());
    }
    #[test]
    fn archives_are_scoped_by_workspace_endpoint_and_model() {
        let c = LiveConfig::default();
        let a = archive_root(Path::new("one"), &c);
        assert_ne!(a, archive_root(Path::new("two"), &c));
        let mut other = c.clone();
        other.model = "other".into();
        assert_ne!(a, archive_root(Path::new("one"), &other));
    }
}
