//! Bounded Tina CLI plugin discovery and command routing.
//!
//! The first plugin surface is deliberately declarative. A plugin can register
//! commands, themes, and lifecycle/tool capability declarations in `plugin.json`.
//! Nothing is executed while discovering a plugin, and an optional `entry` is
//! never launched implicitly. This keeps startup deterministic and gives us a
//! stable protocol boundary for future out-of-process plugins.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

const PLUGIN_API_VERSION: u64 = 1;
const MAX_MANIFEST_BYTES: u64 = 64 * 1024;
const MAX_PLUGINS: usize = 64;
const MAX_COMMANDS_PER_PLUGIN: usize = 32;
const MAX_TEXT_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PluginCommand {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) prompt_prefix: Option<String>,
    pub(crate) prompt_suffix: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PluginManifest {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) api_version: u64,
    pub(crate) description: String,
    pub(crate) capabilities: Vec<String>,
    pub(crate) commands: Vec<PluginCommand>,
    pub(crate) root: PathBuf,
    pub(crate) entry: Option<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PluginCommandOutcome {
    Unhandled,
    Notice(String),
    Prompt(String),
}

#[derive(Debug)]
pub(crate) struct PluginRegistry {
    roots: Vec<PathBuf>,
    plugins: Vec<PluginManifest>,
    diagnostics: Vec<String>,
}

impl PluginRegistry {
    pub(crate) fn discover() -> Self {
        let roots = configured_roots();
        let mut registry = Self {
            roots,
            plugins: Vec::new(),
            diagnostics: Vec::new(),
        };
        registry.reload();
        registry
    }

    #[cfg(test)]
    fn from_roots(roots: Vec<PathBuf>) -> Self {
        let mut registry = Self {
            roots,
            plugins: Vec::new(),
            diagnostics: Vec::new(),
        };
        registry.reload();
        registry
    }

    pub(crate) fn reload(&mut self) {
        self.plugins.clear();
        self.diagnostics.clear();
        let mut manifest_paths = Vec::new();
        for root in &self.roots {
            collect_manifest_paths(root, &mut manifest_paths, &mut self.diagnostics);
        }
        manifest_paths.sort();
        manifest_paths.dedup();

        let mut seen_ids = BTreeSet::new();
        for path in manifest_paths.into_iter().take(MAX_PLUGINS) {
            match load_manifest(&path) {
                Ok(manifest) => {
                    if !seen_ids.insert(manifest.id.clone()) {
                        self.diagnostics.push(format!(
                            "plugin {} ignored: duplicate id at {}",
                            manifest.id,
                            path.display()
                        ));
                        continue;
                    }
                    self.plugins.push(manifest);
                }
                Err(error) => self
                    .diagnostics
                    .push(format!("plugin {} ignored: {error}", path.display())),
            }
        }
        self.plugins.sort_by(|left, right| left.id.cmp(&right.id));
    }

    pub(crate) fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    pub(crate) fn list_text(&self) -> String {
        self.list_text_localized(false)
    }

    pub(crate) fn list_text_localized(&self, chinese: bool) -> String {
        if self.plugins.is_empty() {
            return if chinese {
                "未加载 Tina 插件。请在 .tina/plugins 下添加 plugin.json，或设置 TINA_PLUGIN_DIRS。"
                    .to_string()
            } else {
                "No Tina plugins loaded. Add a plugin.json under .tina/plugins or set TINA_PLUGIN_DIRS."
                    .to_string()
            };
        }
        let mut output = if chinese {
            String::from("已加载 Tina 插件：\n")
        } else {
            String::from("Loaded Tina plugins:\n")
        };
        for plugin in &self.plugins {
            output.push_str("  ");
            output.push_str(&plugin.id);
            output.push(' ');
            output.push_str(&plugin.version);
            output.push_str(" - ");
            output.push_str(&plugin.description);
            if !plugin.commands.is_empty() {
                output.push_str(" [commands: ");
                for (index, command) in plugin.commands.iter().enumerate() {
                    if index != 0 {
                        output.push_str(", ");
                    }
                    output.push_str(&command.name);
                }
                output.push(']');
            }
            if plugin.entry.is_some() {
                output.push_str(if chinese {
                    " [外部入口仅限显式启用]"
                } else {
                    " [external entry explicit-only]"
                });
            }
            output.push('\n');
        }
        output
    }

    /// Route only explicit `/plugins` and `/plugin ...` commands. Unknown slash
    /// commands remain visible to the caller so it can report them without
    /// accidentally sending a control command to the model.
    #[cfg(test)]
    pub(crate) fn dispatch(&mut self, line: &str) -> PluginCommandOutcome {
        self.dispatch_localized(line, false)
    }

    pub(crate) fn dispatch_localized(&mut self, line: &str, chinese: bool) -> PluginCommandOutcome {
        let trimmed = line.trim();
        if matches!(trimmed, "/plugins" | "/插件") {
            return PluginCommandOutcome::Notice(self.list_text_localized(chinese));
        }
        if trimmed == "/help" {
            return PluginCommandOutcome::Notice(if chinese {
                "指令：/help、/plugins、/plugin reload、/plugin <id> [command] [text]、/exit"
                    .to_string()
            } else {
                "Commands: /help, /plugins, /plugin reload, /plugin <id> [command] [text], /exit"
                    .to_string()
            });
        }
        if trimmed == "/plugin reload" {
            self.reload();
            return PluginCommandOutcome::Notice(if chinese {
                format!("插件注册表已重载：{} 个插件。", self.plugins.len())
            } else {
                format!(
                    "Plugin registry reloaded: {} plugin(s).",
                    self.plugins.len()
                )
            });
        }
        let Some(rest) = trimmed.strip_prefix("/plugin ") else {
            return PluginCommandOutcome::Unhandled;
        };
        let mut fields = rest.split_whitespace();
        let Some(plugin_id) = fields.next() else {
            return PluginCommandOutcome::Notice(if chinese {
                "用法：/plugin <id> [command] [text]（或 /plugins）。".to_string()
            } else {
                "Usage: /plugin <id> [command] [text] (or /plugins).".to_string()
            });
        };
        let Some(plugin) = self.plugins.iter().find(|plugin| plugin.id == plugin_id) else {
            return PluginCommandOutcome::Notice(if chinese {
                format!("未知插件 {plugin_id:?}。")
            } else {
                format!("Unknown plugin {plugin_id:?}.")
            });
        };
        if plugin.capabilities.iter().any(|capability| {
            matches!(
                capability.as_str(),
                "filesystem" | "delete" | "write" | "overwrite" | "rename" | "evict" | "exec"
            )
        }) {
            return PluginCommandOutcome::Notice(if chinese {
                format!(
                    "插件 {plugin_id:?} 声明了变更或执行能力；在安装显式 host 合同前，Tina 保持禁用。"
                )
            } else {
                format!(
                    "Plugin {plugin_id:?} declares a mutating/executable capability; the Tina host keeps it disabled until an explicit host contract is installed."
                )
            });
        }

        let remaining = fields.collect::<Vec<_>>();
        let (command_name, text) = if plugin.commands.len() == 1 {
            let default_command = plugin.commands[0].name.as_str();
            if remaining.first().copied() == Some(default_command) {
                (default_command, remaining[1..].join(" "))
            } else {
                (default_command, remaining.join(" "))
            }
        } else if let Some(command) = remaining.first().copied() {
            (command, remaining[1..].join(" "))
        } else {
            let commands = plugin
                .commands
                .iter()
                .map(|command| command.name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return PluginCommandOutcome::Notice(if chinese {
                format!("插件 {plugin_id:?} 的指令：{commands}")
            } else {
                format!("Plugin {plugin_id:?} commands: {commands}")
            });
        };
        let Some(command) = plugin
            .commands
            .iter()
            .find(|command| command.name == command_name)
        else {
            return PluginCommandOutcome::Notice(if chinese {
                format!("插件 {plugin_id:?} 没有指令 {command_name:?}。")
            } else {
                format!("Unknown command {command_name:?} for plugin {plugin_id:?}.")
            });
        };
        if text.len() > MAX_TEXT_BYTES {
            return PluginCommandOutcome::Notice(if chinese {
                format!("插件输入超过 {MAX_TEXT_BYTES} 字节上限。")
            } else {
                format!("Plugin input exceeds the {MAX_TEXT_BYTES}-byte limit.")
            });
        }
        let mut prompt = String::new();
        if let Some(prefix) = &command.prompt_prefix {
            prompt.push_str(prefix);
        }
        if !text.is_empty() {
            if !prompt.is_empty() && !prompt.chars().last().is_some_and(char::is_whitespace) {
                prompt.push(' ');
            }
            prompt.push_str(&text);
        }
        if let Some(suffix) = &command.prompt_suffix {
            prompt.push_str(suffix);
        }
        if prompt.trim().is_empty() {
            return PluginCommandOutcome::Notice(if chinese {
                format!("用法：/plugin {plugin_id} {command_name} <文本>")
            } else {
                format!("Usage: /plugin {plugin_id} {command_name} <text>")
            });
        }
        PluginCommandOutcome::Prompt(prompt)
    }
}

fn configured_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(raw) = env::var_os("TINA_PLUGIN_DIRS") {
        roots.extend(env::split_paths(&raw));
    }
    if let Ok(current) = env::current_dir() {
        roots.push(current.join(".tina").join("plugins"));
    }
    if let Some(home) = env::var_os("TINA_HOME") {
        roots.push(PathBuf::from(home).join("plugins"));
    }
    roots.sort();
    roots.dedup();
    roots
}

fn collect_manifest_paths(root: &Path, output: &mut Vec<PathBuf>, diagnostics: &mut Vec<String>) {
    if root.is_file() {
        if root.file_name().and_then(|name| name.to_str()) == Some("plugin.json") {
            output.push(root.to_path_buf());
        }
        return;
    }
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => {
            diagnostics.push(format!("cannot scan {}: {error}", root.display()));
            return;
        }
    };
    let mut children = entries.filter_map(Result::ok).collect::<Vec<_>>();
    children.sort_by_key(|entry| entry.path());
    for child in children {
        let path = child.path();
        if path.is_dir() {
            let manifest = path.join("plugin.json");
            if manifest.is_file() {
                output.push(manifest);
            }
        } else if path.file_name().and_then(|name| name.to_str()) == Some("plugin.json") {
            output.push(path);
        }
    }
}

fn load_manifest(path: &Path) -> Result<PluginManifest, String> {
    let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(format!("manifest exceeds {MAX_MANIFEST_BYTES} bytes"));
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    parse_manifest(path, &value)
}

fn parse_manifest(path: &Path, value: &Value) -> Result<PluginManifest, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "manifest root must be an object".to_string())?;
    let id = required_string(object.get("id"), "id")?.to_string();
    validate_id(&id)?;
    let api_version = object
        .get("api_version")
        .and_then(Value::as_u64)
        .unwrap_or(PLUGIN_API_VERSION);
    if api_version != PLUGIN_API_VERSION {
        return Err(format!(
            "unsupported api_version {api_version}; supported version is {PLUGIN_API_VERSION}"
        ));
    }
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(&id)
        .to_string();
    let version = object
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("0.1.0")
        .to_string();
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("Tina plugin")
        .to_string();
    let capabilities = string_array(object.get("capabilities"), "capabilities")?;
    let commands = parse_commands(object.get("commands"))?;
    let entry = object
        .get("entry")
        .and_then(Value::as_str)
        .map(|entry| validate_entry(path, entry))
        .transpose()?;
    Ok(PluginManifest {
        id,
        name,
        version,
        api_version,
        description,
        capabilities,
        commands,
        root: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
        entry,
    })
}

fn parse_commands(value: Option<&Value>) -> Result<Vec<PluginCommand>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let array = value
        .as_array()
        .ok_or_else(|| "commands must be an array".to_string())?;
    if array.len() > MAX_COMMANDS_PER_PLUGIN {
        return Err(format!(
            "commands exceed the {MAX_COMMANDS_PER_PLUGIN}-command limit"
        ));
    }
    let mut names = BTreeSet::new();
    let mut commands = Vec::with_capacity(array.len());
    for value in array {
        let object = value
            .as_object()
            .ok_or_else(|| "each command must be an object".to_string())?;
        let name = required_string(object.get("name"), "command.name")?.to_string();
        validate_command_name(&name)?;
        if !names.insert(name.clone()) {
            return Err(format!("duplicate command {name:?}"));
        }
        let description = object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("plugin command")
            .to_string();
        let prompt_prefix = bounded_text(object.get("prompt_prefix"), "prompt_prefix")?;
        let prompt_suffix = bounded_text(object.get("prompt_suffix"), "prompt_suffix")?;
        commands.push(PluginCommand {
            name,
            description,
            prompt_prefix,
            prompt_suffix,
        });
    }
    Ok(commands)
}

fn required_string<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a str, String> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{field} must be a non-empty string"))
}

fn string_array(value: Option<&Value>, field: &str) -> Result<Vec<String>, String> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    value
        .as_array()
        .ok_or_else(|| format!("{field} must be an array"))?
        .iter()
        .map(|value| required_string(Some(value), field).map(str::to_string))
        .collect()
}

fn bounded_text(value: Option<&Value>, field: &str) -> Result<Option<String>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = required_string(Some(value), field)?;
    if text.len() > MAX_TEXT_BYTES {
        return Err(format!("{field} exceeds the {MAX_TEXT_BYTES}-byte limit"));
    }
    Ok(Some(text.to_string()))
}

fn validate_id(value: &str) -> Result<(), String> {
    if value.len() > 64
        || value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
        || !value.as_bytes()[0].is_ascii_lowercase() && !value.as_bytes()[0].is_ascii_digit()
    {
        return Err(format!(
            "invalid plugin id {value:?}; use 1-64 lowercase ASCII letters, digits, '-' or '_'"
        ));
    }
    Ok(())
}

fn validate_command_name(value: &str) -> Result<(), String> {
    if value.len() > 64
        || value.is_empty()
        || value.starts_with('/')
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
        })
    {
        return Err(format!(
            "invalid command name {value:?}; use lowercase ASCII without a leading '/'"
        ));
    }
    Ok(())
}

fn validate_entry(manifest_path: &Path, entry: &str) -> Result<PathBuf, String> {
    let path = Path::new(entry);
    if path.is_absolute()
        || path
            .components()
            .any(|component| component == std::path::Component::ParentDir)
    {
        return Err("entry must be a relative path inside the plugin directory".to_string());
    }
    let root = manifest_path.parent().unwrap_or(Path::new("."));
    Ok(root.join(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("tina-plugin-test-{stamp}"));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn write_plugin(root: &Path, id: &str, commands: &str) -> PathBuf {
        let directory = root.join(id);
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("plugin.json");
        fs::write(
            &path,
            format!(
                r#"{{"api_version":1,"id":"{id}","name":"{id}","version":"1.0.0","description":"test","commands":{commands}}}"#
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn discovers_and_dispatches_declarative_prompt_command() {
        let root = temp_root();
        write_plugin(
            &root,
            "summarizer",
            r#"[{"name":"summarize","prompt_prefix":"Summarize this:"}]"#,
        );
        let mut registry = PluginRegistry::from_roots(vec![root.clone()]);
        assert_eq!(registry.plugins.len(), 1);
        assert_eq!(
            registry.dispatch("/plugin summarizer summarize this text"),
            PluginCommandOutcome::Prompt("Summarize this: this text".to_string())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn mutating_capabilities_are_fail_closed() {
        let root = temp_root();
        write_plugin(&root, "unsafe-tool", r#"[]"#);
        let path = root.join("unsafe-tool").join("plugin.json");
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["capabilities"] = serde_json::json!(["filesystem", "delete"]);
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let mut registry = PluginRegistry::from_roots(vec![root.clone()]);
        let outcome = registry.dispatch("/plugin unsafe-tool");
        assert!(matches!(outcome, PluginCommandOutcome::Notice(text) if text.contains("disabled")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn multi_command_plugin_requires_and_routes_command_name() {
        let root = temp_root();
        write_plugin(
            &root,
            "tools",
            r#"[{"name":"ask","prompt_prefix":"Ask:"},{"name":"rewrite","prompt_prefix":"Rewrite:"}]"#,
        );
        let mut registry = PluginRegistry::from_roots(vec![root.clone()]);
        assert!(matches!(
            registry.dispatch("/plugin tools"),
            PluginCommandOutcome::Notice(text) if text.contains("ask") && text.contains("rewrite")
        ));
        assert_eq!(
            registry.dispatch("/plugin   tools   rewrite   this"),
            PluginCommandOutcome::Prompt("Rewrite: this".to_string())
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn malformed_or_duplicate_plugins_are_ignored_deterministically() {
        let root = temp_root();
        write_plugin(&root, "same", r#"[]"#);
        let other = root.join("other");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("plugin.json"), b"not json").unwrap();
        let registry = PluginRegistry::from_roots(vec![root.clone()]);
        assert_eq!(registry.plugins.len(), 1);
        assert_eq!(registry.diagnostics.len(), 1);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn chinese_dispatch_localizes_host_notices_without_translating_ids() {
        let root = temp_root();
        let mut registry = PluginRegistry::from_roots(vec![root.clone()]);
        let empty = registry.dispatch_localized("/plugins", true);
        assert!(
            matches!(empty, PluginCommandOutcome::Notice(text) if text.contains("未加载 Tina 插件"))
        );
        let unknown = registry.dispatch_localized("/plugin missing", true);
        assert!(
            matches!(unknown, PluginCommandOutcome::Notice(text) if text.contains("未知插件") && text.contains("missing"))
        );
        let _ = fs::remove_dir_all(root);
    }
}
