//! Inline terminal interaction for the Tina Engine product CLI.
//!
//! The module owns only terminal input and presentation. Model/session
//! ownership remains in main.rs, so JSONL and interactive chat share one
//! authoritative request path.

use std::cell::Cell;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::time::Duration;

use crossterm::cursor::{MoveDown, MoveToColumn, MoveUp};
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::queue;
use crossterm::style::Print;
use crossterm::terminal::{self, Clear, ClearType};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::context_budget;

const MAX_EDITOR_ROWS: usize = 4;
const MAX_COMMAND_POPUP_ROWS: usize = 8;
const MIN_BOX_WIDTH: usize = 24;
const MAX_BOX_WIDTH: usize = 112;
// Keep individual tool rows compact, while giving the completed tool batch
// enough vertical air to remain distinct from the next assistant/composer.
const UI_BLOCK_GAP_LINES: usize = 2;
const UI_TOOL_ROW_GAP_LINES: usize = 1;
const UI_TOOL_BATCH_GAP_LINES: usize = 3;

const RESET: &str = "\x1b[0m";
const ACCENT: &str = "\x1b[38;2;139;178;231m";
const MOON: &str = "\x1b[38;2;205;220;242m";
const MOON_BRIGHT: &str = "\x1b[1;38;2;245;249;255m";
const MUTED: &str = "\x1b[38;2;120;137;164m";
const TIDE: &str = "\x1b[38;2;80;159;214m";
const DIFF_GREEN: &str = "\x1b[1;38;2;0;220;72m";
const DIFF_RED: &str = "\x1b[1;38;2;248;45;45m";
const POPUP_SELECTED: &str = "\x1b[7;1;38;2;245;249;255m";
const DEFAULT_ASSISTANT_ID: &str = "Tina";
const MAX_ASSISTANT_ID_CHARS: usize = 32;
const SLASH_COMPLETIONS: &[&str] = &[
    "/assistant",
    "/agent-restart",
    "/agent",
    "/compact",
    "/context",
    "/diff",
    "/environment",
    "/exit",
    "/export",
    "/goal",
    "/help",
    "/language",
    "/memory",
    "/new",
    "/plugins",
    "/project-forget",
    "/project-memory",
    "/project-remember",
    "/remember",
    "/review",
    "/resume",
    "/retry",
    "/search",
    "/sessions",
    "/status",
    "/thinking",
    "/tools",
    "/todo",
    "/上下文",
    "/会话",
    "/助手",
    "/代理",
    "/差异",
    "/导出",
    "/压缩",
    "/帮助",
    "/插件",
    "/项目忘记",
    "/项目记忆",
    "/项目记住",
    "/新会话",
    "/环境",
    "/状态",
    "/待办",
    "/记住",
    "/记忆",
    "/网络搜索",
    "/审查",
    "/恢复",
    "/语言",
    "/目标",
    "/思考",
    "/重试",
    "/重启工具",
    "/工具",
    "/退出",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlashCommandSpec {
    name: &'static str,
    chinese_name: &'static str,
    description: &'static str,
    chinese_description: &'static str,
}

impl SlashCommandSpec {
    fn display_name(self, prefix: &str) -> &'static str {
        if !prefix.is_ascii() && !self.chinese_name.is_empty() {
            self.chinese_name
        } else {
            self.name
        }
    }

    fn description(self, language: UiLanguage) -> &'static str {
        if language.is_chinese() {
            self.chinese_description
        } else {
            self.description
        }
    }
}

// Keep aliases out of the default list so every action appears once. Chinese
// aliases become the displayed command only when the user types a Chinese
// prefix; the English `/` palette stays compact and stable.
const SLASH_COMMAND_SPECS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "/agent",
        chinese_name: "/代理",
        description: "show Agent status and access mode",
        chinese_description: "查看 Agent 状态与权限模式",
    },
    SlashCommandSpec {
        name: "/goal",
        chinese_name: "/目标",
        description: "show, set, or clear the session goal",
        chinese_description: "查看、设置或清除会话目标",
    },
    SlashCommandSpec {
        name: "/thinking",
        chinese_name: "/思考",
        description: "choose off, low, medium, high, or xhigh reasoning",
        chinese_description: "选择 off、low、medium、high 或 xhigh 思考级别",
    },
    SlashCommandSpec {
        name: "/status",
        chinese_name: "/状态",
        description: "show runtime, context, cache, and Agent status",
        chinese_description: "查看运行、上下文、缓存和 Agent 状态",
    },
    SlashCommandSpec {
        name: "/context",
        chinese_name: "/上下文",
        description: "show the configured context budget",
        chinese_description: "查看当前配置的上下文预算",
    },
    SlashCommandSpec {
        name: "/help",
        chinese_name: "/帮助",
        description: "show all commands and argument forms",
        chinese_description: "查看全部指令与参数格式",
    },
    SlashCommandSpec {
        name: "/tools",
        chinese_name: "/工具",
        description: "show enabled tools and safety boundaries",
        chinese_description: "查看已启用工具与安全边界",
    },
    SlashCommandSpec {
        name: "/environment",
        chinese_name: "/环境",
        description: "show the OS, shell, terminal, and cwd",
        chinese_description: "查看 OS、shell、终端与工作目录",
    },
    SlashCommandSpec {
        name: "/diff",
        chinese_name: "/差异",
        description: "show staged and unstaged workspace changes",
        chinese_description: "查看工作树已暂存与未暂存改动",
    },
    SlashCommandSpec {
        name: "/review",
        chinese_name: "/审查",
        description: "run a read-only review of current changes",
        chinese_description: "只读审查当前改动",
    },
    SlashCommandSpec {
        name: "/search",
        chinese_name: "/网络搜索",
        description: "show network-search provider status (never fabricates results)",
        chinese_description: "查看网络搜索提供方状态（不会伪造结果）",
    },
    SlashCommandSpec {
        name: "/todo",
        chinese_name: "/待办",
        description: "show or update the session TodoList",
        chinese_description: "查看或更新会话 TodoList",
    },
    SlashCommandSpec {
        name: "/sessions",
        chinese_name: "/会话",
        description: "list resumable conversation archives",
        chinese_description: "列出可恢复的会话归档",
    },
    SlashCommandSpec {
        name: "/resume",
        chinese_name: "/恢复",
        description: "switch to latest or a full archive ID",
        chinese_description: "切换到最新归档或完整归档 ID",
    },
    SlashCommandSpec {
        name: "/retry",
        chinese_name: "/重试",
        description: "retry the last user turn on a new branch",
        chinese_description: "在新分支重试上一用户回合",
    },
    SlashCommandSpec {
        name: "/new",
        chinese_name: "/新会话",
        description: "start a fresh session and retain the archive",
        chinese_description: "开始新会话并保留旧归档",
    },
    SlashCommandSpec {
        name: "/compact",
        chinese_name: "/压缩",
        description: "create a manual context handoff",
        chinese_description: "主动创建上下文交接",
    },
    SlashCommandSpec {
        name: "/export",
        chinese_name: "/导出",
        description: "export the active branch as JSONL or Markdown",
        chinese_description: "将活动分支导出为 JSONL 或 Markdown",
    },
    SlashCommandSpec {
        name: "/memory",
        chinese_name: "/记忆",
        description: "show pinned conversation memory",
        chinese_description: "查看已固定的会话记忆",
    },
    SlashCommandSpec {
        name: "/remember",
        chinese_name: "/记住",
        description: "pin a fact in this conversation",
        chinese_description: "在当前会话固定一条事实",
    },
    SlashCommandSpec {
        name: "/project-memory",
        chinese_name: "/项目记忆",
        description: "show Git-backed project memory",
        chinese_description: "查看 Git 持久化项目记忆",
    },
    SlashCommandSpec {
        name: "/project-remember",
        chinese_name: "/项目记住",
        description: "persist a project fact across sessions",
        chinese_description: "跨会话保存项目事实",
    },
    SlashCommandSpec {
        name: "/project-forget",
        chinese_name: "/项目忘记",
        description: "forget a project fact through the delete gate",
        chinese_description: "经删除拦截忘记项目事实",
    },
    SlashCommandSpec {
        name: "/plugins",
        chinese_name: "/插件",
        description: "list declarative plugins and readiness",
        chinese_description: "列出声明式插件及就绪状态",
    },
    SlashCommandSpec {
        name: "/assistant",
        chinese_name: "/助手",
        description: "show or change the assistant ID",
        chinese_description: "查看或修改助手 ID",
    },
    SlashCommandSpec {
        name: "/language",
        chinese_name: "/语言",
        description: "show or change the terminal language",
        chinese_description: "查看或切换终端界面语言",
    },
    SlashCommandSpec {
        name: "/agent-restart",
        chinese_name: "/重启工具",
        description: "restart the bounded Agent tool host",
        chinese_description: "重启受控 Agent tool-host",
    },
    SlashCommandSpec {
        name: "/exit",
        chinese_name: "/退出",
        description: "close Tina and retain the conversation archive",
        chinese_description: "关闭 Tina 并保留会话归档",
    },
];

/// Presentation language for the terminal UI. This never changes the model
/// prompt or the persisted `user`/`assistant` role names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UiLanguage {
    English,
    Chinese,
}

impl UiLanguage {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "en" | "en-us" | "en-gb" | "english" => Ok(Self::English),
            "zh" | "zh-cn" | "zh-hans" | "中文" | "简体中文" => Ok(Self::Chinese),
            "auto" => Ok(Self::from_locale()),
            other => Err(format!(
                "invalid UI language {other:?}; expected en, zh, or auto"
            )),
        }
    }

    fn from_locale() -> Self {
        for key in ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"] {
            if let Some(value) = std::env::var_os(key).and_then(|value| value.into_string().ok()) {
                let value = value.to_ascii_lowercase();
                if value.starts_with("zh") || value.contains("chinese") {
                    return Self::Chinese;
                }
                if value.starts_with("en") || value.contains("english") {
                    return Self::English;
                }
            }
        }
        // English is the deterministic fallback on Windows and in minimal
        // containers where locale variables are absent.
        Self::English
    }

    pub(crate) const fn is_chinese(self) -> bool {
        matches!(self, Self::Chinese)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct UiConfig {
    pub(crate) language: UiLanguage,
    pub(crate) assistant_id: String,
}

impl UiConfig {
    pub(crate) fn resolve(
        language_arg: Option<&str>,
        assistant_id_arg: Option<&str>,
    ) -> Result<Self, String> {
        let language_env = std::env::var("TINA_LANG")
            .ok()
            .or_else(|| std::env::var("TINA_LANGUAGE").ok());
        let language_value = language_arg.or(language_env.as_deref()).unwrap_or("auto");
        let language = UiLanguage::parse(language_value)?;
        let assistant_id_env = std::env::var("TINA_ASSISTANT_ID").ok();
        let assistant_id = assistant_id_arg
            .or(assistant_id_env.as_deref())
            .unwrap_or(DEFAULT_ASSISTANT_ID);
        validate_assistant_id(assistant_id).map(|assistant_id| Self {
            language,
            assistant_id,
        })
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            language: UiLanguage::English,
            assistant_id: DEFAULT_ASSISTANT_ID.to_string(),
        }
    }
}

fn validate_assistant_id(value: &str) -> Result<String, String> {
    let value = value.trim();
    let count = value.chars().count();
    if value.is_empty() {
        return Err("assistant ID cannot be empty".to_string());
    }
    if count > MAX_ASSISTANT_ID_CHARS {
        return Err(format!(
            "assistant ID must be at most {MAX_ASSISTANT_ID_CHARS} characters"
        ));
    }
    if value.chars().any(|ch| ch.is_control()) {
        return Err("assistant ID cannot contain control characters".to_string());
    }
    Ok(value.to_string())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AssistantIdCommand {
    Show,
    Set(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum LanguageCommand {
    Show,
    Set(String),
}

/// Product-facing reasoning levels. `off` is deliberately short because it
/// is the fast, low-latency spelling used in scripts and the interactive UI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ThinkingLevel {
    Off,
    Medium,
    High,
    Max,
}

impl ThinkingLevel {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "non-thinking" | "nonthinking" | "chat" | "none" => Ok(Self::Off),
            "medium" | "med" | "low" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "max" | "xhigh" => Ok(Self::Max),
            other => Err(format!(
                "invalid thinking level {other:?}; expected off, medium, high, or max"
            )),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GoalCommand {
    Show,
    Set(String),
    Clear,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ThinkingCommand {
    Show,
    Set(ThinkingLevel),
}

/// Session-local controls are parsed by the terminal layer but executed by
/// `main.rs`, which owns the model session and archive state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SessionCommand {
    Agent,
    Context,
    Status,
    Tools,
    AgentRestart,
    Environment,
    Sessions,
    New,
    Compact,
    Retry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProjectMemoryCommand {
    Status,
    Remember(String),
    Forget(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ResumeTarget {
    Latest,
    Conversation(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WorkspaceCommand {
    Diff,
    Review { focus: Option<String> },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LastTurnStatus<'a> {
    pub(crate) generated_tokens: usize,
    pub(crate) first_token_ready_seconds: f64,
    pub(crate) complete_wall_seconds: f64,
    pub(crate) owner_event: &'a str,
    pub(crate) cache_event: &'a str,
    pub(crate) cache_restore_seconds: f64,
    pub(crate) cache_export_seconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeStatus<'a> {
    pub(crate) conversation_id: &'a str,
    pub(crate) epoch: u64,
    pub(crate) archived_messages: usize,
    pub(crate) active_messages: usize,
    pub(crate) context_positions: usize,
    pub(crate) budget: context_budget::ContextBudget,
    pub(crate) owner_active: bool,
    pub(crate) agent_enabled: bool,
    pub(crate) agent_full_access: bool,
    pub(crate) workspace: Option<&'a Path>,
    pub(crate) tool_host_ready: bool,
    pub(crate) cache_mode: &'a str,
    pub(crate) thinking_level: &'a str,
    pub(crate) cache_directory: Option<&'a Path>,
    pub(crate) last_turn: Option<LastTurnStatus<'a>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExportFormat {
    Jsonl,
    Markdown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ExportCommand {
    pub(crate) format: ExportFormat,
    pub(crate) path: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ApplyPatchConfirmation<'a> {
    pub(crate) target: &'a str,
    pub(crate) expected_sha256: &'a str,
    pub(crate) edit_count: usize,
    pub(crate) removed_bytes: usize,
    pub(crate) added_bytes: usize,
    pub(crate) preview: &'a str,
    pub(crate) preview_truncated: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WriteFileConfirmation<'a> {
    pub(crate) target: &'a str,
    pub(crate) create: bool,
    pub(crate) expected_sha256: Option<&'a str>,
    pub(crate) content_bytes: usize,
    pub(crate) content_sha256: &'a str,
    pub(crate) preview: &'a str,
    pub(crate) preview_truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ChatInput {
    Submit,
    Command(String),
    Exit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RenderedComposer {
    height: usize,
    cursor_row: usize,
}

#[derive(Clone, Copy)]
struct ComposerContext<'a> {
    context_positions: usize,
    capacity: usize,
    agent_enabled: bool,
    thinking: &'a str,
    color: bool,
    language: UiLanguage,
    assistant_id: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VisualLine {
    start: usize,
    end: usize,
}

struct EditorBuffer {
    text: String,
    cursor: usize,
    preferred_column: Option<usize>,
}

impl EditorBuffer {
    fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            preferred_column: None,
        }
    }

    fn insert(&mut self, value: &str) -> bool {
        let normalized = value.replace("\r\n", "\n").replace('\r', "\n");
        if normalized.is_empty()
            || self.text.len().saturating_add(normalized.len())
                > context_budget::MAX_INTERACTIVE_MESSAGE_BYTES
        {
            return false;
        }
        self.text.insert_str(self.cursor, &normalized);
        self.cursor += normalized.len();
        self.preferred_column = None;
        true
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.preferred_column = None;
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let start = self.text[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(offset, _)| offset)
            .unwrap_or(0);
        self.text.drain(start..self.cursor);
        self.cursor = start;
        self.preferred_column = None;
    }

    fn delete(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        let end = self.text[self.cursor..]
            .char_indices()
            .nth(1)
            .map(|(offset, _)| self.cursor + offset)
            .unwrap_or(self.text.len());
        self.text.drain(self.cursor..end);
        self.preferred_column = None;
    }

    fn move_home(&mut self) {
        self.cursor = self.text[..self.cursor]
            .rfind('\n')
            .map(|offset| offset + 1)
            .unwrap_or(0);
        self.preferred_column = None;
    }

    fn move_end(&mut self) {
        self.cursor = self.text[self.cursor..]
            .find('\n')
            .map(|offset| self.cursor + offset)
            .unwrap_or(self.text.len());
        self.preferred_column = None;
    }

    fn delete_to_home(&mut self) {
        let home = self.text[..self.cursor]
            .rfind('\n')
            .map(|offset| offset + 1)
            .unwrap_or(0);
        self.text.drain(home..self.cursor);
        self.cursor = home;
        self.preferred_column = None;
    }

    fn delete_to_end(&mut self) {
        let end = self.text[self.cursor..]
            .find('\n')
            .map(|offset| self.cursor + offset)
            .unwrap_or(self.text.len());
        self.text.drain(self.cursor..end);
        self.preferred_column = None;
    }

    fn delete_word_backwards(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let before = &self.text[..self.cursor];
        let mut boundary = before.len();
        for (offset, ch) in before.char_indices().rev() {
            if !ch.is_whitespace() {
                boundary = offset;
                break;
            }
        }
        let before_word = &before[..boundary];
        let word_start = before_word
            .char_indices()
            .rev()
            .find(|(_, ch)| ch.is_whitespace())
            .map(|(offset, ch)| offset + ch.len_utf8())
            .unwrap_or(0);
        self.text.drain(word_start..self.cursor);
        self.cursor = word_start;
        self.preferred_column = None;
    }

    fn move_horizontal(&mut self, right: bool) {
        if right {
            if let Some((offset, _)) = self.text[self.cursor..].char_indices().nth(1) {
                self.cursor += offset;
            } else if self.cursor < self.text.len() {
                self.cursor = self.text.len();
            }
        } else if self.cursor > 0 {
            self.cursor = self.text[..self.cursor]
                .char_indices()
                .next_back()
                .map(|(offset, _)| offset)
                .unwrap_or(0);
        }
        self.preferred_column = None;
    }

    fn move_vertical(&mut self, width: usize, delta: isize) {
        let lines = visual_lines(&self.text, width);
        let current = line_for_cursor(&lines, self.cursor);
        let current_line = lines[current];
        let current_column = self
            .preferred_column
            .unwrap_or_else(|| display_width(&self.text[current_line.start..self.cursor]));
        let target = if delta.is_negative() {
            current.saturating_sub(delta.unsigned_abs())
        } else {
            current.saturating_add(delta as usize).min(lines.len() - 1)
        };
        let target_line = lines[target];
        self.cursor = target_line.start
            + byte_at_display_column(
                &self.text[target_line.start..target_line.end],
                current_column,
            );
        self.preferred_column = Some(current_column);
    }
}

pub(crate) struct TinaCli {
    interactive: bool,
    color: bool,
    language: UiLanguage,
    assistant_id: String,
    history: Vec<String>,
    history_cursor: Option<usize>,
    assistant_frame_open: Cell<bool>,
    assistant_line_open: Cell<bool>,
    assistant_body_width: Cell<usize>,
    assistant_column: Cell<usize>,
}

impl TinaCli {
    pub(crate) fn with_config(config: UiConfig) -> Self {
        let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
        Self {
            interactive,
            color: interactive && std::env::var_os("NO_COLOR").is_none(),
            language: config.language,
            assistant_id: config.assistant_id,
            history: Vec::new(),
            history_cursor: None,
            assistant_frame_open: Cell::new(false),
            assistant_line_open: Cell::new(false),
            assistant_body_width: Cell::new(0),
            assistant_column: Cell::new(0),
        }
    }

    pub(crate) fn assistant_id(&self) -> &str {
        &self.assistant_id
    }

    pub(crate) fn is_chinese(&self) -> bool {
        self.language.is_chinese()
    }

    pub(crate) fn set_assistant_id(&mut self, value: &str) -> Result<(), String> {
        self.assistant_id = validate_assistant_id(value)?;
        Ok(())
    }

    pub(crate) fn set_language(&mut self, value: &str) -> Result<(), String> {
        self.language = UiLanguage::parse(value)?;
        Ok(())
    }

    pub(crate) fn help_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "指令：/agent Agent 状态 · /goal [set|clear] 会话目标 · /thinking [off|medium|high|max] 思考级别 · /search <查询> 网络搜索 · /help 帮助 · /status 运行状态 · /tools Agent 能力 · /agent-restart 重启 Agent host · /context 上下文 · /diff 工作树差异 · /review [关注点] 只读审查 · /environment 终端环境 · /sessions 会话归档 · /resume <latest|ID> 恢复会话 · /retry 重试分支 · /new 新会话 · /compact 手动压缩 · /export 导出 · /assistant [id] 设置助手名 · /language [en|zh|auto] 切换语言 · /memory 会话记忆 · /remember <事实> · /project-memory Git 项目记忆 · /project-remember <事实> · /project-forget <ID> · /todo、/待办 help · /plugins、/插件 · /exit 退出 · TINA_NO_MOTION=1 关闭潮汐动画"
        } else {
            "Commands: /agent Agent status · /goal [set|clear] session goal · /thinking [off|medium|high|max] reasoning level · /search <query> web search · /help · /status runtime status · /tools Agent capabilities · /agent-restart restart the Agent host · /context budget · /diff working tree · /review [focus] read-only review · /environment host details · /sessions archives · /resume <latest|ID> switch archive · /retry retry branch · /new new session · /compact handoff · /export transcript · /assistant [id] set assistant ID · /language [en|zh|auto] switch UI · /memory session memory · /remember <fact> · /project-memory Git project memory · /project-remember <fact> · /project-forget <ID> · /todo help · /plugins · /exit · TINA_NO_MOTION=1 disables the tide animation"
        }
    }

    pub(crate) fn cli_help_text(language: UiLanguage) -> String {
        let help = if language.is_chinese() {
            "Tina Engine CLI\n\n用法：\n  cq-infer --chat --artifact <目录> --placement-plan <文件> [选项]\n  cq-infer --prompt <文本> --artifact <目录> --placement-plan <文件> [选项]\n  cq-infer --messages-json <文件> --artifact <目录> --placement-plan <文件> [选项]\n  cq-infer --cache-smoke\n\n常用选项：\n  --max-new-tokens N       输出上限（聊天默认 64；非聊天默认 1）\n  --temperature F --top-k N --top-p F --seed N\n  --thinking-mode MODE     chat|thinking（默认 chat/off）\n  --reasoning-effort LEVEL low|high|max（仅 thinking）\n  --language en|zh|auto    终端界面语言\n  --assistant-id ID        初始助手 ID（默认 Tina）\n  --shell SHELL            聊天命令语法：powershell|cmd|bash|zsh|fish|sh\n  --agent --workspace DIR  启用受控 Agent 预览（必须与 --chat 一起使用）\n  --help                   显示帮助；未知选项会直接拒绝\n\n交互指令：\n  /help、/帮助                         显示指令\n  /context、/status、/上下文            查看 256K KV/上下文状态\n  /environment、/env、/环境             查看注入模型的 OS、shell、终端和 cwd\n  /sessions、/会话                     查看可恢复归档\n  /retry、/重试                        创建不可变重试分支\n  /new、/reset、/clear                 开始新会话（保留旧归档）\n  /compact                             主动创建上下文交接\n  /export [jsonl|md] <路径>             导出当前活动分支（覆盖需确认）\n  /assistant [ID]、/name [ID]           查看或修改助手 ID（默认 Tina）\n  /language [en|zh|auto]、/lang        查看或切换界面语言\n  /memory、/记忆                       查看会话记忆\n  /remember、/记住 <事实>              固定会话事实\n  /project-memory、/项目记忆            查看 Git 项目记忆\n  /project-remember、/项目记住 <事实>   跨会话保存项目事实\n  /project-forget、/项目忘记 <ID>       忘记项目事实（需 DELETE）\n  /todo help                           查看待办指令\n  /plugins、/插件                      查看插件\n  /exit、/quit、:q                     退出\n\n生成期间按 Ctrl-C 会取消当前 owner 并安全切换新会话；不会发布半个回答。GPU prefix cache 命中率须以独立资格证据为准；未测量时 CLI 显示 NOT_MEASURED。环境变量：TINA_LANG、TINA_ASSISTANT_ID、TINA_SHELL、TINA_NO_MOTION。界面设置不改变模型的 user/assistant 角色协议。"
        } else {
            "Tina Engine CLI\n\nUsage:\n  cq-infer --chat --artifact <dir> --placement-plan <file> [options]\n  cq-infer --prompt <text> --artifact <dir> --placement-plan <file> [options]\n  cq-infer --messages-json <file> --artifact <dir> --placement-plan <file> [options]\n  cq-infer --cache-smoke\n\nCommon options:\n  --max-new-tokens N       Output limit (chat default: 64; non-chat: 1)\n  --temperature F --top-k N --top-p F --seed N\n  --thinking-mode MODE     chat|thinking (default: chat/off)\n  --reasoning-effort LEVEL low|high|max (thinking only)\n  --language en|zh|auto    Terminal UI language\n  --assistant-id ID        Initial assistant ID (default: Tina)\n  --shell SHELL            Chat command dialect: powershell|cmd|bash|zsh|fish|sh\n  --agent --workspace DIR  Enable the bounded Agent preview (requires --chat)\n  --help                   Show this help; unknown options are rejected\n\nInteractive commands:\n  /help, /commands                    Show commands\n  /context, /status                   Show context/KV owner status\n  /environment, /env                  Show the OS, shell, terminal and cwd injected into the model\n  /sessions                           List resumable archives\n  /retry                              Create an immutable retry branch\n  /new, /reset, /clear                Start a fresh session (archive retained)\n  /compact                            Create a manual context handoff\n  /export [jsonl|md] <path>            Export active branch (overwrite confirms)\n  /assistant [ID], /name [ID]         Show or change assistant ID (default Tina)\n  /language en|zh|auto, /lang         Switch UI language\n  /memory, /记忆                      Show conversation memory\n  /remember <fact>, /记住             Pin a session fact\n  /project-memory                     Show Git-backed project memory\n  /project-remember <fact>             Persist a project fact across sessions\n  /project-forget <ID>                 Forget a project fact (requires DELETE)\n  /todo help                          Show TodoList commands\n  /plugins、/插件                     List plugins\n  /exit、/quit、:q                    Exit\n\nPress Ctrl-C during generation to cancel the owner and safely start a new session; no partial answer is published. GPU prefix-cache hit rate is shown as NOT_MEASURED until independent qualification evidence exists. Environment: TINA_LANG, TINA_ASSISTANT_ID, TINA_SHELL, TINA_NO_MOTION. UI settings do not change the model's user/assistant role protocol."
        };
        let doctor_usage = if language.is_chinese() {
            "  cq-infer --doctor --artifact <目录> --placement-plan <文件>\n  cq-infer --ccswitch-server --artifact <目录> --placement-plan <文件> [--listen 127.0.0.1:15723]\n  cq-infer --prompt <文本> --artifact <目录> --placement-plan <文件> --kv-cache-mode read-write --kv-cache-dir <目录> --kv-cache-namespace <作用域>\n  cq-infer --cache-smoke"
        } else {
            "  cq-infer --doctor --artifact <dir> --placement-plan <file>\n  cq-infer --ccswitch-server --artifact <dir> --placement-plan <file> [--listen 127.0.0.1:15723]\n  cq-infer --prompt <text> --artifact <dir> --placement-plan <file> --kv-cache-mode read-write --kv-cache-dir <dir> --kv-cache-namespace <scope>\n  cq-infer --cache-smoke"
        };
        let (legacy_context_help, product_command_help) = if language.is_chinese() {
            (
                "  /context、/status、/上下文            查看 256K KV/上下文状态",
                "  /status、/状态                       查看会话、Agent、KV 和最近一轮运行状态\n  /tools、/工具                        查看已启用工具和安全边界\n  /search、/网络搜索 [查询]             查看 provider 状态或执行受控网络搜索\n  /agent-restart、/重启工具            重启受控 Agent tool-host（保留会话归档）\n  /context、/上下文                    查看 256K KV/上下文容量\n  /diff、/差异                         查看 Git 状态、暂存及未暂存差异（需 Agent 模式）\n  /review、/审查 [关注点]              只读审查当前工作树（需 Agent 模式）\n  /resume <latest|完整 ID>             在当前进程切换归档（先关闭旧 owner）",
            )
        } else {
            (
                "  /context, /status                   Show context/KV owner status",
                "  /status                             Show session, Agent, KV and last-turn runtime status\n  /tools                              Show enabled tools and safety boundaries\n  /search [query]                     Show provider status or run bounded web search\n  /agent-restart                      Restart the bounded Agent tool host (conversation archive retained)\n  /context                            Show the 256K KV/context budget\n  /diff                               Show Git status plus staged and unstaged diffs (Agent mode)\n  /review [focus]                     Review current changes with read-only tools (Agent mode)\n  /resume <latest|full ID>             Switch archives in this process (old owner closes first)",
            )
        };
        let doctor_option = if language.is_chinese() {
            "  --doctor                 纯 CPU 启动诊断；不创建 CUDA context\n  --ccswitch-server        显式启动本地 CCSwitch 兼容服务；默认不开启\n  --listen ADDR            默认 127.0.0.1:15723，仅允许回环地址\n  --api-key KEY            本地 Bearer token；也可用 TINA_API_KEY\n  --model ID               CCSwitch/API 中显示的模型 ID\n  --kv-cache-mode MODE     off|read|read-write；默认 off，仅支持精确整段 prompt 快照\n  --kv-cache-dir DIR       持久化 KV8/DSA 快照目录；需用 toc-sm70-v1-kv8 构建\n  --kv-cache-namespace NAME 必填的项目/用户隔离域；漂移时按 miss 处理\n  --kv-cache-max-bytes N   可选缓存字节上限（默认 16 GiB）\n  --kv-cache-max-entries N 可选完整快照条目上限（默认 16）\n  --max-new-tokens"
        } else {
            "  --doctor                 CPU-only startup diagnostics; no CUDA context\n  --ccswitch-server        Explicit local CCSwitch compatibility service; off by default\n  --listen ADDR            Defaults to 127.0.0.1:15723; loopback only\n  --api-key KEY            Local bearer token; TINA_API_KEY is also accepted\n  --model ID               Model ID advertised to CCSwitch/API clients\n  --kv-cache-mode MODE     off|read|read-write; default off, exact full-prompt snapshots only\n  --kv-cache-dir DIR       Persisted KV8/DSA snapshot root; build toc-sm70-v1-kv8\n  --kv-cache-namespace NAME Required project/user isolation scope; drift is a miss\n  --kv-cache-max-bytes N   Optional byte quota (default: 16 GiB)\n  --kv-cache-max-entries N Optional complete-entry quota (default: 16)\n  --max-new-tokens"
        };
        let version_option = if language.is_chinese() {
            "  --web-search-url URL     显式 SearXNG JSON provider；仅开放受控只读搜索\n  --agent-full-access      警告并确认后开放整机文件、命令与网络（需 --agent --chat）\n  --version、-V            显示版本并退出\n  --help"
        } else {
            "  --web-search-url URL     Explicit SearXNG JSON provider; enables bounded read-only search only\n  --agent-full-access      Warn and confirm before machine-wide files, shell and network access (requires --agent --chat)\n  --version, -V            Show version and exit\n  --help"
        };
        let (interactive_heading, interactive_commands) = if language.is_chinese() {
            (
                "交互指令：\n",
                "交互指令：\n  /agent、/代理                       查看 Agent 状态和启动方式\n  /goal [set <目标>|clear]、/目标       查看、设置或清除会话目标\n  /thinking [off|medium|high|max]       查看或切换思考级别（默认 off）\n",
            )
        } else {
            (
                "Interactive commands:\n",
                "Interactive commands:\n  /agent                              Show Agent status and startup command\n  /goal [set <text>|clear]            Show, set, or clear the session goal\n  /thinking [off|medium|high|max]     Show or change reasoning (default: off)\n",
            )
        };
        let help = if language.is_chinese() {
            help.to_string()
        } else {
            // Keep English palette punctuation stable while retaining Chinese
            // aliases in the localized help surface.
            help.replace("/plugins、/插件", "/plugins · /插件")
                .replace("/exit、/quit、:q", "/exit, /quit, :q")
        };
        help.replacen(legacy_context_help, product_command_help, 1)
            .replacen(interactive_heading, interactive_commands, 1)
            .replacen("  cq-infer --cache-smoke", doctor_usage, 1)
            .replacen("  --max-new-tokens", doctor_option, 1)
            .replacen("  --help", version_option, 1)
    }

    pub(crate) fn context_status_text(
        &self,
        context_positions: usize,
        budget: context_budget::ContextBudget,
        session_active: bool,
    ) -> String {
        let pressure = match budget.pressure(context_positions) {
            context_budget::ContextPressure::Normal => {
                if self.language.is_chinese() {
                    "正常"
                } else {
                    "normal"
                }
            }
            context_budget::ContextPressure::Prepare => {
                if self.language.is_chinese() {
                    "准备交接"
                } else {
                    "prepare"
                }
            }
            context_budget::ContextPressure::Freeze => {
                if self.language.is_chinese() {
                    "冻结交接"
                } else {
                    "freeze"
                }
            }
            context_budget::ContextPressure::Handoff => {
                if self.language.is_chinese() {
                    "需要交接"
                } else {
                    "handoff"
                }
            }
            context_budget::ContextPressure::HardLimit => {
                if self.language.is_chinese() {
                    "达到上限"
                } else {
                    "hard_limit"
                }
            }
        };
        let owner = if session_active {
            if self.language.is_chinese() {
                "活动"
            } else {
                "active"
            }
        } else if self.language.is_chinese() {
            "未创建"
        } else {
            "not_started"
        };
        if self.language.is_chinese() {
            format!(
                "上下文状态\n  owner: {owner} · 位置: {}/{}\n  可用输入上限: {} · 95% 交接阈值: {}\n  输出预留: {} · 压力: {pressure}\n  GPU prefix cache: NOT_MEASURED（CPU cache-smoke 仅为协议探针）",
                compact_count(context_positions),
                compact_count(budget.owner_positions),
                compact_count(budget.thresholds.hard_limit),
                compact_count(budget.thresholds.handoff_at),
                compact_count(budget.output_reserve),
            )
        } else {
            format!(
                "Context status\n  owner: {owner} · positions: {}/{}\n  usable input limit: {} · 95% handoff: {}\n  output reserve: {} · pressure: {pressure}\n  GPU prefix cache: NOT_MEASURED (CPU cache-smoke is protocol-only)",
                compact_count(context_positions),
                compact_count(budget.owner_positions),
                compact_count(budget.thresholds.hard_limit),
                compact_count(budget.thresholds.handoff_at),
                compact_count(budget.output_reserve),
            )
        }
    }

    pub(crate) fn runtime_status_text(&self, status: RuntimeStatus<'_>) -> String {
        let owner = if status.owner_active {
            "active"
        } else {
            "not_started"
        };
        let pressure = match status.budget.pressure(status.context_positions) {
            context_budget::ContextPressure::Normal => "normal",
            context_budget::ContextPressure::Prepare => "prepare",
            context_budget::ContextPressure::Freeze => "freeze",
            context_budget::ContextPressure::Handoff => "handoff",
            context_budget::ContextPressure::HardLimit => "hard_limit",
        };
        let workspace = status
            .workspace
            .map_or_else(|| "none".to_string(), |path| path.display().to_string());
        let cache_directory = status
            .cache_directory
            .map_or_else(|| "none".to_string(), |path| path.display().to_string());
        let (agent_mode, tool_host) = if status.agent_enabled {
            (
                if status.agent_full_access {
                    "full-machine-shell-network"
                } else {
                    "workspace-bounded"
                },
                if status.tool_host_ready {
                    "ready"
                } else {
                    "unavailable"
                },
            )
        } else {
            ("chat-only", "disabled")
        };
        let last_turn = status.last_turn.map_or_else(
            || {
                if self.language.is_chinese() {
                    "  最近一轮: NOT_RUN".to_string()
                } else {
                    "  last turn: NOT_RUN".to_string()
                }
            },
            |turn| {
                let turn_rate = turn.generated_tokens as f64
                    / turn.complete_wall_seconds.max(f64::EPSILON);
                if self.language.is_chinese() {
                    format!(
                        "  最近一轮: {} tokens · 首 token {:.1} ms · 完整墙钟 {:.1} ms · 本轮 {:.2} tok/s\n  owner 事件: {} · cache 事件: {} · restore {:.1} ms · export {:.1} ms",
                        turn.generated_tokens,
                        turn.first_token_ready_seconds * 1_000.0,
                        turn.complete_wall_seconds * 1_000.0,
                        turn_rate,
                        turn.owner_event,
                        turn.cache_event,
                        turn.cache_restore_seconds * 1_000.0,
                        turn.cache_export_seconds * 1_000.0,
                    )
                } else {
                    format!(
                        "  last turn: {} tokens · first token {:.1} ms · complete wall {:.1} ms · {:.2} tok/s turn\n  owner event: {} · cache event: {} · restore {:.1} ms · export {:.1} ms",
                        turn.generated_tokens,
                        turn.first_token_ready_seconds * 1_000.0,
                        turn.complete_wall_seconds * 1_000.0,
                        turn_rate,
                        turn.owner_event,
                        turn.cache_event,
                        turn.cache_restore_seconds * 1_000.0,
                        turn.cache_export_seconds * 1_000.0,
                    )
                }
            },
        );
        if self.language.is_chinese() {
            format!(
                "Tina 运行状态\n  engine: fast · 0731-main · no-MTP · no-D-Spark\n  会话: {} · epoch {} · 归档消息 {} · 活动消息 {}\n  owner: {owner} · 位置 {}/{} · 压力 {pressure}\n  Agent: {agent_mode} · tool-host {tool_host}\n  workspace: {workspace}\n  thinking: {}\n  KV cache: mode={} · directory={cache_directory} · 全局命中率 NOT_MEASURED\n{last_turn}",
                status.conversation_id,
                status.epoch,
                status.archived_messages,
                status.active_messages,
                compact_count(status.context_positions),
                compact_count(status.budget.display_capacity()),
                status.thinking_level,
                status.cache_mode,
            )
        } else {
            format!(
                "Tina runtime status\n  engine: fast · 0731-main · no-MTP · no-D-Spark\n  conversation: {} · epoch {} · archived messages {} · active messages {}\n  owner: {owner} · positions {}/{} · pressure {pressure}\n  Agent: {agent_mode} · tool-host {tool_host}\n  workspace: {workspace}\n  thinking: {}\n  KV cache: mode={} · directory={cache_directory} · global hit rate NOT_MEASURED\n{last_turn}",
                status.conversation_id,
                status.epoch,
                status.archived_messages,
                status.active_messages,
                compact_count(status.context_positions),
                compact_count(status.budget.display_capacity()),
                status.thinking_level,
                status.cache_mode,
            )
        }
    }

    pub(crate) fn workspace_command_unavailable_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "该指令需要使用 --agent --chat --workspace <目录> 启动 Tina。"
        } else {
            "This command requires Tina to start with --agent --chat --workspace <dir>."
        }
    }

    pub(crate) fn review_started_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "已启动只读代码审查；本轮暴露 read/list/search/git status/git diff，以及已配置时的 web_search，写工具会被运行时拒绝。"
        } else {
            "Started a read-only code review; this turn exposes read/list/search/git status/git diff plus web_search when configured, and the runtime rejects mutating tools."
        }
    }

    pub(crate) fn review_prompt(&self, focus: Option<&str>) -> String {
        let mut prompt = String::from(
            "Review the current workspace changes. Inspect staged, unstaged, and untracked work. Report concrete bugs, security or safety risks, behavioral regressions, and missing tests first, ordered by severity, with file and line references. Keep the review read-only: do not modify files and do not request mutating tools. If there are no findings, say so explicitly and state the remaining test gaps.",
        );
        if let Some(focus) = focus.filter(|value| !value.trim().is_empty()) {
            prompt.push_str("\n\nAdditional user focus:\n");
            prompt.push_str(focus.trim());
        }
        prompt
    }

    pub(crate) fn workspace_diff_text(&self, status: &str, unstaged: &str, staged: &str) -> String {
        let status = nonempty_or(status, "(clean)");
        let unstaged = nonempty_or(unstaged, "(none)");
        let staged = nonempty_or(staged, "(none)");
        if self.language.is_chinese() {
            format!(
                "Git 工作树\n\n状态（含未跟踪文件）\n{status}\n\n未暂存差异\n{unstaged}\n\n已暂存差异\n{staged}"
            )
        } else {
            format!(
                "Git working tree\n\nStatus (including untracked files)\n{status}\n\nUnstaged diff\n{unstaged}\n\nStaged diff\n{staged}"
            )
        }
    }

    pub(crate) fn tools_unavailable_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "Agent tool-host 不可用；请使用 --agent --chat --workspace <目录> 启动 Tina。"
        } else {
            "The Agent tool host is unavailable; start Tina with --agent --chat --workspace <dir>."
        }
    }

    pub(crate) fn web_search_status_text(&self, available: bool) -> &'static str {
        match (self.language.is_chinese(), available) {
            (true, true) => {
                "网络搜索：AVAILABLE（SearXNG JSON，受控只读）。使用 /search <查询>；外部结果按不可信数据处理。"
            }
            (true, false) => {
                "网络搜索：NOT_AVAILABLE。请用 --agent --chat --workspace <目录> --web-search-url <SearXNG URL> 显式配置；不会伪造 URL、摘要或引用。"
            }
            (false, true) => {
                "Web search: AVAILABLE (bounded read-only SearXNG JSON). Use /search <query>; external results are treated as untrusted data."
            }
            (false, false) => {
                "Web search: NOT_AVAILABLE. Configure --agent --chat --workspace <dir> --web-search-url <SearXNG URL>; URLs, summaries, and citations will not be fabricated."
            }
        }
    }

    pub(crate) fn web_search_unavailable_text(&self, query: &str) -> String {
        if self.language.is_chinese() {
            format!("网络搜索未执行：web_search 能力不可用；query={query:?}。状态为 NOT_AVAILABLE，没有网络 receipt。")
        } else {
            format!("Web search was not executed: web_search is unavailable; query={query:?}. Status=NOT_AVAILABLE and no network receipt exists.")
        }
    }

    pub(crate) fn web_search_receipt_text(&self, receipt: &str) -> String {
        if self.language.is_chinese() {
            format!("网络搜索 receipt（外部不可信数据）\n{receipt}")
        } else {
            format!("Web-search receipt (untrusted external data)\n{receipt}")
        }
    }

    pub(crate) fn web_search_failed_text(&self, query: &str, error: &str) -> String {
        if self.language.is_chinese() {
            format!("网络搜索失败：query={query:?}；{error}。没有成功 receipt，不得视为搜索结果。")
        } else {
            format!("Web search failed: query={query:?}; {error}. Without a successful receipt, no search result exists.")
        }
    }

    pub(crate) fn agent_restart_text(&self, detail: &str) -> String {
        if self.language.is_chinese() {
            format!(
                "已重启 Agent tool-host；会话归档和已完成工具结果保留，下一轮将创建新的模型 owner。{detail}"
            )
        } else {
            format!(
                "Restarted the Agent tool host; the conversation archive and completed tool results are retained. The next turn will create a fresh model owner. {detail}"
            )
        }
    }

    pub(crate) fn agent_restart_failed_text(&self, detail: &str) -> String {
        if self.language.is_chinese() {
            format!("Agent tool-host 重启失败；旧 host 仍保留，当前会话未被清空：{detail}")
        } else {
            format!(
                "Agent tool-host restart failed; the old host was retained and the conversation was not cleared: {detail}"
            )
        }
    }

    pub(crate) fn tools_text(
        &self,
        binary: &Path,
        workspace: &Path,
        full_access: bool,
        read_only: &[&str],
        mutating: &[&str],
    ) -> String {
        let read_only = if read_only.is_empty() {
            "(none)".to_string()
        } else {
            read_only.join(", ")
        };
        let mutating = if mutating.is_empty() {
            "(none)".to_string()
        } else {
            mutating.join(", ")
        };
        let scope = if full_access {
            "machine-wide files + shell + network"
        } else if read_only.contains("web_search") {
            "workspace files + configured web search"
        } else {
            "workspace only"
        };
        let mutation_policy = if full_access {
            if self.language.is_chinese() {
                "启动时已确认 FULL_ACCESS；结构化 delete 仍需精确 DELETE"
            } else {
                "startup FULL_ACCESS acknowledged; structured delete still requires exact DELETE"
            }
        } else {
            if self.language.is_chinese() {
                "结构化写入/创建/重命名每次需要精确 APPLY；delete 需要 DELETE"
            } else {
                "each structured write/create/rename requires exact APPLY; delete requires DELETE"
            }
        };
        let execution_policy = if self.language.is_chinese() {
            "workspace_exec 每次都需精确 EXECUTE；它不是 OS 沙箱，构建脚本/测试继承宿主环境与网络"
        } else {
            "workspace_exec always requires exact EXECUTE; it is not an OS sandbox and build scripts/tests inherit host environment and network"
        };
        if self.language.is_chinese() {
            format!(
                "Agent 工具能力\n  host: {}\n  workspace 锚点: {}\n  范围: {scope}\n  只读（自动允许）: {read_only}\n  写入/执行: {mutating}\n  写入授权: {mutation_policy}\n  执行授权: {execution_policy}\n  禁用: MCP、子代理（未来插件扩展点）\n  tool-host 协议: tina.tool-host.v2；输出有界、可取消、失败关闭",
                binary.display(),
                workspace.display(),
            )
        } else {
            format!(
                "Agent tool capabilities\n  host: {}\n  workspace anchor: {}\n  scope: {scope}\n  read-only (auto-approved): {read_only}\n  write/execute: {mutating}\n  write authorization: {mutation_policy}\n  execution authorization: {execution_policy}\n  disabled: MCP, subagents (future plugin extension points)\n  tool-host protocol: tina.tool-host.v2; bounded, cancellable, fail-closed",
                binary.display(),
                workspace.display(),
            )
        }
    }

    pub(crate) fn session_notice(&self, command: SessionCommand, detail: &str) -> String {
        if self.language.is_chinese() {
            match command {
                SessionCommand::Agent => detail.to_string(),
                SessionCommand::New => format!("已开始新会话；旧归档保留。{detail}"),
                SessionCommand::Compact => format!("已创建上下文交接。{detail}"),
                SessionCommand::Context => detail.to_string(),
                SessionCommand::Status => detail.to_string(),
                SessionCommand::Tools => detail.to_string(),
                SessionCommand::AgentRestart => detail.to_string(),
                SessionCommand::Environment => detail.to_string(),
                SessionCommand::Sessions => detail.to_string(),
                SessionCommand::Retry => format!("已创建重试分支。{detail}"),
            }
        } else {
            match command {
                SessionCommand::Agent => detail.to_string(),
                SessionCommand::New => {
                    format!("Started a new session; the old archive is retained. {detail}")
                }
                SessionCommand::Compact => format!("Created a context handoff. {detail}"),
                SessionCommand::Context => detail.to_string(),
                SessionCommand::Status => detail.to_string(),
                SessionCommand::Tools => detail.to_string(),
                SessionCommand::AgentRestart => detail.to_string(),
                SessionCommand::Environment => detail.to_string(),
                SessionCommand::Sessions => detail.to_string(),
                SessionCommand::Retry => format!("Created a retry branch. {detail}"),
            }
        }
    }

    pub(crate) fn conversation_opened_text(
        &self,
        resumed: bool,
        conversation_id: &str,
        epoch: u64,
        archived_messages: usize,
    ) -> String {
        if self.language.is_chinese() {
            if resumed {
                format!(
                    "已恢复 Tina 会话 {conversation_id}，进入新 epoch {epoch}（归档消息 {archived_messages} 条）。"
                )
            } else {
                format!("已开始 Tina 会话 {conversation_id}，epoch {epoch}。")
            }
        } else if resumed {
            format!(
                "Resumed Tina conversation {conversation_id} in fresh epoch {epoch} ({archived_messages} archived messages)."
            )
        } else {
            format!("Started Tina conversation {conversation_id} in epoch {epoch}.")
        }
    }

    pub(crate) fn new_session_detail_text(&self, conversation_id: &str, epoch: u64) -> String {
        if self.language.is_chinese() {
            format!("会话 {conversation_id}，epoch {epoch}。")
        } else {
            format!("conversation {conversation_id} epoch {epoch}.")
        }
    }

    pub(crate) fn handoff_detail_text(
        &self,
        compacted_messages: usize,
        retained_messages: usize,
        source_epoch: u64,
        target_epoch: u64,
    ) -> String {
        if self.language.is_chinese() {
            format!(
                "压缩 {compacted_messages} 条消息，保留 {retained_messages} 条，epoch {source_epoch} -> {target_epoch}。"
            )
        } else {
            format!(
                "compacted {compacted_messages} message(s), retained {retained_messages} and moved epoch {source_epoch} -> {target_epoch}."
            )
        }
    }

    pub(crate) fn automatic_handoff_text(
        &self,
        compacted_messages: usize,
        retained_messages: usize,
        source_epoch: u64,
        target_epoch: u64,
    ) -> String {
        if self.language.is_chinese() {
            format!(
                "上下文已自动交接：压缩 {compacted_messages} 条归档消息，保留 {retained_messages} 条近期消息，epoch {source_epoch} -> {target_epoch}；随后创建新的 owner。"
            )
        } else {
            format!(
                "Context handoff compacted {compacted_messages} archived message(s), retained {retained_messages} recent message(s), and moved epoch {source_epoch} -> {target_epoch} before starting a fresh owner."
            )
        }
    }

    pub(crate) fn deepseek_import_text(
        &self,
        title: &str,
        messages: usize,
        leaf_id: &str,
        branch_nodes: usize,
        omitted_fragments: usize,
        dropped_messages: usize,
    ) -> String {
        if self.language.is_chinese() {
            format!(
                "已导入 DeepSeek 会话 {title:?}：分支 {leaf_id}，{messages} 条消息、{branch_nodes} 个节点；忽略 {omitted_fragments} 个非对话片段，丢弃 {dropped_messages} 条不完整消息。"
            )
        } else {
            format!(
                "Imported DeepSeek conversation {title:?}: {messages} messages on branch {leaf_id} ({branch_nodes} nodes, {omitted_fragments} non-chat fragments omitted, {dropped_messages} incomplete messages dropped)."
            )
        }
    }

    pub(crate) fn pinned_fact_text(&self, pin_id: &str) -> String {
        if self.language.is_chinese() {
            format!("已固定事实 {pin_id}。删除归档仍需通过精确 DELETE 拦截。")
        } else {
            format!("Pinned fact {pin_id}. Archive deletion remains behind the exact DELETE gate.")
        }
    }

    pub(crate) fn todo_invalid_text(&self, detail: &str) -> String {
        if self.language.is_chinese() {
            format!("待办指令无效。{}", self.todo_usage_text())
        } else {
            detail.to_string()
        }
    }

    pub(crate) fn todo_error_text(&self, detail: &str) -> String {
        if self.language.is_chinese() {
            format!("待办操作失败：{detail}")
        } else {
            detail.to_string()
        }
    }

    pub(crate) fn todo_added_text(&self, todo_id: &str, content: &str) -> String {
        if self.language.is_chinese() {
            format!("已添加待办 {todo_id}：{content}")
        } else {
            format!("Added TodoList item {todo_id}: {content}")
        }
    }

    pub(crate) fn todo_status_text(&self, todo_id: &str, status: &str, content: &str) -> String {
        let status = if self.language.is_chinese() {
            match status {
                "pending" => "待处理",
                "in_progress" => "进行中",
                "completed" => "已完成",
                other => other,
            }
        } else {
            status
        };
        if self.language.is_chinese() {
            format!("待办 {todo_id} 当前状态：{status}。{content}")
        } else {
            format!("TodoList item {todo_id} is now {status}: {content}")
        }
    }

    pub(crate) fn todo_delete_target_text(&self, todo_id: &str, content: &str) -> String {
        if self.language.is_chinese() {
            format!("待办 {todo_id}（{content}）")
        } else {
            format!("TodoList item {todo_id} ({content})")
        }
    }

    pub(crate) fn todo_removed_text(&self, todo_id: &str) -> String {
        if self.language.is_chinese() {
            format!("已删除待办 {todo_id}。")
        } else {
            format!("Removed TodoList item {todo_id}.")
        }
    }

    pub(crate) fn todo_no_completed_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "没有已完成的待办可清理。"
        } else {
            "TodoList has no completed items to clear."
        }
    }

    pub(crate) fn completed_todos_delete_target_text(&self, count: usize) -> String {
        if self.language.is_chinese() {
            format!("{count} 条已完成待办")
        } else {
            format!("{count} completed TodoList item(s)")
        }
    }

    pub(crate) fn todos_cleared_text(&self, count: usize) -> String {
        if self.language.is_chinese() {
            format!("已清理 {count} 条已完成待办。")
        } else {
            format!("Cleared {count} completed TodoList item(s).")
        }
    }

    pub(crate) fn agent_tool_result_text(&self, tool: &str, ok: bool) -> String {
        if self.language.is_chinese() {
            format!("工具 {tool}：{}", if ok { "已完成" } else { "失败" })
        } else {
            format!("{tool}: {}", if ok { "completed" } else { "failed" })
        }
    }

    pub(crate) fn agent_budget_closed_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "Agent 工具时间预算已耗尽；所有工具终态已归档，本轮已关闭"
        } else {
            "Agent tool wall budget exhausted; the turn was closed after terminal tool results"
        }
    }

    pub(crate) fn sessions_text(
        &self,
        sessions: &[crate::conversation_memory::ConversationSessionSummary],
        current_id: &str,
    ) -> String {
        if sessions.is_empty() {
            return if self.language.is_chinese() {
                "没有可恢复的会话归档。".to_string()
            } else {
                "No resumable conversation archives were found.".to_string()
            };
        }
        let mut output = if self.language.is_chinese() {
            String::from("会话归档（按已验证的活动时间排序）")
        } else {
            String::from("Conversation archives (ordered by validated activity time)")
        };
        for session in sessions {
            let current = if session.conversation_id == current_id {
                if self.language.is_chinese() {
                    " [当前]"
                } else {
                    " [current]"
                }
            } else {
                ""
            };
            output.push_str(&format!(
                "\n  {}{} · activity_unix_ms={} · epoch={} · turns={} · events={}",
                session.conversation_id,
                current,
                session.last_activity_unix_ms,
                session.current_epoch,
                session.turn_count,
                session.event_count,
            ));
        }
        if self.language.is_chinese() {
            output.push_str("\n恢复：重启时使用 --resume-conversation <完整会话 ID>。");
        } else {
            output.push_str(
                "\nResume by restarting with --resume-conversation <full conversation ID>.",
            );
        }
        output
    }

    pub(crate) fn environment_text(&self, status: &str) -> String {
        if self.language.is_chinese() {
            format!(
                "终端环境（已注入模型 system message）\n  {status}\n  可用 --shell powershell|cmd|bash|zsh|fish|sh 显式覆盖自动探测。"
            )
        } else {
            format!(
                "Terminal environment (injected into the model system message)\n  {status}\n  Override detection with --shell powershell|cmd|bash|zsh|fish|sh."
            )
        }
    }

    pub(crate) fn memory_status_text(
        &self,
        conversation_id: &str,
        epoch: u64,
        archived_messages: usize,
        active_messages: usize,
        unfinished_todos: usize,
        archive_root: &std::path::Path,
    ) -> String {
        if self.language.is_chinese() {
            format!(
                "会话 {} · epoch {} · 归档消息 {} · 活动消息 {} · 未完成待办 {} · 归档 {}",
                conversation_id,
                epoch,
                archived_messages,
                active_messages,
                unfinished_todos,
                archive_root.display(),
            )
        } else {
            format!(
                "Conversation {} | epoch {} | {} archived messages | {} active messages | {} unfinished TodoList items | archive {}",
                conversation_id,
                epoch,
                archived_messages,
                active_messages,
                unfinished_todos,
                archive_root.display(),
            )
        }
    }

    pub(crate) fn remember_usage_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "用法：/remember <事实>（别名 /记住）。固定事实会写入不可变归档。"
        } else {
            "Usage: /remember <fact> (alias /记住). Pinned facts are immutable archive records."
        }
    }

    pub(crate) fn project_memory_usage_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "用法：/project-memory [status|remember <事实>|forget <ID>]；也可用 /project-remember <事实>、/project-forget <ID>。忘记操作必须精确输入 DELETE。"
        } else {
            "Usage: /project-memory [status|remember <fact>|forget <ID>], /project-remember <fact>, or /project-forget <ID>. Forgetting requires the exact token DELETE."
        }
    }

    pub(crate) fn project_memory_status_text(
        &self,
        status: crate::project_memory::ProjectMemoryStatus<'_>,
    ) -> String {
        let state = if self.language.is_chinese() {
            match status.state {
                crate::project_memory::ProjectMemoryState::Ready => "可用",
                crate::project_memory::ProjectMemoryState::Unavailable => "不可用",
                crate::project_memory::ProjectMemoryState::Corrupt => "损坏并关闭",
            }
        } else {
            status.state.as_str()
        };
        let commit = status
            .commit
            .and_then(|value| value.get(..12))
            .unwrap_or("not-created");
        let mut text = if self.language.is_chinese() {
            format!(
                "Git 项目记忆\n  状态：{state}\n  workspace：{}\n  ref：{}\n  commit：{commit}\n  当前事实：{}",
                status.workspace.display(),
                status.reference,
                status.entries.len(),
            )
        } else {
            format!(
                "Git project memory\n  state: {state}\n  workspace: {}\n  ref: {}\n  commit: {commit}\n  current facts: {}",
                status.workspace.display(),
                status.reference,
                status.entries.len(),
            )
        };
        for entry in status.entries.iter().rev().take(20) {
            text.push_str(&format!(
                "\n  {}  {}",
                entry.entry_id.get(..12).unwrap_or(&entry.entry_id),
                single_line_excerpt(&entry.content, 120),
            ));
        }
        if status.entries.len() > 20 {
            text.push_str(&format!(
                "\n  ... {}",
                if self.language.is_chinese() {
                    format!("另有 {} 条事实未显示", status.entries.len() - 20)
                } else {
                    format!("{} additional fact(s) omitted", status.entries.len() - 20)
                }
            ));
        }
        if let Some(diagnostic) = status.diagnostic {
            text.push_str(if self.language.is_chinese() {
                "\n  诊断："
            } else {
                "\n  diagnostic: "
            });
            text.push_str(diagnostic);
        }
        text
    }

    pub(crate) fn project_memory_remembered_text(
        &self,
        entry_id: &str,
        commit_oid: &str,
        created: bool,
    ) -> String {
        let entry_id = entry_id.get(..12).unwrap_or(entry_id);
        let commit_oid = commit_oid.get(..12).unwrap_or(commit_oid);
        if self.language.is_chinese() {
            if created {
                format!("已保存项目事实 {entry_id}，Git 记忆提交 {commit_oid}。")
            } else {
                format!("项目事实 {entry_id} 已存在；未创建重复提交。")
            }
        } else if created {
            format!("Saved project fact {entry_id} in Git memory commit {commit_oid}.")
        } else {
            format!("Project fact {entry_id} already exists; no duplicate commit was created.")
        }
    }

    pub(crate) fn project_memory_forget_target_text(
        &self,
        entry_id: &str,
        content: &str,
    ) -> String {
        let entry_id = entry_id.get(..12).unwrap_or(entry_id);
        let content = single_line_excerpt(content, 160);
        if self.language.is_chinese() {
            format!("Git 项目记忆 {entry_id}（{content}）")
        } else {
            format!("Git project-memory fact {entry_id} ({content})")
        }
    }

    pub(crate) fn project_memory_forgotten_text(&self, entry_id: &str, commit_oid: &str) -> String {
        let entry_id = entry_id.get(..12).unwrap_or(entry_id);
        let commit_oid = commit_oid.get(..12).unwrap_or(commit_oid);
        if self.language.is_chinese() {
            format!("已从当前项目记忆忘记 {entry_id}；历史保留在父提交，当前提交 {commit_oid}。")
        } else {
            format!("Forgot project fact {entry_id}; history remains in the Git parent commit and the current commit is {commit_oid}.")
        }
    }

    pub(crate) fn project_memory_error_text(&self, detail: &str) -> String {
        if self.language.is_chinese() {
            format!("项目记忆操作失败：{detail}")
        } else {
            format!("Project memory operation failed: {detail}")
        }
    }

    pub(crate) fn todo_usage_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "用法：/todo（别名 /待办、/任务）[list|add <任务>|start <编号>|done <编号>|reopen <编号>|remove <编号>|clear-completed|help]"
        } else {
            "Usage: /todo (aliases /待办, /任务) [list|add <task>|start <id>|done <id>|reopen <id>|remove <id>|clear-completed|help]"
        }
    }

    pub(crate) fn export_usage_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "用法：/export [jsonl|md] <路径>。jsonl 可直接作为 --conversation-jsonl 输入；覆盖已有文件需要输入 OVERWRITE。"
        } else {
            "Usage: /export [jsonl|md] <path>. JSONL can be fed to --conversation-jsonl; existing files require OVERWRITE."
        }
    }

    pub(crate) fn export_success_text(
        &self,
        format: ExportFormat,
        path: &std::path::Path,
        messages: usize,
        bytes: usize,
    ) -> String {
        if self.language.is_chinese() {
            format!(
                "已导出 {} 条活动消息为 {}（{} 字节）：{}",
                messages,
                match format {
                    ExportFormat::Jsonl => "JSONL",
                    ExportFormat::Markdown => "Markdown",
                },
                bytes,
                path.display(),
            )
        } else {
            format!(
                "Exported {} active message(s) as {} ({} bytes): {}",
                messages,
                match format {
                    ExportFormat::Jsonl => "JSONL",
                    ExportFormat::Markdown => "Markdown",
                },
                bytes,
                path.display(),
            )
        }
    }

    pub(crate) fn export_error_text(&self, error: &str) -> String {
        if self.language.is_chinese() {
            format!("导出失败：{error}")
        } else {
            format!("Export failed: {error}")
        }
    }

    pub(crate) fn unknown_command_text(&self, command: &str) -> String {
        if self.language.is_chinese() {
            format!("未知指令 {command:?}。输入 /help 查看可用指令。")
        } else {
            format!("Unknown command {command:?}. Use /help to list available commands.")
        }
    }

    pub(crate) fn assistant_id_text(&self) -> String {
        if self.language.is_chinese() {
            format!(
                "助手 ID：{}。使用 /assistant <id> 修改。",
                self.assistant_id
            )
        } else {
            format!(
                "Assistant ID: {}. Use /assistant <id> to change it.",
                self.assistant_id
            )
        }
    }

    pub(crate) fn language_text(&self) -> String {
        let name = if self.language.is_chinese() {
            "中文"
        } else {
            "English"
        };
        if self.language.is_chinese() {
            format!("界面语言：{name}。使用 /language en|zh|auto 切换。")
        } else {
            format!("UI language: {name}. Use /language en|zh|auto to switch.")
        }
    }

    pub(crate) fn set_assistant_id_text(&self) -> String {
        if self.language.is_chinese() {
            format!("助手 ID 已切换为 {}。", self.assistant_id)
        } else {
            format!("Assistant ID changed to {}.", self.assistant_id)
        }
    }

    pub(crate) fn set_language_text(&self) -> String {
        if self.language.is_chinese() {
            "界面语言已切换为中文。".to_string()
        } else {
            "UI language changed to English.".to_string()
        }
    }

    pub(crate) fn thinking_text(&self, level: ThinkingLevel) -> String {
        if self.language.is_chinese() {
            format!(
                "当前思考级别：{}。使用 /thinking off|medium|high|max 切换。",
                level.as_str()
            )
        } else {
            format!(
                "Thinking level: {}. Use /thinking off|medium|high|max to change.",
                level.as_str()
            )
        }
    }

    pub(crate) fn set_thinking_text(&self, level: ThinkingLevel) -> String {
        if self.language.is_chinese() {
            format!(
                "思考级别已切换为 {}。当前 owner 将在下一轮重新创建。",
                level.as_str()
            )
        } else {
            format!("Thinking level changed to {}. The current owner will be recreated for the next turn.", level.as_str())
        }
    }

    pub(crate) fn goal_text(&self, goal: Option<&str>) -> String {
        match (self.language.is_chinese(), goal) {
            (true, Some(goal)) => format!("当前会话目标：{goal}"),
            (true, None) => "当前会话未设置目标。使用 /goal set <目标>。".to_string(),
            (false, Some(goal)) => format!("Current session goal: {goal}"),
            (false, None) => "No session goal is set. Use /goal set <text>.".to_string(),
        }
    }

    pub(crate) fn goal_changed_text(&self, goal: Option<&str>) -> String {
        match (self.language.is_chinese(), goal) {
            (true, Some(_)) => "会话目标已保存；当前 owner 将在下一轮重新创建。".to_string(),
            (true, None) => "会话目标已清除；当前 owner 将在下一轮重新创建。".to_string(),
            (false, Some(_)) => {
                "Session goal saved; the current owner will be recreated for the next turn."
                    .to_string()
            }
            (false, None) => {
                "Session goal cleared; the current owner will be recreated for the next turn."
                    .to_string()
            }
        }
    }

    pub(crate) fn agent_status_text(
        &self,
        enabled: bool,
        full_access: bool,
        workspace: Option<&Path>,
        tool_host_ready: bool,
    ) -> String {
        let workspace =
            workspace.map_or_else(|| "none".to_string(), |path| path.display().to_string());
        if self.language.is_chinese() {
            if enabled {
                format!(
                    "Agent：已启用 · 模式：{} · tool-host：{} · workspace：{workspace}",
                    if full_access {
                        "full-access"
                    } else {
                        "bounded"
                    },
                    if tool_host_ready {
                        "ready"
                    } else {
                        "unavailable"
                    },
                )
            } else {
                "Agent：未启用。启动方式：cq-infer --chat --agent --workspace <目录>".to_string()
            }
        } else if enabled {
            format!(
                "Agent: enabled · mode: {} · tool-host: {} · workspace: {workspace}",
                if full_access {
                    "full-access"
                } else {
                    "bounded"
                },
                if tool_host_ready {
                    "ready"
                } else {
                    "unavailable"
                },
            )
        } else {
            "Agent: disabled. Start with: cq-infer --chat --agent --workspace <dir>".to_string()
        }
    }

    pub(crate) fn invalid_setting_text(&self, error: &str) -> String {
        if self.language.is_chinese() {
            format!("设置无效：{error}")
        } else {
            format!("Invalid setting: {error}")
        }
    }

    pub(crate) fn start(&mut self) -> io::Result<()> {
        if !self.interactive {
            return Ok(());
        }
        let mut out = io::stdout().lock();
        let rows = [
            "0731-main · single-sequence · no-MTP/no-D-Spark".to_string(),
            format!(
                "{} · {} · context 0/258K · KV opt-in · thinking off",
                startup_os_label(),
                startup_shell_label(),
            ),
        ];
        for line in startup_frame_lines(
            &format!("TINA ENGINE · {}", self.assistant_id()),
            "FAST",
            &rows,
            composer_width(),
        ) {
            write_styled(&mut out, MOON_BRIGHT, self.color, &format!("{line}\r\n"))?;
        }
        write!(out, "\r\n")?;
        write_styled(
            &mut out,
            MUTED,
            self.color,
            &format!(
                "  {}\r\n\r\n\r\n",
                if self.language.is_chinese() {
                    "输入消息开始 · / 打开指令 · ↑↓ 选择 · Tab 补全 · Ctrl-C 取消"
                } else {
                    "Enter a message · / opens commands · ↑↓ select · Tab completes · Ctrl-C cancels"
                }
            ),
        )?;
        out.flush()
    }

    pub(crate) fn start_live(
        &mut self,
        model: &str,
        endpoint: &str,
        capacity: usize,
        thinking: &str,
    ) -> io::Result<()> {
        if !self.interactive {
            return Ok(());
        }
        let rows = [
            format!(
                "{} · {}",
                terminal_safe_line(model),
                terminal_safe_line(endpoint)
            ),
            format!(
                "{} · {} · context {} · thinking {}",
                startup_os_label(),
                startup_shell_label(),
                compact_count(capacity),
                thinking
            ),
        ];
        let mut out = io::stdout().lock();
        for line in startup_frame_lines(
            &format!("TINA CLI · {}", self.assistant_id()),
            "LIVE",
            &rows,
            composer_width(),
        ) {
            write_styled(&mut out, MOON_BRIGHT, self.color, &format!("{line}\r\n"))?;
        }
        write!(
            out,
            "\r\n  /help · /context · /new · /resume · Ctrl-C cancel\r\n\r\n"
        )?;
        out.flush()
    }

    pub(crate) fn read_message(
        &mut self,
        context_positions: usize,
        capacity: usize,
        agent_enabled: bool,
        thinking: &str,
    ) -> io::Result<(ChatInput, String)> {
        if !self.interactive {
            let mut out = io::stdout().lock();
            write_styled(&mut out, MOON_BRIGHT, self.color, "User > ")?;
            out.flush()?;
            let mut line = String::new();
            let read = std::io::stdin().read_line(&mut line)?;
            if read == 0 {
                return Ok((ChatInput::Exit, String::new()));
            }
            let message = line.trim_end_matches(['\r', '\n']).to_string();
            if is_exit_command(&message) {
                return Ok((ChatInput::Exit, String::new()));
            }
            if let Some(command) = command_line(&message) {
                return Ok((ChatInput::Command(command), String::new()));
            }
            return Ok((ChatInput::Submit, unescape_literal_prompt(message)));
        }

        let raw = RawInputGuard::enable()?;
        let mut out = io::stdout().lock();
        execute!(out, EnableBracketedPaste)?;
        let mut editor = EditorBuffer::new();
        let mut slash_selection = 0;
        let mut rendered = None;
        let mut show_placeholder = true;
        let composer = ComposerContext {
            context_positions,
            capacity,
            agent_enabled,
            thinking,
            color: self.color,
            language: self.language,
            assistant_id: &self.assistant_id,
        };
        redraw_composer(
            &mut out,
            &editor,
            composer,
            slash_selection,
            show_placeholder,
            &mut rendered,
        )?;

        loop {
            let input = event::read()?;
            let key = match input {
                Event::Paste(pasted) => {
                    slash_selection = 0;
                    if !pasted.is_empty() {
                        show_placeholder = false;
                    }
                    if !editor.insert(&pasted) && !pasted.is_empty() {
                        clear_composer(&mut out, rendered)?;
                        rendered = None;
                        let warning = paste_rejected_text(self.language, pasted.len());
                        write_styled(&mut out, MUTED, self.color, &format!("  {warning}\r\n\r\n"))?;
                    }
                    redraw_composer(
                        &mut out,
                        &editor,
                        composer,
                        slash_selection,
                        show_placeholder,
                        &mut rendered,
                    )?;
                    continue;
                }
                Event::Key(key) => key,
                Event::Resize(_, _) => {
                    redraw_composer(
                        &mut out,
                        &editor,
                        composer,
                        slash_selection,
                        show_placeholder,
                        &mut rendered,
                    )?;
                    continue;
                }
                _ => continue,
            };
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                continue;
            }
            show_placeholder = false;

            let width = composer_inner_width();
            let popup_count = slash_popup_candidates(&editor.text, self.language).len();
            if popup_count == 0 {
                slash_selection = 0;
            } else {
                slash_selection = slash_selection.min(popup_count - 1);
            }
            let plain_enter = matches!(key.code, KeyCode::Enter)
                && !key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT | KeyModifiers::CONTROL);
            if plain_enter && popup_count > 0 && slash_popup_selectable(&editor) {
                replace_with_slash_candidate(&mut editor, self.language, slash_selection);
                slash_selection = 0;
            }
            let result = if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c' | 'C'))
            {
                if composer_interrupt_requests_exit(&mut editor) {
                    Some((ChatInput::Exit, String::new()))
                } else {
                    self.history_cursor = None;
                    None
                }
            } else if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('d'))
                && editor.text.is_empty()
            {
                Some((ChatInput::Exit, String::new()))
            } else if popup_count > 0
                && slash_popup_selectable(&editor)
                && matches!(key.code, KeyCode::Up | KeyCode::Down)
            {
                move_slash_selection(
                    &mut slash_selection,
                    popup_count,
                    matches!(key.code, KeyCode::Down),
                );
                None
            } else if popup_count > 0
                && slash_popup_selectable(&editor)
                && matches!(key.code, KeyCode::Tab)
            {
                replace_with_slash_candidate(&mut editor, self.language, slash_selection);
                slash_selection = 0;
                self.history_cursor = None;
                None
            } else if plain_enter {
                if editor.text.trim().is_empty() {
                    None
                } else if is_exit_command(&editor.text) {
                    Some((ChatInput::Exit, String::new()))
                } else if let Some(command) = command_line(&editor.text) {
                    self.history.push(editor.text.clone());
                    self.history_cursor = None;
                    Some((ChatInput::Command(command), String::new()))
                } else {
                    let message = unescape_literal_prompt(editor.text.clone());
                    self.history.push(message.clone());
                    self.history_cursor = None;
                    Some((ChatInput::Submit, message))
                }
            } else if matches!(key.code, KeyCode::Enter)
                || (matches!(key.code, KeyCode::Char('j'))
                    && key.modifiers.contains(KeyModifiers::CONTROL))
            {
                editor.insert("\n");
                None
            } else {
                handle_editor_key(
                    &mut editor,
                    key,
                    width,
                    &self.history,
                    &mut self.history_cursor,
                );
                slash_selection = 0;
                None
            };

            if let Some(result) = result {
                clear_composer(&mut out, rendered)?;
                out.flush()?;
                // Release stdout before RawInputGuard restores terminal state. The
                // guard's Drop implementation acquires stdout for cleanup.
                drop(out);
                drop(raw);
                return Ok(result);
            }
            redraw_composer(
                &mut out,
                &editor,
                composer,
                slash_selection,
                show_placeholder,
                &mut rendered,
            )?;
        }
    }

    pub(crate) fn show_user(&self, message: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.interactive {
            let width = composer_width();
            write_frame_top(&mut out, width, "User", MOON_BRIGHT, self.color)?;
            write_framed_text(
                &mut out,
                message,
                width.saturating_sub(6).max(1),
                MOON,
                self.color,
            )?;
            write_frame_bottom(&mut out, width, "", MUTED, "", MOON, self.color)?;
            write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
            return out.flush();
        }

        let mut first = true;
        for line in message.lines() {
            write_styled(
                &mut out,
                MOON_BRIGHT,
                self.color,
                if first { "User › " } else { "       " },
            )?;
            write_safe_assistant_piece(&mut out, line)?;
            write!(out, "\r\n")?;
            first = false;
        }
        if first {
            write_styled(&mut out, MOON_BRIGHT, self.color, "User › ")?;
            write!(out, "\r\n")?;
        }
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        out.flush()
    }

    pub(crate) fn begin_assistant(&self) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.interactive {
            if self.assistant_frame_open.get() {
                self.close_assistant_frame(&mut out, " interrupted ", "")?;
            }
            let width = composer_width();
            write_frame_top(
                &mut out,
                width,
                &format!("{} ☾", self.assistant_id()),
                MOON_BRIGHT,
                self.color,
            )?;
            write_frame_body_prefix(&mut out, self.color)?;
            self.assistant_frame_open.set(true);
            self.assistant_line_open.set(true);
            self.assistant_body_width
                .set(width.saturating_sub(6).max(1));
            self.assistant_column.set(0);
            return out.flush();
        }
        write_styled(
            &mut out,
            MOON_BRIGHT,
            self.color,
            &format!("{}\r\n", self.assistant_id()),
        )?;
        out.flush()
    }

    /// Put an interactive terminal in raw mode while decode is in flight so
    /// Ctrl-C can be observed at token boundaries. Non-terminal callers get a
    /// no-op guard and retain the JSONL/noninteractive behavior.
    pub(crate) fn begin_generation(&self) -> io::Result<GenerationGuard> {
        if !self.interactive {
            return Ok(GenerationGuard { enabled: false });
        }
        terminal::enable_raw_mode()?;
        Ok(GenerationGuard { enabled: true })
    }

    /// Drain pending terminal events and report only an explicit Ctrl-C. Other
    /// keys are ignored while the model owns the terminal; they are not fed to
    /// the prompt editor accidentally.
    pub(crate) fn poll_generation_cancel(&self) -> io::Result<bool> {
        if !self.interactive {
            return Ok(false);
        }
        while event::poll(Duration::from_millis(0))? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                continue;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c' | 'C'))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub(crate) fn generation_cancelled_text(&self) -> &'static str {
        if self.language.is_chinese() {
            "生成已取消；当前 KV owner 已安全丢弃，已切换到新会话归档。"
        } else {
            "Generation cancelled; the current KV owner was safely discarded and a new session archive is active."
        }
    }

    pub(crate) fn stream_assistant(&self, piece: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.interactive && self.assistant_frame_open.get() {
            let body_width = self.assistant_body_width.get().max(1);
            let mut column = self.assistant_column.get();
            let mut line_open = self.assistant_line_open.get();
            write_streamed_frame_piece(
                &mut out,
                piece,
                body_width,
                &mut column,
                &mut line_open,
                MOON_BRIGHT,
                self.color,
            )?;
            self.assistant_column.set(column);
            self.assistant_line_open.set(line_open);
            return out.flush();
        }
        write_bright_assistant_piece(&mut out, piece, self.color)?;
        out.flush()
    }

    pub(crate) fn end_assistant(
        &self,
        generated_tokens: usize,
        elapsed_seconds: f64,
        committed_positions: usize,
        capacity: usize,
    ) -> io::Result<()> {
        let mut out = io::stdout().lock();
        let rate = generated_tokens as f64 / elapsed_seconds.max(f64::EPSILON);
        let rate_text = if generated_tokens == 0 {
            "usage unavailable".to_string()
        } else {
            format!("{rate:.1} tok/s")
        };
        if self.interactive && self.assistant_frame_open.get() {
            let left = if self.language.is_chinese() {
                format!(" HTTP · 本轮 {rate_text} ")
            } else {
                format!(" HTTP · {rate_text} ")
            };
            let right = format!(
                " {}/{} ",
                compact_count(committed_positions),
                compact_count(capacity)
            );
            self.close_assistant_frame_with_reflection(&mut out, &left, &right)?;
            write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
            return out.flush();
        }

        write!(out, "\r\n")?;
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        let status = if self.language.is_chinese() {
            format!(
                "  · HTTP · 本轮 {} · 上下文 {}/{}\r\n",
                rate_text,
                compact_count(committed_positions),
                compact_count(capacity),
            )
        } else {
            format!(
                "  · HTTP · {} turn · {}/{} positions\r\n",
                rate_text,
                compact_count(committed_positions),
                compact_count(capacity),
            )
        };
        write_styled(&mut out, MUTED, self.color, &status)?;
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        out.flush()
    }

    pub(crate) fn end_error(&self, error: &str) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.interactive && self.assistant_frame_open.get() {
            self.close_assistant_frame(&mut out, " incomplete ", "")?;
        } else {
            write!(out, "\r\n")?;
        }
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        let prefix = if self.language.is_chinese() {
            "  ! 错误："
        } else {
            "  ! "
        };
        write_styled(&mut out, MUTED, self.color, prefix)?;
        write_safe_assistant_piece(&mut out, error)?;
        write!(out, "\r\n")?;
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        out.flush()
    }

    pub(crate) fn notice(&self, message: &str) -> io::Result<()> {
        self.notice_untrusted(message)
    }

    /// Render one tool result with a compact row gap. The enclosing call site
    /// finishes the batch with `end_tool_batch` before the next model step.
    pub(crate) fn tool_notice(&self, message: &str) -> io::Result<()> {
        self.notice_untrusted_with_gap(message, UI_TOOL_ROW_GAP_LINES)
    }

    /// Finish a group of tool results with an outer gap. Together with the
    /// final row gap this gives the next transcript block a clear reset while
    /// keeping the batch itself compact.
    pub(crate) fn end_tool_batch(&self) -> io::Result<()> {
        let mut out = io::stdout().lock();
        write_block_gap(&mut out, UI_TOOL_BATCH_GAP_LINES)?;
        out.flush()
    }

    /// Render tool/model supplied text without allowing terminal control
    /// sequences to become executable terminal input. Newlines and tabs are
    /// retained so Git diffs and structured diagnostics remain readable.
    pub(crate) fn notice_untrusted(&self, message: &str) -> io::Result<()> {
        self.notice_untrusted_with_gap(message, UI_BLOCK_GAP_LINES)
    }

    fn notice_untrusted_with_gap(&self, message: &str, blank_lines: usize) -> io::Result<()> {
        let mut out = io::stdout().lock();
        if self.interactive && self.assistant_frame_open.get() {
            self.close_assistant_frame(&mut out, " interrupted ", "")?;
            write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        }
        write_styled(&mut out, MUTED, self.color, "  ")?;
        write_safe_assistant_piece(&mut out, message)?;
        write!(out, "\r\n")?;
        write_block_gap(&mut out, blank_lines)?;
        out.flush()
    }

    fn close_assistant_frame(
        &self,
        out: &mut impl Write,
        left: &str,
        right: &str,
    ) -> io::Result<()> {
        if !self.assistant_frame_open.get() {
            return Ok(());
        }
        if self.assistant_line_open.get() {
            write_frame_body_suffix(
                out,
                self.assistant_body_width.get().max(1),
                self.assistant_column.get(),
                MOON_BRIGHT,
                self.color,
            )?;
        }
        write_frame_bottom(
            out,
            self.assistant_body_width.get().saturating_add(6),
            left,
            MUTED,
            right,
            MOON,
            self.color,
        )?;
        self.assistant_frame_open.set(false);
        self.assistant_line_open.set(false);
        self.assistant_body_width.set(0);
        self.assistant_column.set(0);
        Ok(())
    }

    fn close_assistant_frame_with_reflection(
        &self,
        out: &mut impl Write,
        left: &str,
        right: &str,
    ) -> io::Result<()> {
        if !self.assistant_frame_open.get() {
            return Ok(());
        }
        self.finish_assistant_body(out)?;
        let width = self.assistant_body_width.get().saturating_add(6);
        if self.motion_enabled() {
            for frame in 0..8 {
                let phase = frame as f64 * 0.52;
                let line = assistant_bottom_line(width, left, right, Some(phase), self.color);
                write!(out, "\r{line}")?;
                out.flush()?;
                std::thread::sleep(Duration::from_millis(35));
            }
        }
        let line = assistant_bottom_line(width, left, right, None, self.color);
        write!(out, "\r{line}\r\n")?;
        self.reset_assistant_frame_state();
        Ok(())
    }

    fn finish_assistant_body(&self, out: &mut impl Write) -> io::Result<()> {
        if self.assistant_line_open.get() {
            write_frame_body_suffix(
                out,
                self.assistant_body_width.get().max(1),
                self.assistant_column.get(),
                MOON_BRIGHT,
                self.color,
            )?;
        }
        Ok(())
    }

    fn reset_assistant_frame_state(&self) {
        self.assistant_frame_open.set(false);
        self.assistant_line_open.set(false);
        self.assistant_body_width.set(0);
        self.assistant_column.set(0);
    }

    fn motion_enabled(&self) -> bool {
        self.interactive && std::env::var_os("TINA_NO_MOTION").is_none()
    }

    /// Full machine access is never enabled silently. The exact token is
    /// intentionally ASCII and is required from a real terminal before the
    /// model or GPU runtime is constructed.
    pub(crate) fn confirm_full_access(&self, workspace: &Path) -> io::Result<bool> {
        if !self.interactive || workspace.as_os_str().is_empty() {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let warning = if self.language.is_chinese() {
            format!(
                "警告：--agent-full-access 将让 Tina 获得整台机器的文件读写、命令执行和网络访问，不再限制在 workspace。模型生成的命令可能删除数据、修改系统或访问外部服务；MCP 仍保持关闭。Agent 锚点 workspace：{}\r\n若你确认风险，请在下一行只输入 FULL_ACCESS： ",
                workspace.display()
            )
        } else {
            format!(
                "WARNING: --agent-full-access gives Tina machine-wide file read/write, command execution, and network access instead of limiting access to the workspace. Model-generated commands can delete data, modify the system, or contact external services; MCP remains disabled. Agent anchor workspace: {}\r\nType FULL_ACCESS on the next line to continue: ",
                workspace.display()
            )
        };
        write_styled(&mut out, MUTED, self.color, &warning)?;
        out.flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let confirmed = answer.trim_end_matches(['\r', '\n']) == "FULL_ACCESS";
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n整机访问未启用。\r\n"
            } else {
                "\r\nFull machine access was not enabled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    /// Require a real terminal response before an exact, stale-read guarded
    /// patch may mutate an existing workspace file.
    pub(crate) fn confirm_apply_patch(
        &self,
        confirmation: ApplyPatchConfirmation<'_>,
    ) -> io::Result<bool> {
        let ApplyPatchConfirmation {
            target,
            expected_sha256,
            edit_count,
            removed_bytes,
            added_bytes,
            preview,
            preview_truncated,
        } = confirmation;
        if !self.interactive
            || target.is_empty()
            || expected_sha256.len() != 64
            || !expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || edit_count == 0
            || preview.is_empty()
        {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let safe_target = terminal_safe_line(target);
        let title = if preview_truncated {
            format!("edit · {safe_target} · partial")
        } else {
            format!("edit · {safe_target}")
        };
        write_edit_preview(
            &mut out,
            composer_width(),
            &title,
            preview,
            &format!(" +{added_bytes} / -{removed_bytes} bytes "),
            &format!(" base {}... ", &expected_sha256[..12]),
            self.color,
        )?;
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        let prompt = if self.language.is_chinese() {
            format!(
                "对 {safe_target:?} 应用 {edit_count} 个精确修改（-{removed_bytes}/+{added_bytes} 字节，基线 {}...）？请输入 APPLY 确认： ",
                &expected_sha256[..12]
            )
        } else {
            format!(
                "Apply {edit_count} exact edit(s) to {safe_target:?} (-{removed_bytes}/+{added_bytes} bytes, base {}...)? Type APPLY to confirm: ",
                &expected_sha256[..12]
            )
        };
        write_styled(&mut out, MUTED, self.color, &prompt)?;
        out.flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let confirmed = apply_patch_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n修改已取消。\r\n"
            } else {
                "\r\nPatch cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    pub(crate) fn confirm_write_file(
        &self,
        confirmation: WriteFileConfirmation<'_>,
    ) -> io::Result<bool> {
        let WriteFileConfirmation {
            target,
            create,
            expected_sha256,
            content_bytes,
            content_sha256,
            preview,
            preview_truncated,
        } = confirmation;
        if !self.interactive
            || target.trim().is_empty()
            || content_sha256.len() != 64
            || !content_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || expected_sha256.is_some_and(|digest| {
                digest.len() != 64
                    || !digest
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let safe_target = terminal_safe_line(target);
        let action = if create { "create" } else { "write" };
        let title = if preview_truncated {
            format!("{action} · {safe_target} · partial")
        } else {
            format!("{action} · {safe_target}")
        };
        write_edit_preview(
            &mut out,
            composer_width(),
            &title,
            preview,
            &format!(" {content_bytes} bytes "),
            &format!(" sha {}... ", &content_sha256[..12]),
            self.color,
        )?;
        write_block_gap(&mut out, UI_BLOCK_GAP_LINES)?;
        let mode = if create {
            if self.language.is_chinese() {
                "创建"
            } else {
                "create"
            }
        } else if self.language.is_chinese() {
            "改写"
        } else {
            "replace"
        };
        let base = expected_sha256
            .map(|digest| format!(", base {}...", &digest[..12]))
            .unwrap_or_default();
        let prompt = if self.language.is_chinese() {
            format!(
                "{mode} {safe_target:?}（{content_bytes} 字节，内容 SHA-256 {}...{base}）？请输入 APPLY 确认： ",
                &content_sha256[..12]
            )
        } else {
            format!(
                "{mode} {safe_target:?} ({content_bytes} bytes, content SHA-256 {}...{base})? Type APPLY to confirm: ",
                &content_sha256[..12]
            )
        };
        write_styled(&mut out, MUTED, self.color, &prompt)?;
        out.flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let confirmed = apply_patch_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n文件写入已取消。\r\n"
            } else {
                "\r\nFile write cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    pub(crate) fn confirm_workspace_mutation(
        &self,
        operation: &str,
        detail: &str,
    ) -> io::Result<bool> {
        if !self.interactive || operation.trim().is_empty() || detail.trim().is_empty() {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let prompt = if self.language.is_chinese() {
            format!("执行 workspace {operation}：{detail}？请输入 APPLY 确认： ")
        } else {
            format!("Run workspace {operation}: {detail}? Type APPLY to confirm: ")
        };
        write_styled(&mut out, MUTED, self.color, &prompt)?;
        out.flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let confirmed = apply_patch_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n操作已取消。\r\n"
            } else {
                "\r\nMutation cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    /// Validation commands are a separate approval class from file mutation.
    /// Full Access does not bypass this gate: a build script or test can still
    /// execute arbitrary code with the host account's environment and network.
    pub(crate) fn confirm_workspace_exec(&self, detail: &str) -> io::Result<bool> {
        if !self.interactive || detail.trim().is_empty() {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let prompt = if self.language.is_chinese() {
            "执行受控 workspace 验证命令（不是 OS 沙箱；构建脚本/测试继承宿主权限）：\r\n"
        } else {
            "Run bounded workspace validation (not an OS sandbox; build scripts/tests inherit host permissions):\r\n"
        };
        write_styled(&mut out, MUTED, self.color, prompt)?;
        write_safe_assistant_piece(&mut out, detail)?;
        let suffix = if self.language.is_chinese() {
            "\r\n请输入 EXECUTE 确认： "
        } else {
            "\r\nType EXECUTE to confirm: "
        };
        write_styled(&mut out, MUTED, self.color, suffix)?;
        out.flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let confirmed = run_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n验证命令已取消。\r\n"
            } else {
                "\r\nValidation command cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    /// Require an explicit confirmation before a tool deletes or replaces a path.
    ///
    /// This is intentionally fail-closed: only a line whose content is exactly
    /// `DELETE` (apart from the line ending) authorizes the operation. Model
    /// output, `y`, an empty answer, and whitespace-padded answers are rejected.
    pub(crate) fn confirm_delete(&self, target: &str) -> io::Result<bool> {
        if !self.interactive || target.trim().is_empty() {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let prompt = if self.language.is_chinese() {
            format!("删除 {target:?}？请输入 DELETE 确认： ")
        } else {
            format!("Delete {target:?}? Type DELETE to confirm: ")
        };
        write_styled(&mut out, MUTED, self.color, &prompt)?;
        out.flush()?;
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        let confirmed = delete_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n删除已取消。\r\n"
            } else {
                "\r\nDeletion cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }

    /// Existing exports are never replaced implicitly. This prompt is only
    /// available on a real terminal; piped/noninteractive sessions fail closed.
    pub(crate) fn confirm_overwrite(&self, target: &str) -> io::Result<bool> {
        if !self.interactive || target.trim().is_empty() {
            return Ok(false);
        }
        let mut out = io::stdout().lock();
        let prompt = if self.language.is_chinese() {
            format!("导出目标已存在 {target:?}。覆盖它？请输入 OVERWRITE 确认： ")
        } else {
            format!("Export target already exists at {target:?}. Type OVERWRITE to confirm: ")
        };
        write_styled(&mut out, MUTED, self.color, &prompt)?;
        out.flush()?;
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        let confirmed = overwrite_confirmation_is_exact(&answer);
        if !confirmed {
            let message = if self.language.is_chinese() {
                "\r\n导出已取消。\r\n"
            } else {
                "\r\nExport cancelled.\r\n"
            };
            write_styled(&mut out, MUTED, self.color, message)?;
        }
        Ok(confirmed)
    }
}

/// Testable deletion boundary shared by future file-mutating tools and plugins.
/// A terminal line ending is transport framing, not part of the confirmation.
pub(crate) fn delete_confirmation_is_exact(answer: &str) -> bool {
    answer.trim_end_matches(['\r', '\n']) == "DELETE"
}

/// The model can propose a patch, but only the user's exact terminal response
/// can authorize it. Whitespace is deliberately not normalized.
pub(crate) fn apply_patch_confirmation_is_exact(answer: &str) -> bool {
    answer.trim_end_matches(['\r', '\n']) == "APPLY"
}

/// A validation command is authorized only by a line containing exactly
/// `EXECUTE`; whitespace padding and model text are never accepted.
pub(crate) fn run_confirmation_is_exact(answer: &str) -> bool {
    answer.trim_end_matches(['\r', '\n']) == "EXECUTE"
}

pub(crate) fn overwrite_confirmation_is_exact(answer: &str) -> bool {
    answer.trim_end_matches(['\r', '\n']) == "OVERWRITE"
}

struct RawInputGuard {
    enabled: bool,
}

pub(crate) struct GenerationGuard {
    enabled: bool,
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        if self.enabled {
            let _ = terminal::disable_raw_mode();
            let mut out = io::stdout().lock();
            let _ = out.flush();
            self.enabled = false;
        }
    }
}

impl RawInputGuard {
    fn enable() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self { enabled: true })
    }
}

impl Drop for RawInputGuard {
    fn drop(&mut self) {
        if self.enabled {
            let _ = terminal::disable_raw_mode();
            let mut out = io::stdout().lock();
            let _ = execute!(out, DisableBracketedPaste);
            let _ = out.flush();
            self.enabled = false;
        }
    }
}

fn composer_interrupt_requests_exit(editor: &mut EditorBuffer) -> bool {
    if editor.text.is_empty() {
        true
    } else {
        editor.clear();
        false
    }
}

fn paste_rejected_text(language: UiLanguage, pasted_bytes: usize) -> String {
    if language.is_chinese() {
        format!(
            "粘贴内容为 {pasted_bytes} 字节，加入草稿后将超过 {} 字节上限；本次粘贴未写入。",
            context_budget::MAX_INTERACTIVE_MESSAGE_BYTES,
        )
    } else {
        format!(
            "Paste is {pasted_bytes} bytes and would exceed the {}-byte message limit; nothing was inserted.",
            context_budget::MAX_INTERACTIVE_MESSAGE_BYTES,
        )
    }
}

fn complete_slash_command(editor: &mut EditorBuffer) -> bool {
    if editor.cursor != editor.text.len()
        || !editor.text.starts_with('/')
        || editor.text.starts_with("//")
        || editor.text.chars().any(char::is_whitespace)
    {
        return false;
    }
    let candidates = SLASH_COMPLETIONS
        .iter()
        .copied()
        .filter(|command| command.starts_with(&editor.text))
        .collect::<Vec<_>>();
    let Some(first) = candidates.first().copied() else {
        return false;
    };
    let completion = candidates.iter().skip(1).fold(first, |prefix, candidate| {
        common_utf8_prefix(prefix, candidate)
    });
    if completion.len() <= editor.text.len() {
        return true;
    }
    editor.text.clear();
    editor.text.push_str(completion);
    editor.cursor = editor.text.len();
    editor.preferred_column = None;
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlashPopupItem {
    name: &'static str,
    description: &'static str,
}

fn slash_command_token(text: &str) -> Option<&str> {
    let first_line = text.lines().next().unwrap_or_default();
    if !first_line.starts_with('/') || first_line.starts_with("//") {
        return None;
    }
    Some(first_line.split_whitespace().next().unwrap_or(first_line))
}

fn slash_popup_candidates(text: &str, language: UiLanguage) -> Vec<SlashPopupItem> {
    let Some(prefix) = slash_command_token(text) else {
        return Vec::new();
    };
    let prefix_lower = prefix.to_lowercase();
    let mut exact = Vec::new();
    let mut prefixed = Vec::new();
    for spec in SLASH_COMMAND_SPECS {
        let name = spec.display_name(prefix);
        let name_lower = name.to_lowercase();
        let item = SlashPopupItem {
            name,
            description: spec.description(language),
        };
        if name_lower == prefix_lower {
            exact.push(item);
        } else if name_lower.starts_with(&prefix_lower) {
            prefixed.push(item);
        }
    }
    exact.extend(prefixed);
    exact
}

fn slash_popup_selectable(editor: &EditorBuffer) -> bool {
    editor.cursor == editor.text.len()
        && slash_command_token(&editor.text).is_some_and(|token| token.len() == editor.text.len())
}

fn replace_with_slash_candidate(
    editor: &mut EditorBuffer,
    language: UiLanguage,
    selected: usize,
) -> bool {
    if !slash_popup_selectable(editor) {
        return false;
    }
    let candidates = slash_popup_candidates(&editor.text, language);
    let Some(candidate) = candidates.get(selected.min(candidates.len().saturating_sub(1))) else {
        return false;
    };
    editor.text.clear();
    editor.text.push_str(candidate.name);
    editor.cursor = editor.text.len();
    editor.preferred_column = None;
    true
}

fn move_slash_selection(selected: &mut usize, candidate_count: usize, down: bool) {
    if candidate_count == 0 {
        *selected = 0;
    } else if down {
        *selected = (*selected + 1) % candidate_count;
    } else if *selected == 0 {
        *selected = candidate_count - 1;
    } else {
        *selected -= 1;
    }
}

fn common_utf8_prefix<'a>(left: &'a str, right: &str) -> &'a str {
    let mut end = 0;
    for ((left_offset, left_character), right_character) in left.char_indices().zip(right.chars()) {
        if left_character != right_character {
            break;
        }
        end = left_offset + left_character.len_utf8();
    }
    &left[..end]
}

fn handle_editor_key(
    editor: &mut EditorBuffer,
    key: KeyEvent,
    width: usize,
    history: &[String],
    history_cursor: &mut Option<usize>,
) {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    match key.code {
        KeyCode::Char(ch) if control && ch == 'a' => editor.move_home(),
        KeyCode::Char(ch) if control && ch == 'e' => editor.move_end(),
        KeyCode::Char(ch) if control && ch == 'u' => editor.delete_to_home(),
        KeyCode::Char(ch) if control && ch == 'k' => editor.delete_to_end(),
        KeyCode::Char(ch) if control && ch == 'w' => editor.delete_word_backwards(),
        KeyCode::Char(ch) if alt && ch == 'b' => editor.move_horizontal(false),
        KeyCode::Char(ch) if alt && ch == 'f' => editor.move_horizontal(true),
        KeyCode::Char(ch) if alt && (ch == 'd' || ch == '\u{7f}') => editor.delete(),
        KeyCode::Char(_) if control || alt => {}
        KeyCode::Char(ch) => {
            editor.insert(&ch.to_string());
            *history_cursor = None;
        }
        KeyCode::Backspace => {
            if alt {
                editor.delete_word_backwards();
            } else {
                editor.backspace();
            }
            *history_cursor = None;
        }
        KeyCode::Delete => editor.delete(),
        KeyCode::Left => editor.move_horizontal(false),
        KeyCode::Right => editor.move_horizontal(true),
        KeyCode::Home => editor.move_home(),
        KeyCode::End => editor.move_end(),
        KeyCode::Up => {
            let lines = visual_lines(&editor.text, width);
            if line_for_cursor(&lines, editor.cursor) > 0 {
                editor.move_vertical(width, -1);
            } else if !history.is_empty() {
                let index = history_cursor.unwrap_or(history.len()).saturating_sub(1);
                editor.text = history[index].clone();
                editor.cursor = editor.text.len();
                *history_cursor = Some(index);
            }
        }
        KeyCode::Down => {
            let lines = visual_lines(&editor.text, width);
            if line_for_cursor(&lines, editor.cursor) + 1 < lines.len() {
                editor.move_vertical(width, 1);
            } else if let Some(index) = history_cursor {
                if *index + 1 < history.len() {
                    *index += 1;
                    editor.text = history[*index].clone();
                    editor.cursor = editor.text.len();
                } else {
                    editor.text.clear();
                    editor.cursor = 0;
                    *history_cursor = None;
                }
            }
        }
        KeyCode::Tab => {
            if !complete_slash_command(editor) {
                editor.insert("    ");
            }
            *history_cursor = None;
        }
        _ => {}
    }
}

fn write_frame_top(
    out: &mut impl Write,
    width: usize,
    label: &str,
    label_color: &str,
    color: bool,
) -> io::Result<()> {
    let width = width.max(MIN_BOX_WIDTH);
    let label = format!(" {} ", fit_display_width(label, width.saturating_sub(5)));
    let fill = width.saturating_sub(3 + display_width(&label));
    write_styled(out, MOON_BRIGHT, color, "╭─")?;
    write_styled(out, label_color, color, &label)?;
    write_styled(
        out,
        MOON_BRIGHT,
        color,
        &format!("{}╮\r\n", "─".repeat(fill)),
    )
}

fn write_frame_bottom(
    out: &mut impl Write,
    width: usize,
    left: &str,
    left_color: &str,
    right: &str,
    right_color: &str,
    color: bool,
) -> io::Result<()> {
    let width = width.max(MIN_BOX_WIDTH);
    let available = width.saturating_sub(3);
    let right = fit_display_width(right, available);
    let left = fit_display_width(left, available.saturating_sub(display_width(&right)));
    let fill = available.saturating_sub(display_width(&left) + display_width(&right));
    write_styled(out, MOON_BRIGHT, color, "╰─")?;
    write_styled(out, left_color, color, &left)?;
    write_styled(out, MOON_BRIGHT, color, &"─".repeat(fill))?;
    write_styled(out, right_color, color, &right)?;
    write_styled(out, MOON_BRIGHT, color, "╯\r\n")
}

fn write_composer_bottom(
    out: &mut impl Write,
    width: usize,
    left: &str,
    right: &str,
    color: bool,
) -> io::Result<()> {
    let width = width.max(MIN_BOX_WIDTH);
    let available = width.saturating_sub(2);
    let right = fit_display_width(right, available);
    let left = fit_display_width(left, available.saturating_sub(display_width(&right)));
    let fill = available.saturating_sub(display_width(&left) + display_width(&right));
    write_styled(out, MOON_BRIGHT, color, "╰")?;
    write_styled(out, MUTED, color, &left)?;
    write_styled(out, MOON_BRIGHT, color, &"─".repeat(fill))?;
    write_styled(out, MOON, color, &right)?;
    write_styled(out, MOON_BRIGHT, color, "╯")
}

fn tide_line(width: usize, phase: f64) -> String {
    const GLYPHS: [&str; 5] = ["·", "·", ".", "~", "≈"];
    let mut line = String::with_capacity(width);
    for x in 0..width {
        let wave = 0.5 + 0.5 * (((2.0 * std::f64::consts::PI * x as f64) / 18.0) - phase).sin();
        let index = (wave * (GLYPHS.len() - 1) as f64).round() as usize;
        line.push_str(GLYPHS[index.min(GLYPHS.len() - 1)]);
    }
    line
}

fn assistant_bottom_line(
    width: usize,
    left: &str,
    right: &str,
    phase: Option<f64>,
    color: bool,
) -> String {
    let width = width.max(MIN_BOX_WIDTH);
    let available = width.saturating_sub(4);
    let right = fit_display_width(right, available);
    let left = fit_display_width(left, available.saturating_sub(display_width(&right)));
    let flow_width = available.saturating_sub(display_width(&left) + display_width(&right));
    let flow = phase
        .map(|phase| tide_line(flow_width, phase))
        .unwrap_or_else(|| "─".repeat(flow_width));
    format!(
        "{}{}{}{}{}",
        styled(MOON_BRIGHT, color, "╰─"),
        styled(MUTED, color, &left),
        styled(
            if phase.is_some() { TIDE } else { MOON_BRIGHT },
            color,
            &flow
        ),
        styled(MOON, color, &right),
        styled(MOON_BRIGHT, color, "─╯")
    )
}

fn write_frame_body_prefix(out: &mut impl Write, color: bool) -> io::Result<()> {
    write_styled(out, MOON_BRIGHT, color, "│  ")
}

fn write_frame_body_suffix(
    out: &mut impl Write,
    body_width: usize,
    column: usize,
    body_color: &str,
    color: bool,
) -> io::Result<()> {
    write_styled(
        out,
        body_color,
        color,
        &" ".repeat(body_width.saturating_sub(column)),
    )?;
    write_styled(out, MOON_BRIGHT, color, "  │\r\n")
}

fn safe_terminal_fragment(character: char, column: usize) -> String {
    if character == '\t' {
        return " ".repeat(4 - (column % 4));
    }
    if character.is_control() {
        format!("\\u{{{:04x}}}", character as u32)
    } else {
        character.to_string()
    }
}

fn flush_frame_body_run(
    out: &mut impl Write,
    run: &mut String,
    body_color: &str,
    color: bool,
) -> io::Result<()> {
    if run.is_empty() {
        return Ok(());
    }
    write_styled(out, body_color, color, run)?;
    run.clear();
    Ok(())
}

fn write_streamed_frame_piece(
    out: &mut impl Write,
    piece: &str,
    body_width: usize,
    column: &mut usize,
    line_open: &mut bool,
    body_color: &str,
    color: bool,
) -> io::Result<()> {
    let mut run = String::new();
    for source_character in piece.chars() {
        if source_character == '\n' {
            flush_frame_body_run(out, &mut run, body_color, color)?;
            if !*line_open {
                write_frame_body_prefix(out, color)?;
            }
            write_frame_body_suffix(out, body_width, *column, body_color, color)?;
            *line_open = false;
            *column = 0;
            continue;
        }
        let fragment = safe_terminal_fragment(source_character, *column);
        for character in fragment.chars() {
            let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
            if *line_open && *column > 0 && *column + character_width > body_width {
                flush_frame_body_run(out, &mut run, body_color, color)?;
                write_frame_body_suffix(out, body_width, *column, body_color, color)?;
                *line_open = false;
                *column = 0;
            }
            if !*line_open {
                flush_frame_body_run(out, &mut run, body_color, color)?;
                write_frame_body_prefix(out, color)?;
                *line_open = true;
            }
            run.push(character);
            *column += character_width;
        }
    }
    flush_frame_body_run(out, &mut run, body_color, color)
}

fn write_framed_text(
    out: &mut impl Write,
    text: &str,
    body_width: usize,
    body_color: &str,
    color: bool,
) -> io::Result<()> {
    let mut line_open = true;
    let mut column = 0;
    write_frame_body_prefix(out, color)?;
    write_streamed_frame_piece(
        out,
        text,
        body_width,
        &mut column,
        &mut line_open,
        body_color,
        color,
    )?;
    if line_open {
        write_frame_body_suffix(out, body_width, column, body_color, color)?;
    }
    Ok(())
}

fn terminal_safe_line(value: &str) -> String {
    let mut safe = String::new();
    let mut column = 0;
    for character in value.chars() {
        if matches!(character, '\r' | '\n') {
            break;
        }
        let fragment = safe_terminal_fragment(character, column);
        column += display_width(&fragment);
        safe.push_str(&fragment);
    }
    safe
}

fn write_edit_preview(
    out: &mut impl Write,
    width: usize,
    title: &str,
    preview: &str,
    summary: &str,
    identity: &str,
    color: bool,
) -> io::Result<()> {
    let width = width.max(MIN_BOX_WIDTH);
    write_frame_top(out, width, &terminal_safe_line(title), ACCENT, color)?;
    let body_width = width.saturating_sub(4).max(1);
    let mut wrote_line = false;
    for line in preview.lines() {
        let safe = terminal_safe_line(line);
        let content = fit_display_width(&safe, body_width);
        let line_color = match safe.trim_start().chars().next() {
            Some('+') => DIFF_GREEN,
            Some('-') => DIFF_RED,
            _ => MUTED,
        };
        write_styled(out, MOON_BRIGHT, color, "│ ")?;
        write_styled(out, line_color, color, &content)?;
        write_styled(
            out,
            MOON_BRIGHT,
            color,
            &format!(
                "{} │\r\n",
                " ".repeat(body_width.saturating_sub(display_width(&content)))
            ),
        )?;
        wrote_line = true;
    }
    if !wrote_line {
        write_styled(
            out,
            MOON_BRIGHT,
            color,
            &format!("│ {} │\r\n", " ".repeat(body_width)),
        )?;
    }
    write_frame_bottom(out, width, summary, MUTED, identity, MOON, color)
}

fn redraw_composer(
    out: &mut impl Write,
    editor: &EditorBuffer,
    context: ComposerContext<'_>,
    slash_selection: usize,
    show_placeholder: bool,
    previous: &mut Option<RenderedComposer>,
) -> io::Result<()> {
    if let Some(previous) = *previous {
        move_to_composer_top(out, previous)?;
        clear_rows(out, previous.height)?;
    }

    let inner_width = composer_inner_width();
    let lines = visual_lines(&editor.text, inner_width);
    let cursor_line = line_for_cursor(&lines, editor.cursor);
    let first_line = cursor_line.saturating_sub(MAX_EDITOR_ROWS - 1);
    let last_line = (first_line + MAX_EDITOR_ROWS).min(lines.len());
    let visible = &lines[first_line..last_line];
    let cursor_row = 1 + cursor_line.saturating_sub(first_line);
    let cursor_col = 4 + if editor.text.is_empty() {
        0
    } else {
        let line = lines[cursor_line];
        display_width(&terminal_safe_line(&editor.text[line.start..editor.cursor]))
    };
    let height = visible.len() + 2;
    let width = composer_width();
    let placeholder = if context.language.is_chinese() {
        format!("向 {} 提问...", context.assistant_id)
    } else {
        format!("Ask {}...", context.assistant_id)
    };

    queue!(out, Print("\x1b[?2026h"))?;
    write_frame_top(out, width, "User", MOON_BRIGHT, context.color)?;
    for (index, line) in visible.iter().enumerate() {
        let raw_content = if editor.text.is_empty() && show_placeholder {
            fit_display_width(&placeholder, inner_width)
        } else {
            fit_display_width(
                &terminal_safe_line(&editor.text[line.start..line.end]),
                inner_width,
            )
        };
        let is_first_line = first_line + index == 0;
        let prefix = if is_first_line { "› " } else { "  " };
        let visible_width = display_width(&raw_content);
        write_styled(out, MOON_BRIGHT, context.color, "│ ")?;
        write_styled(out, MOON_BRIGHT, context.color, prefix)?;
        write_styled(
            out,
            if editor.text.is_empty() && show_placeholder {
                MUTED
            } else {
                MOON
            },
            context.color,
            &raw_content,
        )?;
        write_styled(
            out,
            MOON_BRIGHT,
            context.color,
            &format!(
                "{} │",
                " ".repeat(inner_width.saturating_sub(visible_width))
            ),
        )?;
        if index + 1 < visible.len() {
            write!(out, "\r\n")?;
        }
    }
    write!(out, "\r\n")?;
    let label = format!(
        " {} / {} ",
        compact_count(context.context_positions),
        compact_count(context.capacity)
    );
    let agent = if context.agent_enabled { "on" } else { "off" };
    let state = if context.language.is_chinese() {
        format!(" Agent {agent} · 思考 {} ", context.thinking)
    } else {
        format!(" Agent {agent} · think {} ", context.thinking)
    };
    write_composer_bottom(out, width, &state, &label, context.color)?;
    let popup = slash_popup_candidates(&editor.text, context.language);
    let popup_rows = render_slash_popup(out, &popup, slash_selection, width, context.color)?;
    let total_height = height + popup_rows;
    queue!(
        out,
        Print("\x1b[?2026l"),
        MoveUp((total_height - 1 - cursor_row) as u16),
        MoveToColumn(cursor_col as u16)
    )?;
    out.flush()?;
    *previous = Some(RenderedComposer {
        height: total_height,
        cursor_row,
    });
    Ok(())
}

fn render_slash_popup(
    out: &mut impl Write,
    candidates: &[SlashPopupItem],
    selected: usize,
    width: usize,
    color: bool,
) -> io::Result<usize> {
    if candidates.is_empty() {
        return Ok(0);
    }

    let visible_count = candidates.len().min(MAX_COMMAND_POPUP_ROWS);
    let selected = selected.min(candidates.len() - 1);
    let start = selected
        .saturating_sub(visible_count / 2)
        .min(candidates.len() - visible_count);
    let visible = &candidates[start..start + visible_count];
    let name_width = visible
        .iter()
        .map(|item| display_width(item.name))
        .max()
        .unwrap_or(1)
        .min((width / 2).max(1));
    let description_width = width.saturating_sub(6 + name_width);

    for (offset, item) in visible.iter().enumerate() {
        let index = start + offset;
        let name = fit_display_width(item.name, name_width);
        let name_padding = " ".repeat(name_width.saturating_sub(display_width(&name)));
        let description = fit_display_width(item.description, description_width);
        let marker = if index == selected { "›" } else { " " };
        let mut line = format!("  {marker} {name}{name_padding}  {description}");
        let padding = width.saturating_sub(display_width(&line));
        line.push_str(&" ".repeat(padding));
        write!(out, "\r\n")?;
        if index == selected {
            write_styled(out, POPUP_SELECTED, color, &line)?;
        } else {
            let name_part = format!("    {name}{name_padding}  ");
            write_styled(out, MOON, color, &name_part)?;
            write_styled(
                out,
                MUTED,
                color,
                &format!("{description}{}", " ".repeat(padding)),
            )?;
        }
    }
    Ok(visible_count)
}

fn clear_composer(out: &mut impl Write, previous: Option<RenderedComposer>) -> io::Result<()> {
    let Some(previous) = previous else {
        return Ok(());
    };
    move_to_composer_top(out, previous)?;
    clear_rows(out, previous.height)?;
    out.flush()
}

fn move_to_composer_top(out: &mut impl Write, previous: RenderedComposer) -> io::Result<()> {
    queue!(out, MoveToColumn(0))?;
    if previous.cursor_row > 0 {
        queue!(out, MoveUp(previous.cursor_row as u16))?;
    }
    Ok(())
}

fn clear_rows(out: &mut impl Write, height: usize) -> io::Result<()> {
    for row in 0..height {
        queue!(out, Clear(ClearType::CurrentLine), MoveToColumn(0))?;
        if row + 1 < height {
            queue!(out, MoveDown(1), MoveToColumn(0))?;
        }
    }
    if height > 1 {
        queue!(out, MoveUp((height - 1) as u16), MoveToColumn(0))?;
    }
    Ok(())
}

fn visual_lines(text: &str, width: usize) -> Vec<VisualLine> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut start = 0;
    let mut line_width = 0;
    for (offset, ch) in text.char_indices() {
        if ch == '\n' {
            lines.push(VisualLine { start, end: offset });
            start = offset + ch.len_utf8();
            line_width = 0;
            continue;
        }
        let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if line_width + char_width > width && offset > start {
            lines.push(VisualLine { start, end: offset });
            start = offset;
            line_width = 0;
        }
        line_width += char_width;
    }
    lines.push(VisualLine {
        start,
        end: text.len(),
    });
    lines
}

fn line_for_cursor(lines: &[VisualLine], cursor: usize) -> usize {
    lines
        .iter()
        .enumerate()
        .find(|(index, line)| {
            cursor < line.end
                || (cursor == line.end
                    && (*index + 1 == lines.len() || lines[*index + 1].start != cursor))
        })
        .map(|(index, _)| index)
        .unwrap_or_else(|| lines.len().saturating_sub(1))
}

fn byte_at_display_column(text: &str, target: usize) -> usize {
    let mut width = 0;
    for (offset, ch) in text.char_indices() {
        let next = width + UnicodeWidthChar::width(ch).unwrap_or(0);
        if next > target {
            return offset;
        }
        width = next;
    }
    text.len()
}

fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn fit_display_width(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if display_width(text) <= width {
        return text.to_string();
    }
    if width < 3 {
        return ".".repeat(width);
    }
    let ellipsis = "...";
    let target = width.saturating_sub(display_width(ellipsis));
    let mut fitted = String::new();
    let mut used = 0;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used + character_width > target {
            break;
        }
        fitted.push(character);
        used += character_width;
    }
    fitted.push_str(ellipsis);
    fitted
}

fn startup_frame_lines(title: &str, badge: &str, rows: &[String], width: usize) -> Vec<String> {
    let width = width.max(MIN_BOX_WIDTH);
    let border_width = width.saturating_sub(2);
    let right = format!(" {badge} ─");
    let title_width = border_width
        .saturating_sub(display_width(&right) + 3)
        .max(1);
    let title = fit_display_width(title, title_width);
    let left = format!("─ {title} ");
    let fill = border_width.saturating_sub(display_width(&left) + display_width(&right));
    let mut lines = vec![format!("╭{left}{}{right}╮", "─".repeat(fill))];
    let content_width = width.saturating_sub(4);
    for row in rows {
        let row = fit_display_width(row, content_width);
        let remaining = content_width.saturating_sub(display_width(&row));
        let left_padding = remaining / 2;
        let right_padding = remaining.saturating_sub(left_padding);
        lines.push(format!(
            "│ {}{}{} │",
            " ".repeat(left_padding),
            row,
            " ".repeat(right_padding)
        ));
    }
    lines.push(format!("╰{}╯", "─".repeat(border_width)));
    lines
}

fn terminal_width() -> usize {
    terminal::size()
        .map(|(width, _)| usize::from(width))
        .unwrap_or(80)
        .max(20)
}

/// Keep the startup frame deterministic and terminal-safe. The full detected
/// environment is still available through `/environment`; the first screen
/// only needs a short, trusted label that cannot emit control sequences.
fn startup_os_label() -> &'static str {
    match std::env::consts::OS {
        "windows" => "Windows",
        "linux" => "Linux",
        "macos" => "macOS",
        _ => "Unix",
    }
}

fn startup_shell_label() -> &'static str {
    let configured = std::env::var("TINA_SHELL")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match configured.as_str() {
        "cmd" => "cmd",
        "bash" => "bash",
        "zsh" => "zsh",
        "fish" => "fish",
        "sh" => "sh",
        "powershell" | "pwsh" => "PowerShell",
        _ if cfg!(windows) => "PowerShell",
        _ => "shell",
    }
}

fn composer_width() -> usize {
    terminal_width()
        .saturating_sub(2)
        .clamp(MIN_BOX_WIDTH, MAX_BOX_WIDTH)
}

fn composer_inner_width() -> usize {
    composer_width().saturating_sub(6).max(1)
}

fn compact_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn nonempty_or<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    if value.trim().is_empty() {
        fallback
    } else {
        value
    }
}

fn single_line_excerpt(value: &str, max_chars: usize) -> String {
    let flattened = value
        .chars()
        .map(|character| {
            if matches!(character, '\r' | '\n' | '\t') {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let mut excerpt = flattened.chars().take(max_chars).collect::<String>();
    if flattened.chars().count() > max_chars {
        excerpt.push_str("...");
    }
    excerpt
}

fn is_exit_command(message: &str) -> bool {
    matches!(message.trim(), "/exit" | "/quit" | "/退出" | ":q")
}

fn write_block_gap(out: &mut impl Write, blank_lines: usize) -> io::Result<()> {
    for _ in 0..blank_lines {
        write!(out, "\r\n")?;
    }
    Ok(())
}

fn command_line(message: &str) -> Option<String> {
    let command = message.trim();
    (command.starts_with('/') && !command.starts_with("//") && !is_exit_command(command))
        .then(|| command.to_string())
}

fn unescape_literal_prompt(message: String) -> String {
    let leading = message.len().saturating_sub(message.trim_start().len());
    if message[leading..].starts_with("//") {
        let mut unescaped = message;
        unescaped.remove(leading);
        unescaped
    } else {
        message
    }
}

fn write_safe_assistant_piece(out: &mut impl Write, piece: &str) -> io::Result<()> {
    let needs_escape = piece
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\t'));
    if !needs_escape {
        return out.write_all(piece.as_bytes());
    }
    for character in piece.chars() {
        if character.is_control() && !matches!(character, '\n' | '\t') {
            write!(out, "\\u{{{:04x}}}", character as u32)?;
        } else {
            write!(out, "{character}")?;
        }
    }
    Ok(())
}

fn write_bright_assistant_piece(out: &mut impl Write, piece: &str, color: bool) -> io::Result<()> {
    if color {
        out.write_all(MOON_BRIGHT.as_bytes())?;
    }
    write_safe_assistant_piece(out, piece)?;
    if color {
        out.write_all(RESET.as_bytes())?;
    }
    Ok(())
}

fn command_value<'a>(command: &'a str, names: &[&str]) -> Option<Option<&'a str>> {
    let command = command.trim();
    for name in names {
        if command == *name {
            return Some(None);
        }
        if let Some(rest) = command.strip_prefix(name) {
            if rest
                .chars()
                .next()
                .is_some_and(|character| character.is_whitespace())
            {
                return Some(Some(rest.trim()));
            }
        }
    }
    None
}

pub(crate) fn assistant_id_command(command: &str) -> Option<AssistantIdCommand> {
    command_value(
        command,
        &[
            "/assistant",
            "/assistant-id",
            "/id",
            "/name",
            "/助手",
            "/名称",
        ],
    )
    .map(|value| match value {
        Some(value) => AssistantIdCommand::Set(value.to_string()),
        None => AssistantIdCommand::Show,
    })
}

pub(crate) fn language_command(command: &str) -> Option<LanguageCommand> {
    command_value(command, &["/language", "/lang", "/语言"]).map(|value| match value {
        Some(value) => LanguageCommand::Set(value.to_string()),
        None => LanguageCommand::Show,
    })
}

pub(crate) fn goal_command(command: &str) -> Option<Result<GoalCommand, String>> {
    let value = command_value(command, &["/goal", "/目标"])?;
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Some(Ok(GoalCommand::Show));
    };
    let mut parts = value.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(match verb.to_ascii_lowercase().as_str() {
        "status" | "show" | "查看" if argument.is_none() => Ok(GoalCommand::Show),
        "clear" | "reset" | "delete" | "清除" | "清空" if argument.is_none() => {
            Ok(GoalCommand::Clear)
        }
        "set" | "设置" => argument
            .filter(|value| !value.is_empty())
            .map(|value| GoalCommand::Set(value.to_string()))
            .ok_or_else(|| "Usage: /goal set <text>".to_string()),
        _ => Err("Usage: /goal [status|set <text>|clear]".to_string()),
    })
}

pub(crate) fn thinking_command(command: &str) -> Option<Result<ThinkingCommand, String>> {
    let value = command_value(command, &["/thinking", "/think", "/思考"])?;
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Some(Ok(ThinkingCommand::Show));
    };
    Some(ThinkingLevel::parse(value).map(ThinkingCommand::Set))
}

pub(crate) fn is_help_command(command: &str) -> bool {
    matches!(command.trim(), "/help" | "/commands" | "/帮助" | "/指令")
}

pub(crate) fn is_memory_command(command: &str) -> bool {
    matches!(command.trim(), "/memory" | "/记忆")
}

pub(crate) fn remember_command(command: &str) -> Option<Option<&str>> {
    command_value(command, &["/remember", "/记住", "/固定"])
}

pub(crate) fn project_memory_command(
    command: &str,
) -> Option<Result<ProjectMemoryCommand, String>> {
    if let Some(value) = command_value(command, &["/project-remember", "/项目记住"]) {
        return Some(required_project_memory_value(value).map(ProjectMemoryCommand::Remember));
    }
    if let Some(value) = command_value(command, &["/project-forget", "/项目忘记"]) {
        return Some(required_project_memory_value(value).map(ProjectMemoryCommand::Forget));
    }
    let value = command_value(command, &["/project-memory", "/项目记忆"])?;
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Some(Ok(ProjectMemoryCommand::Status));
    };
    let mut parts = value.splitn(2, char::is_whitespace);
    let verb = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(match verb {
        "status" | "show" | "list" | "状态" | "查看" | "列表" if argument.is_none() => {
            Ok(ProjectMemoryCommand::Status)
        }
        "remember" | "add" | "记住" | "添加" => {
            required_project_memory_value(argument).map(ProjectMemoryCommand::Remember)
        }
        "forget" | "remove" | "delete" | "忘记" | "删除" => {
            required_project_memory_value(argument).map(ProjectMemoryCommand::Forget)
        }
        _ => Err("invalid project-memory command".to_string()),
    })
}

fn required_project_memory_value(value: Option<&str>) -> Result<String, String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "project-memory command requires a value".to_string())
}

pub(crate) fn session_command(command: &str) -> Option<SessionCommand> {
    match command.trim() {
        "/agent" | "/代理" => Some(SessionCommand::Agent),
        "/context" | "/上下文" => Some(SessionCommand::Context),
        "/status" | "/状态" => Some(SessionCommand::Status),
        "/tools" | "/工具" => Some(SessionCommand::Tools),
        "/agent-restart" | "/agent_restart" | "/代理重启" | "/重启工具" => {
            Some(SessionCommand::AgentRestart)
        }
        "/environment" | "/env" | "/环境" | "/终端" => Some(SessionCommand::Environment),
        "/sessions" | "/会话" | "/会话列表" => Some(SessionCommand::Sessions),
        "/new" | "/reset" | "/clear" | "/新会话" | "/重置" | "/清空" => {
            Some(SessionCommand::New)
        }
        "/compact" | "/压缩" => Some(SessionCommand::Compact),
        "/retry" | "/重试" => Some(SessionCommand::Retry),
        _ => None,
    }
}

/// Parse the local search control without sending a slash command to the
/// model. An empty query is a status request; a non-empty query is still only
/// a request for the configured provider and never implies that a result
/// exists.
pub(crate) fn web_search_command(command: &str) -> Option<Result<Option<String>, String>> {
    let trimmed = command.trim();
    let query = if matches!(trimmed, "/search" | "/网络搜索") {
        None
    } else if let Some(value) = trimmed
        .strip_prefix("/search ")
        .or_else(|| trimmed.strip_prefix("/网络搜索 "))
    {
        Some(value.trim())
    } else {
        return None;
    };
    let Some(query) = query.filter(|value| !value.is_empty()) else {
        return Some(Ok(None));
    };
    if query.len() > 4_096 || query.chars().any(char::is_control) {
        return Some(Err(
            "search query must be 1..=4096 bytes and contain no control characters".to_string(),
        ));
    }
    Some(Ok(Some(query.to_string())))
}

/// Parse an in-process archive switch. A full lower-case SHA-256 id is
/// required so a typo cannot silently select a different conversation.
pub(crate) fn resume_command(command: &str) -> Option<Result<ResumeTarget, String>> {
    let value = command_value(command, &["/resume", "/恢复"])?;
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Some(Err(
            "Usage: /resume <latest|full 64-character conversation ID>".to_string(),
        ));
    };
    if value.eq_ignore_ascii_case("latest") || value == "最新" {
        return Some(Ok(ResumeTarget::Latest));
    }
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Some(Ok(ResumeTarget::Conversation(value.to_string())));
    }
    Some(Err(
        "resume target must be 'latest' or a full lower-case 64-character conversation ID"
            .to_string(),
    ))
}

const MAX_REVIEW_FOCUS_BYTES: usize = 8 * 1024;

/// Parse workspace-local, read-only commands without allowing an unknown
/// argument to fall through as a model prompt. `/diff` is deliberately
/// argument-free; `/review` accepts one bounded focus string.
pub(crate) fn workspace_command(command: &str) -> Option<Result<WorkspaceCommand, String>> {
    let command = command.trim();
    if let Some(value) = command_value(command, &["/diff", "/差异"]) {
        return Some(if value.is_none() {
            Ok(WorkspaceCommand::Diff)
        } else {
            Err("Usage: /diff (or /差异)".to_string())
        });
    }
    command_value(command, &["/review", "/审查"]).map(parse_review_focus)
}

fn parse_review_focus(value: Option<&str>) -> Result<WorkspaceCommand, String> {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(WorkspaceCommand::Review { focus: None });
    };
    if value.chars().any(char::is_control) {
        return Err("Review focus cannot contain control characters".to_string());
    }
    if value.len() > MAX_REVIEW_FOCUS_BYTES {
        return Err(format!(
            "Review focus is limited to {MAX_REVIEW_FOCUS_BYTES} UTF-8 bytes"
        ));
    }
    Ok(WorkspaceCommand::Review {
        focus: Some(value.to_string()),
    })
}

pub(crate) fn export_command(command: &str) -> Option<Result<ExportCommand, String>> {
    let value = command_value(command, &["/export", "/导出"])?;
    let Some(value) = value else {
        return Some(Err(
            "Usage: /export [jsonl|md] <path> (or /导出 [jsonl|md] <路径>)".to_string(),
        ));
    };
    let value = value.trim();
    if value.is_empty() {
        return Some(Err(
            "Usage: /export [jsonl|md] <path> (or /导出 [jsonl|md] <路径>)".to_string(),
        ));
    }
    let mut fields = value.splitn(2, char::is_whitespace);
    let first = fields.next().unwrap_or_default();
    let remainder = fields.next().map(str::trim).filter(|path| !path.is_empty());
    let (format, raw_path) = match first.to_ascii_lowercase().as_str() {
        "jsonl" => (Some(ExportFormat::Jsonl), remainder),
        "md" | "markdown" => (Some(ExportFormat::Markdown), remainder),
        _ => (None, Some(value)),
    };
    let Some(raw_path) = raw_path else {
        return Some(Err(
            "Usage: /export [jsonl|md] <path> (or /导出 [jsonl|md] <路径>)".to_string(),
        ));
    };
    let path = strip_matching_quotes(raw_path.trim());
    if path.is_empty() || path == "-" || path.chars().any(|character| character.is_control()) {
        return Some(Err(
            "Export path must be a non-empty regular file path, not '-'.".to_string(),
        ));
    }
    let format = format.unwrap_or_else(|| {
        match std::path::Path::new(path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref()
        {
            Some("md") | Some("markdown") => ExportFormat::Markdown,
            _ => ExportFormat::Jsonl,
        }
    });
    Some(Ok(ExportCommand {
        format,
        path: path.to_string(),
    }))
}

fn strip_matching_quotes(value: &str) -> &str {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\''))
    {
        &value[1..value.len() - 1]
    } else {
        value
    }
}

fn styled(code: &str, enabled: bool, value: &str) -> String {
    if enabled {
        format!("{code}{value}{RESET}")
    } else {
        value.to_string()
    }
}

fn write_styled(out: &mut impl Write, code: &str, enabled: bool, value: &str) -> io::Result<()> {
    out.write_all(styled(code, enabled, value).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_unicode_without_splitting_utf8() {
        let text = "月光潮汐";
        let lines = visual_lines(text, 4);
        assert_eq!(lines.len(), 2);
        assert_eq!(&text[lines[0].start..lines[0].end], "月光");
        assert_eq!(&text[lines[1].start..lines[1].end], "潮汐");
    }

    #[test]
    fn wraps_explicit_newlines_and_keeps_empty_tail() {
        let text = "one\n";
        let lines = visual_lines(text, 20);
        assert_eq!(lines.len(), 2);
        assert_eq!(&text[lines[0].start..lines[0].end], "one");
        assert_eq!(&text[lines[1].start..lines[1].end], "");
    }

    #[test]
    fn editor_insert_and_delete_respect_char_boundaries() {
        let mut editor = EditorBuffer::new();
        assert!(editor.insert("月"));
        assert!(editor.insert("光"));
        editor.backspace();
        assert_eq!(editor.text, "月");
        editor.move_home();
        editor.delete();
        assert!(editor.text.is_empty());
    }

    #[test]
    fn composer_interrupt_clears_a_draft_before_it_exits() {
        let mut editor = EditorBuffer::new();
        editor.insert("unfinished prompt");
        assert!(!composer_interrupt_requests_exit(&mut editor));
        assert!(editor.text.is_empty());
        assert!(composer_interrupt_requests_exit(&mut editor));
    }

    #[test]
    fn tab_completion_is_scoped_to_slash_command_prefixes() {
        let mut editor = EditorBuffer::new();
        editor.insert("/cont");
        assert!(complete_slash_command(&mut editor));
        assert_eq!(editor.text, "/context");

        let mut ambiguous = EditorBuffer::new();
        ambiguous.insert("/e");
        assert!(complete_slash_command(&mut ambiguous));
        assert_eq!(ambiguous.text, "/e");

        let mut prompt = EditorBuffer::new();
        prompt.insert("ordinary prompt");
        assert!(!complete_slash_command(&mut prompt));
    }

    #[test]
    fn slash_popup_filters_localizes_and_selects_commands() {
        let status = slash_popup_candidates("/st", UiLanguage::English);
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].name, "/status");
        assert!(status[0].description.contains("runtime"));

        let thinking = slash_popup_candidates("/思", UiLanguage::Chinese);
        assert_eq!(thinking.len(), 1);
        assert_eq!(thinking[0].name, "/思考");
        assert!(thinking[0].description.contains("思考级别"));

        let mut editor = EditorBuffer::new();
        editor.insert("/th");
        assert!(replace_with_slash_candidate(
            &mut editor,
            UiLanguage::English,
            0
        ));
        assert_eq!(editor.text, "/thinking");
    }

    #[test]
    fn slash_popup_selection_wraps_and_rendering_is_bounded() {
        let mut selected = 0;
        move_slash_selection(&mut selected, 3, false);
        assert_eq!(selected, 2);
        move_slash_selection(&mut selected, 3, true);
        assert_eq!(selected, 0);

        let candidates = slash_popup_candidates("/", UiLanguage::English);
        let mut output = Vec::new();
        let rows = render_slash_popup(&mut output, &candidates, 0, 40, false).unwrap();
        assert_eq!(rows, MAX_COMMAND_POPUP_ROWS);
        let output = String::from_utf8(output).unwrap();
        assert_eq!(output.matches("\r\n").count(), MAX_COMMAND_POPUP_ROWS);
        assert!(output.contains("/agent"));
        assert!(output.lines().all(|line| display_width(line) <= 40));
    }

    #[test]
    fn display_fitting_handles_narrow_and_wide_text() {
        assert_eq!(fit_display_width("abcdef", 2), "..");
        assert_eq!(display_width(&fit_display_width("镜花水月", 5)), 5);
    }

    #[test]
    fn startup_frame_is_cell_width_safe() {
        let rows = vec![
            "离线 · 无模型 · 无工具".to_string(),
            "0731-main · single-sequence · no-MTP/no-D-Spark".to_string(),
        ];
        for width in [MIN_BOX_WIDTH, 40, MAX_BOX_WIDTH] {
            let frame = startup_frame_lines("TINA ENGINE · Tina", "FAST", &rows, width);
            assert_eq!(frame.len(), rows.len() + 2);
            assert!(frame.iter().all(|line| display_width(line) == width));
        }
    }

    #[test]
    fn product_frames_match_the_demo_width_and_user_label() {
        for width in [MIN_BOX_WIDTH, 40, 80, MAX_BOX_WIDTH] {
            let mut output = Vec::new();
            write_frame_top(&mut output, width, "User", MOON_BRIGHT, false).unwrap();
            write_framed_text(
                &mut output,
                "› 月光\n第二行\x1b[2J",
                width.saturating_sub(6),
                MOON,
                false,
            )
            .unwrap();
            write_frame_bottom(&mut output, width, "", MUTED, "", MOON, false).unwrap();
            let rendered = String::from_utf8(output).unwrap();
            assert!(rendered.contains("╭─ User"));
            assert!(!rendered.contains('\x1b'));
            assert!(rendered.lines().all(|line| display_width(line) == width));
        }
    }

    #[test]
    fn composer_placeholder_disappears_when_editing_begins() {
        let editor = EditorBuffer::new();
        let context = ComposerContext {
            context_positions: 0,
            capacity: 258_048,
            agent_enabled: true,
            thinking: "off",
            color: false,
            language: UiLanguage::Chinese,
            assistant_id: "Tina",
        };

        let mut idle_output = Vec::new();
        let mut idle_render = None;
        redraw_composer(
            &mut idle_output,
            &editor,
            context,
            0,
            true,
            &mut idle_render,
        )
        .unwrap();
        assert!(String::from_utf8(idle_output)
            .unwrap()
            .contains("向 Tina 提问..."));

        let mut editing_output = Vec::new();
        let mut editing_render = None;
        redraw_composer(
            &mut editing_output,
            &editor,
            context,
            0,
            false,
            &mut editing_render,
        )
        .unwrap();
        let editing_output = String::from_utf8(editing_output).unwrap();
        assert!(!editing_output.contains("向 Tina 提问..."));
        assert!(editing_output.contains("│ › "));
    }

    #[test]
    fn assistant_stream_chunks_keep_borders_and_escape_controls() {
        let width = 40;
        let mut output = Vec::new();
        write_frame_top(&mut output, width, "Tina ☾", MOON_BRIGHT, false).unwrap();
        write_frame_body_prefix(&mut output, false).unwrap();
        let mut column = 0;
        let mut line_open = true;
        write_streamed_frame_piece(
            &mut output,
            "第一段\n第二段 ",
            width - 6,
            &mut column,
            &mut line_open,
            MOON_BRIGHT,
            false,
        )
        .unwrap();
        write_streamed_frame_piece(
            &mut output,
            "继续\x1b[2J",
            width - 6,
            &mut column,
            &mut line_open,
            MOON_BRIGHT,
            false,
        )
        .unwrap();
        if line_open {
            write_frame_body_suffix(&mut output, width - 6, column, MOON_BRIGHT, false).unwrap();
        }
        write_frame_bottom(
            &mut output,
            width,
            " FAST · 30.0 tok/s ",
            MUTED,
            " 5/258K ",
            MOON,
            false,
        )
        .unwrap();
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("第二段 继续"));
        assert!(rendered.contains("\\u{001b}[2J"));
        assert!(!rendered.contains('\x1b'));
        assert!(rendered.lines().all(|line| display_width(line) == width));
    }

    #[test]
    fn assistant_reflection_keeps_a_fixed_width_for_every_phase() {
        for phase in [None, Some(0.0), Some(1.57), Some(4.2)] {
            let line = assistant_bottom_line(80, " FAST · 30.0 tok/s ", " 5/258K ", phase, false);
            assert_eq!(display_width(&line), 80);
            assert!(!line.contains('\x1b'));
        }
    }

    #[test]
    fn edit_preview_is_bounded_and_control_safe() {
        let mut output = Vec::new();
        write_edit_preview(
            &mut output,
            40,
            "edit · src/main.rs\x1b[2J",
            "- old line\n+ new line\n",
            " +9 / -8 bytes ",
            " base abcdef012345 ",
            false,
        )
        .unwrap();
        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("edit · src/main.rs\\u{001b}[2J"));
        assert!(rendered.lines().all(|line| display_width(line) == 40));
        assert!(!rendered.contains('\x1b'));
    }

    #[test]
    fn paste_limit_warning_is_localized_and_explicit() {
        let english = paste_rejected_text(UiLanguage::English, 123);
        assert!(english.contains("123 bytes"));
        assert!(english.contains("nothing was inserted"));
        let chinese = paste_rejected_text(UiLanguage::Chinese, 456);
        assert!(chinese.contains("456 字节"));
        assert!(chinese.contains("未写入"));
    }

    #[test]
    fn exit_commands_are_local_only() {
        assert!(is_exit_command("/exit"));
        assert!(is_exit_command(" :q "));
        assert!(!is_exit_command("please /exit later"));
    }

    #[test]
    fn slash_commands_are_separated_from_model_prompts() {
        assert_eq!(command_line(" /plugins "), Some("/plugins".to_string()));
        assert_eq!(
            command_line("/plugin summarize hi"),
            Some("/plugin summarize hi".to_string())
        );
        assert_eq!(command_line("/exit"), None);
        assert_eq!(command_line("//src/main.rs"), None);
        assert_eq!(
            unescape_literal_prompt("//src/main.rs".to_string()),
            "/src/main.rs"
        );
        assert_eq!(
            unescape_literal_prompt("  //workspace/file".to_string()),
            "  /workspace/file"
        );
        assert_eq!(command_line("ordinary prompt"), None);
    }

    #[test]
    fn web_search_control_requires_a_real_provider_and_bounds_queries() {
        assert_eq!(web_search_command("/search"), Some(Ok(None)));
        assert_eq!(
            web_search_command("/网络搜索 rust cuda"),
            Some(Ok(Some("rust cuda".to_string())))
        );
        assert!(web_search_command("/search bad\nquery").unwrap().is_err());
        assert!(web_search_command("ordinary /search text").is_none());
        let ui = TinaCli::with_config(UiConfig {
            language: UiLanguage::English,
            assistant_id: "Tina".to_string(),
        });
        let unavailable = ui.web_search_status_text(false);
        assert!(unavailable.contains("NOT_AVAILABLE"));
        assert!(unavailable.contains("--web-search-url"));
        let available = ui.web_search_status_text(true);
        assert!(available.contains("AVAILABLE"));
        assert!(ui
            .web_search_receipt_text(r#"{"result_count":1}"#)
            .contains("untrusted external data"));
    }

    #[test]
    fn assistant_output_escapes_terminal_controls_but_keeps_layout_whitespace() {
        let mut output = Vec::new();
        write_safe_assistant_piece(&mut output, "ok\n\t\x1b[2J\x07").unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "ok\n\t\\u{001b}[2J\\u{0007}"
        );
    }

    #[test]
    fn assistant_stream_uses_bright_moon_white_without_leaking_style() {
        let mut output = Vec::new();
        write_bright_assistant_piece(&mut output, "月光\x1b[2J", true).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.starts_with(MOON_BRIGHT));
        assert!(output.ends_with(RESET));
        assert!(output.contains("月光\\u{001b}[2J"));
    }

    #[test]
    fn ui_language_accepts_explicit_english_chinese_and_auto() {
        assert_eq!(UiLanguage::parse("en").unwrap(), UiLanguage::English);
        assert_eq!(UiLanguage::parse("zh-CN").unwrap(), UiLanguage::Chinese);
        assert_eq!(UiLanguage::parse("中文").unwrap(), UiLanguage::Chinese);
        assert!(UiLanguage::parse("auto").is_ok());
        assert!(UiLanguage::parse("fr").is_err());
    }

    #[test]
    fn chinese_product_notices_are_localized_without_changing_protocol_names() {
        let ui = TinaCli::with_config(UiConfig {
            language: UiLanguage::Chinese,
            assistant_id: "Tina".to_string(),
        });
        assert!(ui
            .conversation_opened_text(true, "conversation-id", 3, 8)
            .contains("已恢复 Tina 会话"));
        assert!(ui
            .deepseek_import_text("title", 4, "leaf-id", 5, 1, 2)
            .contains("已导入 DeepSeek 会话"));
        assert!(ui
            .todo_status_text("todo-id", "in_progress", "任务")
            .contains("进行中"));
        let tool = ui.agent_tool_result_text("git_status", true);
        assert!(tool.contains("已完成"));
        assert!(tool.contains("git_status"));
        assert!(ui.agent_restart_text("host.exe").contains("下一轮"));
        assert!(ui.agent_restart_failed_text("spawn").contains("旧 host"));
        let help = TinaCli::cli_help_text(UiLanguage::Chinese);
        assert!(help.contains("--doctor"));
        assert!(help.contains("不创建 CUDA context"));
        assert!(help.contains("精确整段 prompt 快照"));
        assert!(help.contains("/agent-restart"));
        let english_help = TinaCli::cli_help_text(UiLanguage::English);
        assert!(english_help.contains("/plugins · /插件"));
        assert!(english_help.contains("/exit, /quit, :q"));
        assert!(english_help.contains("TINA_NO_MOTION"));
    }

    #[test]
    fn assistant_id_is_configurable_but_terminal_safe() {
        let mut ui = TinaCli::with_config(UiConfig::default());
        assert_eq!(ui.assistant_id(), "Tina");
        ui.set_assistant_id("月汐").unwrap();
        assert_eq!(ui.assistant_id(), "月汐");
        assert!(ui.set_assistant_id("  ").is_err());
        assert!(ui.set_assistant_id("bad\nname").is_err());
        assert!(ui.set_assistant_id(&"x".repeat(33)).is_err());
    }

    #[test]
    fn assistant_language_and_help_commands_are_local_only() {
        assert_eq!(
            assistant_id_command("/assistant Tina"),
            Some(AssistantIdCommand::Set("Tina".to_string()))
        );
        assert_eq!(
            assistant_id_command("/助手 月汐"),
            Some(AssistantIdCommand::Set("月汐".to_string()))
        );
        assert_eq!(
            assistant_id_command("/name"),
            Some(AssistantIdCommand::Show)
        );
        assert_eq!(
            language_command("/language zh"),
            Some(LanguageCommand::Set("zh".to_string()))
        );
        assert_eq!(language_command("/语言"), Some(LanguageCommand::Show));
        assert!(is_help_command("/帮助"));
        assert!(is_exit_command("/退出"));
        assert_eq!(assistant_id_command("/assistantish Tina"), None);
    }

    #[test]
    fn session_commands_are_explicit_and_local() {
        assert_eq!(session_command("/agent"), Some(SessionCommand::Agent));
        assert_eq!(session_command("/代理"), Some(SessionCommand::Agent));
        assert_eq!(session_command("/context"), Some(SessionCommand::Context));
        assert_eq!(session_command("/status"), Some(SessionCommand::Status));
        assert_eq!(session_command("/状态"), Some(SessionCommand::Status));
        assert_eq!(session_command("/tools"), Some(SessionCommand::Tools));
        assert_eq!(session_command("/工具"), Some(SessionCommand::Tools));
        assert_eq!(
            session_command("/agent-restart"),
            Some(SessionCommand::AgentRestart)
        );
        assert_eq!(
            session_command("/重启工具"),
            Some(SessionCommand::AgentRestart)
        );
        assert_eq!(
            session_command("/environment"),
            Some(SessionCommand::Environment)
        );
        assert_eq!(session_command("/环境"), Some(SessionCommand::Environment));
        assert_eq!(session_command("/会话"), Some(SessionCommand::Sessions));
        assert_eq!(session_command("/新会话"), Some(SessionCommand::New));
        assert_eq!(session_command("/compact"), Some(SessionCommand::Compact));
        assert_eq!(session_command("/retry"), Some(SessionCommand::Retry));
        assert!(is_memory_command("/记忆"));
        assert_eq!(
            remember_command("/记住 机器是 V100"),
            Some(Some("机器是 V100"))
        );
        assert_eq!(
            project_memory_command("/project-memory"),
            Some(Ok(ProjectMemoryCommand::Status))
        );
        assert_eq!(
            project_memory_command("/project-remember V100 is the release target"),
            Some(Ok(ProjectMemoryCommand::Remember(
                "V100 is the release target".to_string()
            )))
        );
        assert_eq!(
            project_memory_command("/项目记忆 忘记 abcdef123456"),
            Some(Ok(ProjectMemoryCommand::Forget("abcdef123456".to_string())))
        );
        assert!(project_memory_command("/project-forget")
            .expect("recognized project-memory command")
            .is_err());
        assert_eq!(project_memory_command("/project-memoryish"), None);
        assert_eq!(session_command("/context now"), None);
        assert_eq!(session_command("ordinary prompt"), None);
    }

    #[test]
    fn goal_and_thinking_commands_are_bounded_and_alias_safe() {
        assert_eq!(goal_command("/goal"), Some(Ok(GoalCommand::Show)));
        assert_eq!(
            goal_command("/目标 set ship the CLI"),
            Some(Ok(GoalCommand::Set("ship the CLI".to_string())))
        );
        assert_eq!(goal_command("/goal clear"), Some(Ok(GoalCommand::Clear)));
        assert!(goal_command("/goal set").unwrap().is_err());
        assert_eq!(
            thinking_command("/thinking off"),
            Some(Ok(ThinkingCommand::Set(ThinkingLevel::Off)))
        );
        assert_eq!(
            thinking_command("/thinking non-thinking"),
            Some(Ok(ThinkingCommand::Set(ThinkingLevel::Off)))
        );
        assert_eq!(
            thinking_command("/思考 xhigh"),
            Some(Ok(ThinkingCommand::Set(ThinkingLevel::Max)))
        );
        assert!(thinking_command("/thinking nonsense").unwrap().is_err());
    }

    #[test]
    fn resume_command_requires_latest_or_a_full_lowercase_id() {
        assert_eq!(
            resume_command("/resume latest"),
            Some(Ok(ResumeTarget::Latest))
        );
        assert_eq!(resume_command("/恢复 最新"), Some(Ok(ResumeTarget::Latest)));
        let id = "a".repeat(64);
        assert_eq!(
            resume_command(&format!("/resume {id}")),
            Some(Ok(ResumeTarget::Conversation(id)))
        );
        assert!(resume_command("/resume").unwrap().is_err());
        assert!(resume_command("/resume abc").unwrap().is_err());
        assert!(resume_command(&format!("/resume {}", "A".repeat(64)))
            .unwrap()
            .is_err());
        assert_eq!(resume_command("/resuming latest"), None);
    }

    #[test]
    fn tools_text_discloses_execution_surfaces_by_scope() {
        let ui = TinaCli::with_config(UiConfig::default());
        let bounded = ui.tools_text(
            Path::new("tool-host"),
            Path::new("workspace"),
            false,
            &["read", "git_diff"],
            &["apply_patch"],
        );
        assert!(bounded.contains("read-only"));
        assert!(bounded.contains("apply_patch"));
        assert!(bounded.contains("disabled"));
        assert!(bounded.contains("DELETE"));
        assert!(bounded.contains("subagents"));
        assert!(bounded.contains("exact EXECUTE"));
        assert!(bounded.contains("not an OS sandbox"));
        assert!(!bounded.contains("machine-wide"));
        let with_web = ui.tools_text(
            Path::new("tool-host"),
            Path::new("workspace"),
            false,
            &["read", "web_search"],
            &["write_file", "delete"],
        );
        assert!(with_web.contains("configured web search"));
        let full = ui.tools_text(
            Path::new("tool-host"),
            Path::new("workspace"),
            true,
            &["read", "git_diff"],
            &["apply_patch", "shell"],
        );
        assert!(full.contains("shell"));
        assert!(full.contains("network"));
        assert!(full.contains("FULL_ACCESS acknowledged"));
        assert!(full.contains("workspace_exec always requires exact EXECUTE"));
    }

    #[test]
    fn workspace_commands_keep_diff_argument_free_and_bound_review_focus() {
        assert_eq!(workspace_command("/diff"), Some(Ok(WorkspaceCommand::Diff)));
        assert_eq!(
            workspace_command("/差异 extra"),
            Some(Err("Usage: /diff (or /差异)".to_string()))
        );
        assert_eq!(
            workspace_command("/review"),
            Some(Ok(WorkspaceCommand::Review { focus: None }))
        );
        assert_eq!(
            workspace_command("/审查 unsafe 边界"),
            Some(Ok(WorkspaceCommand::Review {
                focus: Some("unsafe 边界".to_string())
            }))
        );
        assert!(workspace_command("/review bad\u{1b}").unwrap().is_err());
        assert!(
            workspace_command(&format!("/review {}", "x".repeat(8 * 1024 + 1)))
                .unwrap()
                .is_err()
        );
        assert_eq!(workspace_command("/reviewing"), None);
    }

    #[test]
    fn workspace_diff_text_uses_explicit_empty_markers() {
        let ui = TinaCli::with_config(UiConfig::default());
        let text = ui.workspace_diff_text("## main\n", "", "diff --cached\n");
        assert!(text.contains("(none)"));
        assert!(text.contains("## main"));
        assert!(text.contains("diff --cached"));
    }

    #[test]
    fn notices_escape_terminal_controls() {
        let mut output = Vec::new();
        write_safe_assistant_piece(&mut output, "status\r\x1b[2J\n").unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "status\\u{000d}\\u{001b}[2J\n"
        );
    }

    #[test]
    fn export_command_preserves_paths_and_infers_safe_formats() {
        assert_eq!(
            export_command(r#"/export md "F:\chat logs\today.md""#),
            Some(Ok(ExportCommand {
                format: ExportFormat::Markdown,
                path: r#"F:\chat logs\today.md"#.to_string(),
            }))
        );
        assert_eq!(
            export_command("/导出 F:/chat.jsonl"),
            Some(Ok(ExportCommand {
                format: ExportFormat::Jsonl,
                path: "F:/chat.jsonl".to_string(),
            }))
        );
        assert!(export_command("/export").unwrap().is_err());
        assert!(export_command("/export -").unwrap().is_err());
        assert_eq!(export_command("/exporting file.md"), None);
    }

    #[test]
    fn context_status_is_explicit_about_unmeasured_gpu_cache() {
        let ui = TinaCli::with_config(UiConfig::default());
        let budget = context_budget::ContextBudget::for_owner(
            context_budget::PRODUCT_KV_POSITIONS,
            context_budget::PRODUCT_OUTPUT_RESERVE,
        );
        let status = ui.context_status_text(245_145, budget, true);
        assert!(status.contains("NOT_MEASURED"));
        assert!(status.contains("258.0K"));
    }

    #[test]
    fn deletion_interceptor_requires_exact_confirmation() {
        assert!(delete_confirmation_is_exact("DELETE\n"));
        assert!(delete_confirmation_is_exact("DELETE\r\n"));
        assert!(!delete_confirmation_is_exact("delete\n"));
        assert!(!delete_confirmation_is_exact(" DELETE\n"));
        assert!(!delete_confirmation_is_exact("DELETE \n"));
        assert!(!delete_confirmation_is_exact("y\n"));
        assert!(!delete_confirmation_is_exact("\n"));
    }

    #[test]
    fn overwrite_confirmation_requires_exact_text() {
        assert!(overwrite_confirmation_is_exact("OVERWRITE\n"));
        assert!(overwrite_confirmation_is_exact("OVERWRITE\r\n"));
        assert!(!overwrite_confirmation_is_exact("overwrite\n"));
        assert!(!overwrite_confirmation_is_exact(" OVERWRITE\n"));
    }

    #[test]
    fn patch_interceptor_requires_exact_confirmation() {
        assert!(apply_patch_confirmation_is_exact("APPLY\n"));
        assert!(apply_patch_confirmation_is_exact("APPLY\r\n"));
        assert!(!apply_patch_confirmation_is_exact("apply\n"));
        assert!(!apply_patch_confirmation_is_exact(" APPLY\n"));
        assert!(!apply_patch_confirmation_is_exact("APPLY \n"));
        assert!(!apply_patch_confirmation_is_exact("y\n"));
        assert!(!apply_patch_confirmation_is_exact("\n"));
    }

    #[test]
    fn execution_interceptor_requires_exact_confirmation() {
        assert!(run_confirmation_is_exact("EXECUTE\n"));
        assert!(run_confirmation_is_exact("EXECUTE\r\n"));
        assert!(!run_confirmation_is_exact("execute\n"));
        assert!(!run_confirmation_is_exact(" EXECUTE\n"));
        assert!(!run_confirmation_is_exact("EXECUTE \n"));
        assert!(!run_confirmation_is_exact("APPLY\n"));
    }

    #[test]
    fn mutating_approval_is_fail_closed_without_a_terminal() {
        let ui = TinaCli {
            interactive: false,
            color: false,
            language: UiLanguage::English,
            assistant_id: DEFAULT_ASSISTANT_ID.to_string(),
            history: Vec::new(),
            history_cursor: None,
            assistant_frame_open: Cell::new(false),
            assistant_line_open: Cell::new(false),
            assistant_body_width: Cell::new(0),
            assistant_column: Cell::new(0),
        };
        assert!(!ui
            .confirm_apply_patch(ApplyPatchConfirmation {
                target: "src/lib.rs",
                expected_sha256: &"ab".repeat(32),
                edit_count: 1,
                removed_bytes: 1,
                added_bytes: 1,
                preview: "edit 1\n- \"a\"\n+ \"b\"\n",
                preview_truncated: false,
            })
            .unwrap());
        assert!(!ui
            .confirm_write_file(WriteFileConfirmation {
                target: "src/generated.rs",
                create: true,
                expected_sha256: None,
                content_bytes: 2,
                content_sha256: &"ab".repeat(32),
                preview: "x\n",
                preview_truncated: false,
            })
            .unwrap());
        assert!(!ui
            .confirm_workspace_mutation("mkdir", "path=generated recursive=false")
            .unwrap());
        assert!(!ui
            .confirm_workspace_exec("argv=\"cargo\" \"check\"; policy=bounded-argv-cwd-v1")
            .unwrap());
        assert!(!ui.confirm_delete("src/lib.rs").unwrap());
    }
}
