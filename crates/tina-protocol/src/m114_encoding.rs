use serde_json::{json, Map, Value};
use std::collections::HashMap;

pub const M114_BOS: &str = "<｜begin▁of▁sentence｜>";
pub const M114_EOS: &str = "<｜end▁of▁sentence｜>";
pub const M114_USER: &str = "<｜User｜>";
pub const M114_ASSISTANT: &str = "<｜Assistant｜>";
pub const M114_LATEST_REMINDER: &str = "<｜latest_reminder｜>";
pub const M114_THINK_START: &str = "<think>";
pub const M114_THINK_END: &str = "</think>";
pub const M114_DSML: &str = "｜DSML｜";

const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

const TOOLS_PREFIX: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_SUFFIX: &str = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M114ThinkingModeV1 {
    Chat,
    Thinking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum M114ReasoningEffortV1 {
    High,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct M114EncodingOptionsV1 {
    pub thinking_mode: M114ThinkingModeV1,
    pub drop_thinking: bool,
    pub add_bos: bool,
    pub reasoning_effort: Option<M114ReasoningEffortV1>,
}

impl M114EncodingOptionsV1 {
    pub const fn thinking() -> Self {
        Self {
            thinking_mode: M114ThinkingModeV1::Thinking,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: None,
        }
    }

    pub const fn chat() -> Self {
        Self {
            thinking_mode: M114ThinkingModeV1::Chat,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum M114EncodingErrorV1 {
    Json(String),
    RootNotArray,
    MissingRole,
    InvalidRole(String),
    InvalidField(&'static str),
    InvalidTask(String),
    ToolRoleAfterPreprocess,
}

impl core::fmt::Display for M114EncodingErrorV1 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "M114 encoding error: {self:?}")
    }
}

impl std::error::Error for M114EncodingErrorV1 {}

pub fn encode_m114_messages_v1(
    bytes: &[u8],
    options: M114EncodingOptionsV1,
) -> Result<String, M114EncodingErrorV1> {
    crate::validate_strict_json(bytes)
        .map_err(|error| M114EncodingErrorV1::Json(error.to_string()))?;
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| M114EncodingErrorV1::Json(error.to_string()))?;
    let messages = value
        .as_array()
        .cloned()
        .ok_or(M114EncodingErrorV1::RootNotArray)?;
    encode_values(messages, options)
}

fn encode_values(
    messages: Vec<Value>,
    options: M114EncodingOptionsV1,
) -> Result<String, M114EncodingErrorV1> {
    let mut messages = merge_tool_messages(messages)?;
    sort_tool_results(&mut messages)?;
    let mut effective_drop = options.drop_thinking;
    if messages
        .iter()
        .any(|message| field(message, "tools").is_some())
    {
        effective_drop = false;
    }
    if options.thinking_mode == M114ThinkingModeV1::Thinking && effective_drop {
        messages = drop_thinking_messages(messages)?;
    }
    let mut prompt = if options.add_bos {
        M114_BOS.to_owned()
    } else {
        String::new()
    };
    for index in 0..messages.len() {
        prompt.push_str(&render_message(index, &messages, options, effective_drop)?);
    }
    Ok(prompt)
}

fn merge_tool_messages(messages: Vec<Value>) -> Result<Vec<Value>, M114EncodingErrorV1> {
    let mut merged: Vec<Value> = Vec::new();
    for message in messages {
        let role = string_field(&message, "role")?;
        if role == "tool" {
            let block = json!({
                "type": "tool_result",
                "tool_use_id": optional_string(&message, "tool_call_id").unwrap_or_default(),
                "content": optional_string(&message, "content").unwrap_or_default(),
            });
            if let Some(previous) = merged.last_mut() {
                if string_field(previous, "role")? == "user"
                    && field(previous, "content_blocks").is_some()
                {
                    object_mut(previous)?
                        .get_mut("content_blocks")
                        .and_then(Value::as_array_mut)
                        .ok_or(M114EncodingErrorV1::InvalidField("content_blocks"))?
                        .push(block);
                    continue;
                }
            }
            merged.push(json!({"role": "user", "content_blocks": [block]}));
        } else if role == "user" {
            let text = optional_string(&message, "content").unwrap_or_default();
            let block = json!({"type": "text", "text": text});
            if let Some(previous) = merged.last_mut() {
                if string_field(previous, "role")? == "user"
                    && field(previous, "content_blocks").is_some()
                    && field(previous, "task").is_none()
                {
                    object_mut(previous)?
                        .get_mut("content_blocks")
                        .and_then(Value::as_array_mut)
                        .ok_or(M114EncodingErrorV1::InvalidField("content_blocks"))?
                        .push(block);
                    continue;
                }
            }
            let mut object = Map::new();
            object.insert("role".into(), Value::String("user".into()));
            object.insert("content".into(), Value::String(text));
            object.insert("content_blocks".into(), Value::Array(vec![block]));
            for key in ["task", "wo_eos", "mask"] {
                if let Some(value) = field(&message, key) {
                    object.insert(key.into(), value.clone());
                }
            }
            merged.push(Value::Object(object));
        } else {
            merged.push(message);
        }
    }
    Ok(merged)
}

fn sort_tool_results(messages: &mut [Value]) -> Result<(), M114EncodingErrorV1> {
    let mut order = HashMap::<String, usize>::new();
    for message in messages {
        let role = string_field(message, "role")?;
        if role == "assistant" {
            if let Some(calls) = field(message, "tool_calls").and_then(Value::as_array) {
                order.clear();
                for (index, call) in calls.iter().enumerate() {
                    let id = optional_string(call, "id").or_else(|| {
                        field(call, "function").and_then(|value| optional_string(value, "id"))
                    });
                    if let Some(id) = id.filter(|value| !value.is_empty()) {
                        order.insert(id, index);
                    }
                }
            }
        } else if role == "user" && !order.is_empty() {
            let Some(blocks) = object_mut(message)?
                .get_mut("content_blocks")
                .and_then(Value::as_array_mut)
            else {
                continue;
            };
            let mut tools = blocks
                .iter()
                .filter(|block| optional_string(block, "type").as_deref() == Some("tool_result"))
                .cloned()
                .collect::<Vec<_>>();
            tools.sort_by_key(|block| {
                optional_string(block, "tool_use_id")
                    .and_then(|id| order.get(&id).copied())
                    .unwrap_or(0)
            });
            let mut next = 0;
            for block in blocks {
                if optional_string(block, "type").as_deref() == Some("tool_result") {
                    *block = tools[next].clone();
                    next += 1;
                }
            }
        }
    }
    Ok(())
}

fn drop_thinking_messages(messages: Vec<Value>) -> Result<Vec<Value>, M114EncodingErrorV1> {
    let last = last_user_index(&messages)?;
    let mut output = Vec::new();
    for (index, mut message) in messages.into_iter().enumerate() {
        let role = string_field(&message, "role")?;
        if matches!(
            role.as_str(),
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
        ) || index >= last
        {
            output.push(message);
        } else if role == "assistant" {
            object_mut(&mut message)?.shift_remove("reasoning_content");
            output.push(message);
        }
    }
    Ok(output)
}

fn render_message(
    index: usize,
    messages: &[Value],
    options: M114EncodingOptionsV1,
    drop_thinking: bool,
) -> Result<String, M114EncodingErrorV1> {
    let message = &messages[index];
    let role = string_field(message, "role")?;
    let last_user = last_user_index(messages)?;
    let mut prompt = String::new();
    if index == 0
        && options.thinking_mode == M114ThinkingModeV1::Thinking
        && options.reasoning_effort == Some(M114ReasoningEffortV1::Max)
    {
        prompt.push_str(REASONING_EFFORT_MAX);
    }
    match role.as_str() {
        "system" => {
            prompt.push_str(&optional_string(message, "content").unwrap_or_default());
            append_tools_and_response(&mut prompt, message)?;
        }
        "developer" => {
            let content = optional_string(message, "content")
                .filter(|value| !value.is_empty())
                .ok_or(M114EncodingErrorV1::InvalidField("developer.content"))?;
            prompt.push_str(M114_USER);
            prompt.push_str(&content);
            append_tools_and_response(&mut prompt, message)?;
        }
        "user" => {
            prompt.push_str(M114_USER);
            if let Some(blocks) = field(message, "content_blocks").and_then(Value::as_array) {
                let mut parts = Vec::new();
                for block in blocks {
                    match optional_string(block, "type").as_deref() {
                        Some("text") => {
                            parts.push(optional_string(block, "text").unwrap_or_default())
                        }
                        Some("tool_result") => {
                            let content = field(block, "content")
                                .map(render_tool_result_content)
                                .transpose()?
                                .unwrap_or_default();
                            parts.push(format!("<tool_result>{content}</tool_result>"));
                        }
                        Some(value) => parts.push(format!("[Unsupported {value}]")),
                        None => {
                            return Err(M114EncodingErrorV1::InvalidField("content_blocks.type"))
                        }
                    }
                }
                prompt.push_str(&parts.join("\n\n"));
            } else {
                prompt.push_str(&optional_string(message, "content").unwrap_or_default());
            }
        }
        "latest_reminder" => {
            prompt.push_str(M114_LATEST_REMINDER);
            prompt.push_str(&optional_string(message, "content").unwrap_or_default());
        }
        "assistant" => {
            let previous_task = index > 0 && field(&messages[index - 1], "task").is_some();
            if options.thinking_mode == M114ThinkingModeV1::Thinking
                && !previous_task
                && (!drop_thinking || index > last_user)
            {
                prompt.push_str(&optional_string(message, "reasoning_content").unwrap_or_default());
                prompt.push_str(M114_THINK_END);
            }
            prompt.push_str(&optional_string(message, "content").unwrap_or_default());
            if let Some(calls) = field(message, "tool_calls").and_then(Value::as_array) {
                if !calls.is_empty() {
                    prompt.push_str("\n\n<｜DSML｜tool_calls>\n");
                    let mut rendered = Vec::new();
                    for call in calls {
                        rendered.push(render_tool_call(call)?);
                    }
                    prompt.push_str(&rendered.join("\n"));
                    prompt.push_str("\n</｜DSML｜tool_calls>");
                }
            }
            if !bool_field(message, "wo_eos").unwrap_or(false) {
                prompt.push_str(M114_EOS);
            }
        }
        "tool" => return Err(M114EncodingErrorV1::ToolRoleAfterPreprocess),
        value => return Err(M114EncodingErrorV1::InvalidRole(value.into())),
    }

    if index + 1 < messages.len() {
        let next = string_field(&messages[index + 1], "role")?;
        if !matches!(next.as_str(), "assistant" | "latest_reminder") {
            return Ok(prompt);
        }
    }
    if let Some(task) = optional_string(message, "task") {
        let token = match task.as_str() {
            "action" => "<｜action｜>",
            "query" => "<｜query｜>",
            "authority" => "<｜authority｜>",
            "domain" => "<｜domain｜>",
            "title" => "<｜title｜>",
            "read_url" => "<｜read_url｜>",
            value => return Err(M114EncodingErrorV1::InvalidTask(value.into())),
        };
        if task == "action" {
            prompt.push_str(M114_ASSISTANT);
            prompt.push_str(if options.thinking_mode == M114ThinkingModeV1::Thinking {
                M114_THINK_START
            } else {
                M114_THINK_END
            });
        }
        prompt.push_str(token);
    } else if matches!(role.as_str(), "user" | "developer") {
        prompt.push_str(M114_ASSISTANT);
        if options.thinking_mode == M114ThinkingModeV1::Thinking
            && (!drop_thinking || index >= last_user)
        {
            prompt.push_str(M114_THINK_START);
        } else {
            prompt.push_str(M114_THINK_END);
        }
    }
    Ok(prompt)
}

fn append_tools_and_response(
    prompt: &mut String,
    message: &Value,
) -> Result<(), M114EncodingErrorV1> {
    if let Some(tools) = field(message, "tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(TOOLS_PREFIX);
            let mut schemas = Vec::new();
            for tool in tools {
                schemas
                    .push(python_json(field(tool, "function").ok_or(
                        M114EncodingErrorV1::InvalidField("tools.function"),
                    )?)?);
            }
            prompt.push_str(&schemas.join("\n"));
            prompt.push_str(TOOLS_SUFFIX);
        }
    }
    if let Some(format) = field(message, "response_format") {
        prompt.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
        prompt.push_str(&python_json(format)?);
    }
    Ok(())
}

fn render_tool_call(call: &Value) -> Result<String, M114EncodingErrorV1> {
    let function =
        field(call, "function").ok_or(M114EncodingErrorV1::InvalidField("tool_calls.function"))?;
    let name = string_field(function, "name")?;
    let raw = optional_string(function, "arguments").unwrap_or_default();
    let arguments: Value = serde_json::from_str(&raw).unwrap_or_else(|_| json!({"arguments": raw}));
    let object = arguments
        .as_object()
        .ok_or(M114EncodingErrorV1::InvalidField("tool_calls.arguments"))?;
    let mut parameters = Vec::new();
    for (key, value) in object {
        let (is_string, rendered) = if let Some(value) = value.as_str() {
            ("true", value.to_owned())
        } else {
            ("false", python_json(value)?)
        };
        parameters.push(format!(
            "<｜DSML｜parameter name=\"{key}\" string=\"{is_string}\">{rendered}</｜DSML｜parameter>"
        ));
    }
    Ok(format!(
        "<｜DSML｜invoke name=\"{name}\">\n{}\n</｜DSML｜invoke>",
        parameters.join("\n")
    ))
}

fn render_tool_result_content(value: &Value) -> Result<String, M114EncodingErrorV1> {
    if let Some(value) = value.as_str() {
        return Ok(value.to_owned());
    }
    let blocks = value
        .as_array()
        .ok_or(M114EncodingErrorV1::InvalidField("tool_result.content"))?;
    let mut parts = Vec::new();
    for block in blocks {
        match optional_string(block, "type").as_deref() {
            Some("text") => parts.push(optional_string(block, "text").unwrap_or_default()),
            Some(value) => parts.push(format!("[Unsupported {value}]")),
            None => {
                return Err(M114EncodingErrorV1::InvalidField(
                    "tool_result.content.type",
                ))
            }
        }
    }
    Ok(parts.join("\n\n"))
}

fn python_json(value: &Value) -> Result<String, M114EncodingErrorV1> {
    let compact = serde_json::to_string(value)
        .map_err(|error| M114EncodingErrorV1::Json(error.to_string()))?;
    let mut output = String::with_capacity(compact.len() + compact.len() / 8);
    let mut quoted = false;
    let mut escaped = false;
    for character in compact.chars() {
        output.push(character);
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            quoted = true;
        } else if matches!(character, ',' | ':') {
            output.push(' ');
        }
    }
    Ok(output)
}

fn last_user_index(messages: &[Value]) -> Result<usize, M114EncodingErrorV1> {
    for (index, message) in messages.iter().enumerate().rev() {
        if matches!(
            string_field(message, "role")?.as_str(),
            "user" | "developer"
        ) {
            return Ok(index);
        }
    }
    Ok(usize::MAX)
}

fn object_mut(value: &mut Value) -> Result<&mut Map<String, Value>, M114EncodingErrorV1> {
    value
        .as_object_mut()
        .ok_or(M114EncodingErrorV1::InvalidField("message object"))
}

fn field<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.as_object().and_then(|object| object.get(key))
}

fn string_field(value: &Value, key: &'static str) -> Result<String, M114EncodingErrorV1> {
    field(value, key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or(if key == "role" {
            M114EncodingErrorV1::MissingRole
        } else {
            M114EncodingErrorV1::InvalidField(key)
        })
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    field(value, key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn bool_field(value: &Value, key: &str) -> Option<bool> {
    field(value, key).and_then(Value::as_bool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_thinking_and_drop_thinking_match_official_order() {
        let value = json!([
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"Hello"},
            {"role":"assistant","reasoning_content":"old","content":"Hi"},
            {"role":"user","content":"2+2?"},
            {"role":"assistant","reasoning_content":"four","content":"4"}
        ]);
        let actual = encode_m114_messages_v1(
            serde_json::to_string(&value).unwrap().as_bytes(),
            M114EncodingOptionsV1::thinking(),
        )
        .unwrap();
        assert_eq!(
            actual,
            "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hello<｜Assistant｜></think>Hi<｜end▁of▁sentence｜><｜User｜>2+2?<｜Assistant｜><think>four</think>4<｜end▁of▁sentence｜>"
        );
    }

    #[test]
    fn tool_arguments_preserve_strings_and_json_values() {
        let value = json!([
            {"role":"user","content":"call"},
            {"role":"assistant","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{\"s\":\"x\",\"n\":2}"}}]}
        ]);
        let actual = encode_m114_messages_v1(
            serde_json::to_string(&value).unwrap().as_bytes(),
            M114EncodingOptionsV1::thinking(),
        )
        .unwrap();
        assert!(actual.contains("name=\"s\" string=\"true\">x"));
        assert!(actual.contains("name=\"n\" string=\"false\">2"));
    }

    #[test]
    fn duplicate_json_keys_fail_closed() {
        assert!(encode_m114_messages_v1(
            br#"[{"role":"user","role":"assistant"}]"#,
            M114EncodingOptionsV1::chat(),
        )
        .is_err());
    }
}
