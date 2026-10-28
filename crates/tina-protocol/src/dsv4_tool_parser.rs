//! Strict parser for the DeepSeek-V4 0731 DSML completion format.
//!
//! The official 0731 encoder defines the wire format, but its reference
//! parser intentionally accepts only well-formed completions. This module
//! keeps the same contract while adding bounded input and fail-closed errors
//! for a local Agent loop. It does not resolve or execute tool names.

use std::fmt;

use serde_json::{Map, Value};

use crate::dsv4_0731_encoding::Dsv4ThinkingModeV2;
use crate::{validate_strict_json, M114_DSML, M114_EOS, M114_THINK_END};

pub const DSV4_MAX_COMPLETION_BYTES: usize = 1 << 20;
pub const DSV4_MAX_TOOL_CALLS: usize = 64;
pub const DSV4_MAX_PARAMETERS_PER_CALL: usize = 128;
pub const DSV4_MAX_TOOL_NAME_BYTES: usize = 256;
pub const DSV4_MAX_PARAMETER_NAME_BYTES: usize = 256;
pub const DSV4_MAX_ARGUMENT_BYTES: usize = 256 * 1024;

const TOOL_CALLS_START: &str = "\n\n<｜DSML｜tool_calls";
const TOOL_CALLS_END: &str = "</｜DSML｜tool_calls>";
const INVOKE_START: &str = "<｜DSML｜invoke";
const INVOKE_END: &str = "</｜DSML｜invoke>";
const PARAMETER_START: &str = "<｜DSML｜parameter";
const PARAMETER_END: &str = "</｜DSML｜parameter>";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dsv4ParsedToolCallV1 {
    pub name: String,
    pub arguments: Value,
    pub arguments_json: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dsv4ParsedAssistantTurnV1 {
    pub reasoning_content: String,
    pub content: String,
    pub tool_calls: Vec<Dsv4ParsedToolCallV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Dsv4ToolParseErrorV1 {
    CompletionTooLarge { bytes: usize, limit: usize },
    MissingThinkingEnd,
    MissingEos,
    UnexpectedEos,
    UnexpectedContentAfterToolCalls,
    InvalidSpecialToken,
    InvalidMarkup(&'static str),
    InvalidToolName,
    InvalidParameterName,
    DuplicateParameter(String),
    TooManyToolCalls { count: usize, limit: usize },
    TooManyParameters { count: usize, limit: usize },
    ArgumentTooLarge { bytes: usize, limit: usize },
    InvalidJson(String),
}

impl fmt::Display for Dsv4ToolParseErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompletionTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "DSML completion is {bytes} bytes; limit is {limit}"
                )
            }
            Self::MissingThinkingEnd => formatter.write_str("DSML completion is missing </think>"),
            Self::MissingEos => formatter.write_str("DSML completion is missing EOS"),
            Self::UnexpectedEos => {
                formatter.write_str("DSML completion contains an unexpected EOS")
            }
            Self::UnexpectedContentAfterToolCalls => {
                formatter.write_str("DSML completion has content after tool calls")
            }
            Self::InvalidSpecialToken => {
                formatter.write_str("DSML special token appeared in visible content")
            }
            Self::InvalidMarkup(field) => write!(formatter, "invalid DSML markup: {field}"),
            Self::InvalidToolName => formatter.write_str("invalid DSML tool name"),
            Self::InvalidParameterName => formatter.write_str("invalid DSML parameter name"),
            Self::DuplicateParameter(name) => {
                write!(formatter, "duplicate DSML parameter {name:?}")
            }
            Self::TooManyToolCalls { count, limit } => {
                write!(formatter, "DSML tool call count {count} exceeds {limit}")
            }
            Self::TooManyParameters { count, limit } => {
                write!(formatter, "DSML parameter count {count} exceeds {limit}")
            }
            Self::ArgumentTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "DSML arguments are {bytes} bytes; limit is {limit}"
                )
            }
            Self::InvalidJson(error) => write!(formatter, "invalid DSML JSON: {error}"),
        }
    }
}

impl std::error::Error for Dsv4ToolParseErrorV1 {}

/// Parse one complete 0731 assistant completion without executing anything.
///
/// The parser intentionally requires the model's EOS and rejects any content
/// after the closing tool-call block. Call IDs are assigned by the Agent host,
/// because the DSML wire format does not carry them.
pub fn parse_dsv4_completion_v1(
    text: &str,
    thinking_mode: Dsv4ThinkingModeV2,
) -> Result<Dsv4ParsedAssistantTurnV1, Dsv4ToolParseErrorV1> {
    if text.len() > DSV4_MAX_COMPLETION_BYTES {
        return Err(Dsv4ToolParseErrorV1::CompletionTooLarge {
            bytes: text.len(),
            limit: DSV4_MAX_COMPLETION_BYTES,
        });
    }

    let mut cursor = 0;
    let reasoning_content = if thinking_mode == Dsv4ThinkingModeV2::Thinking {
        let end = text[cursor..]
            .find(M114_THINK_END)
            .ok_or(Dsv4ToolParseErrorV1::MissingThinkingEnd)?
            + cursor;
        let reasoning = text[cursor..end].to_owned();
        cursor = end + M114_THINK_END.len();
        reasoning
    } else {
        String::new()
    };

    let Some((summary_end, marker)) = find_next_marker(&text[cursor..]) else {
        return Err(Dsv4ToolParseErrorV1::MissingEos);
    };
    let summary_end = cursor + summary_end;
    let content = text[cursor..summary_end].to_owned();
    reject_special_tokens(&content)?;
    cursor = summary_end;

    if marker == M114_EOS {
        cursor += M114_EOS.len();
        if cursor != text.len() {
            return Err(Dsv4ToolParseErrorV1::UnexpectedContentAfterToolCalls);
        }
        return Ok(Dsv4ParsedAssistantTurnV1 {
            reasoning_content,
            content,
            tool_calls: Vec::new(),
        });
    }

    cursor += TOOL_CALLS_START.len();
    let tool_calls = parse_tool_calls(text, &mut cursor)?;
    if !text[cursor..].starts_with(M114_EOS) {
        return Err(Dsv4ToolParseErrorV1::MissingEos);
    }
    cursor += M114_EOS.len();
    if cursor != text.len() {
        return Err(Dsv4ToolParseErrorV1::UnexpectedContentAfterToolCalls);
    }

    Ok(Dsv4ParsedAssistantTurnV1 {
        reasoning_content,
        content,
        tool_calls,
    })
}

fn find_next_marker(text: &str) -> Option<(usize, &'static str)> {
    let eos = text.find(M114_EOS);
    let calls = text.find(TOOL_CALLS_START);
    match (eos, calls) {
        (None, None) => None,
        (Some(index), None) => Some((index, M114_EOS)),
        (None, Some(index)) => Some((index, TOOL_CALLS_START)),
        (Some(left), Some(right)) if left <= right => Some((left, M114_EOS)),
        (Some(_), Some(right)) => Some((right, TOOL_CALLS_START)),
    }
}

fn parse_tool_calls(
    text: &str,
    cursor: &mut usize,
) -> Result<Vec<Dsv4ParsedToolCallV1>, Dsv4ToolParseErrorV1> {
    expect(text, cursor, ">\n", "tool_calls header")?;
    let mut calls = Vec::new();
    loop {
        if text[*cursor..].starts_with(TOOL_CALLS_END) {
            *cursor += TOOL_CALLS_END.len();
            return Ok(calls);
        }
        if calls.len() == DSV4_MAX_TOOL_CALLS {
            return Err(Dsv4ToolParseErrorV1::TooManyToolCalls {
                count: calls.len() + 1,
                limit: DSV4_MAX_TOOL_CALLS,
            });
        }
        expect(text, cursor, INVOKE_START, "invoke start")?;
        let header = take_until(text, cursor, ">\n", "invoke header")?;
        let name = parse_invoke_name(header)?;
        let mut parameters = Map::new();
        let mut parameter_count = 0;
        loop {
            if text[*cursor..].starts_with(INVOKE_END) {
                *cursor += INVOKE_END.len();
                if text[*cursor..].starts_with('\n') {
                    *cursor += 1;
                }
                break;
            }
            if parameter_count == DSV4_MAX_PARAMETERS_PER_CALL {
                return Err(Dsv4ToolParseErrorV1::TooManyParameters {
                    count: parameter_count + 1,
                    limit: DSV4_MAX_PARAMETERS_PER_CALL,
                });
            }
            expect(text, cursor, PARAMETER_START, "parameter start")?;
            let parameter_header = take_until(text, cursor, ">", "parameter header")?;
            let (parameter_name, is_string) = parse_parameter_header(parameter_header)?;
            let value = take_until(text, cursor, PARAMETER_END, "parameter value")?;
            if parameters.contains_key(&parameter_name) {
                return Err(Dsv4ToolParseErrorV1::DuplicateParameter(parameter_name));
            }
            if is_string {
                parameters.insert(parameter_name, Value::String(value.to_owned()));
            } else {
                validate_strict_json(value.as_bytes())
                    .map_err(|error| Dsv4ToolParseErrorV1::InvalidJson(error.to_string()))?;
                let value = serde_json::from_str(value)
                    .map_err(|error| Dsv4ToolParseErrorV1::InvalidJson(error.to_string()))?;
                parameters.insert(parameter_name, value);
            }
            parameter_count += 1;
            if text[*cursor..].starts_with('\n') {
                *cursor += 1;
            }
        }

        let arguments = Value::Object(parameters);
        let arguments_json = serde_json::to_string(&arguments)
            .map_err(|error| Dsv4ToolParseErrorV1::InvalidJson(error.to_string()))?;
        if arguments_json.len() > DSV4_MAX_ARGUMENT_BYTES {
            return Err(Dsv4ToolParseErrorV1::ArgumentTooLarge {
                bytes: arguments_json.len(),
                limit: DSV4_MAX_ARGUMENT_BYTES,
            });
        }
        calls.push(Dsv4ParsedToolCallV1 {
            name,
            arguments,
            arguments_json,
        });
    }
}

fn parse_invoke_name(header: &str) -> Result<String, Dsv4ToolParseErrorV1> {
    let Some(rest) = header.strip_prefix(" name=\"") else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup("invoke name"));
    };
    let Some(end) = rest.find('"') else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup("invoke name quote"));
    };
    let name = &rest[..end];
    if name.is_empty()
        || name.len() > DSV4_MAX_TOOL_NAME_BYTES
        || name
            .chars()
            .any(|value| value.is_control() || value == '<' || value == '>')
        || !rest[end + 1..].is_empty()
    {
        return Err(Dsv4ToolParseErrorV1::InvalidToolName);
    }
    Ok(name.to_owned())
}

fn parse_parameter_header(header: &str) -> Result<(String, bool), Dsv4ToolParseErrorV1> {
    let Some(rest) = header.strip_prefix(" name=\"") else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup("parameter name"));
    };
    let Some(name_end) = rest.find('"') else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup("parameter name quote"));
    };
    let name = &rest[..name_end];
    let rest = &rest[name_end + 1..];
    let Some(rest) = rest.strip_prefix(" string=\"") else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup("parameter string flag"));
    };
    let Some(flag_end) = rest.find('"') else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup(
            "parameter string quote",
        ));
    };
    let flag = &rest[..flag_end];
    if name.is_empty()
        || name.len() > DSV4_MAX_PARAMETER_NAME_BYTES
        || name
            .chars()
            .any(|value| value.is_control() || value == '<' || value == '>')
        || !rest[flag_end + 1..].is_empty()
    {
        return Err(Dsv4ToolParseErrorV1::InvalidParameterName);
    }
    let is_string = match flag {
        "true" => true,
        "false" => false,
        _ => {
            return Err(Dsv4ToolParseErrorV1::InvalidMarkup(
                "parameter string flag value",
            ))
        }
    };
    Ok((name.to_owned(), is_string))
}

fn expect(
    text: &str,
    cursor: &mut usize,
    expected: &str,
    field: &'static str,
) -> Result<(), Dsv4ToolParseErrorV1> {
    if text[*cursor..].starts_with(expected) {
        *cursor += expected.len();
        Ok(())
    } else {
        Err(Dsv4ToolParseErrorV1::InvalidMarkup(field))
    }
}

fn take_until<'a>(
    text: &'a str,
    cursor: &mut usize,
    marker: &str,
    field: &'static str,
) -> Result<&'a str, Dsv4ToolParseErrorV1> {
    let Some(offset) = text[*cursor..].find(marker) else {
        return Err(Dsv4ToolParseErrorV1::InvalidMarkup(field));
    };
    let start = *cursor;
    *cursor += offset + marker.len();
    Ok(&text[start..start + offset])
}

fn reject_special_tokens(content: &str) -> Result<(), Dsv4ToolParseErrorV1> {
    if content.contains(M114_DSML) || content.contains(M114_EOS) || content.contains(M114_THINK_END)
    {
        return Err(Dsv4ToolParseErrorV1::InvalidSpecialToken);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thinking_completion(body: &str) -> String {
        format!("reasoning{M114_THINK_END}{body}{M114_EOS}")
    }

    #[test]
    fn parses_plain_thinking_completion() {
        let parsed =
            parse_dsv4_completion_v1(&thinking_completion("answer"), Dsv4ThinkingModeV2::Thinking)
                .unwrap();
        assert_eq!(parsed.reasoning_content, "reasoning");
        assert_eq!(parsed.content, "answer");
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parses_string_and_json_parameters() {
        let completion = format!(
            "reasoning{M114_THINK_END}read file\n\n<{M114_DSML}tool_calls>\n<{M114_DSML}invoke name=\"read\">\n<{M114_DSML}parameter name=\"path\" string=\"true\">src/main.rs</{M114_DSML}parameter>\n<{M114_DSML}parameter name=\"line\" string=\"false\">12</{M114_DSML}parameter>\n</{M114_DSML}invoke>\n</{M114_DSML}tool_calls>{M114_EOS}"
        );
        let parsed = parse_dsv4_completion_v1(&completion, Dsv4ThinkingModeV2::Thinking).unwrap();
        assert_eq!(parsed.content, "read file");
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments["path"], "src/main.rs");
        assert_eq!(parsed.tool_calls[0].arguments["line"], 12);
        assert_eq!(
            parsed.tool_calls[0].arguments_json,
            r#"{"path":"src/main.rs","line":12}"#
        );
    }

    #[test]
    fn rejects_duplicate_parameters_and_trailing_content() {
        let duplicate = format!(
            "\n\n<{M114_DSML}tool_calls>\n<{M114_DSML}invoke name=\"read\">\n<{M114_DSML}parameter name=\"path\" string=\"true\">a</{M114_DSML}parameter>\n<{M114_DSML}parameter name=\"path\" string=\"true\">b</{M114_DSML}parameter>\n</{M114_DSML}invoke>\n</{M114_DSML}tool_calls>{M114_EOS}"
        );
        assert!(matches!(
            parse_dsv4_completion_v1(&duplicate, Dsv4ThinkingModeV2::Chat),
            Err(Dsv4ToolParseErrorV1::DuplicateParameter(value)) if value == "path"
        ));

        let trailing = format!("answer{M114_EOS}leak");
        assert_eq!(
            parse_dsv4_completion_v1(&trailing, Dsv4ThinkingModeV2::Chat),
            Err(Dsv4ToolParseErrorV1::UnexpectedContentAfterToolCalls)
        );
    }

    #[test]
    fn rejects_missing_eos_and_invalid_json() {
        assert_eq!(
            parse_dsv4_completion_v1("answer", Dsv4ThinkingModeV2::Chat),
            Err(Dsv4ToolParseErrorV1::MissingEos)
        );
        let invalid = format!(
            "\n\n<{M114_DSML}tool_calls>\n<{M114_DSML}invoke name=\"read\">\n<{M114_DSML}parameter name=\"line\" string=\"false\">NaN</{M114_DSML}parameter>\n</{M114_DSML}invoke>\n</{M114_DSML}tool_calls>{M114_EOS}"
        );
        assert!(matches!(
            parse_dsv4_completion_v1(&invalid, Dsv4ThinkingModeV2::Chat),
            Err(Dsv4ToolParseErrorV1::InvalidJson(_))
        ));
    }
}
