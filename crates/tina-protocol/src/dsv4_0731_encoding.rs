use crate::{
    encode_m114_messages_v1, M114EncodingErrorV1, M114EncodingOptionsV1, M114ReasoningEffortV1,
    M114ThinkingModeV1, M114_BOS,
};

pub const DSV4_0731_ENCODING_REVISION: &str = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb";

pub const DSV4_0731_REASONING_EFFORT_HIGH: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

pub const DSV4_0731_REASONING_EFFORT_MAX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dsv4ThinkingModeV2 {
    Chat,
    Thinking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Dsv4ReasoningEffortV2 {
    Low,
    High,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Dsv4EncodingOptionsV2 {
    pub thinking_mode: Dsv4ThinkingModeV2,
    pub drop_thinking: bool,
    pub add_bos: bool,
    pub reasoning_effort: Dsv4ReasoningEffortV2,
}

impl Dsv4EncodingOptionsV2 {
    pub const fn thinking() -> Self {
        Self {
            thinking_mode: Dsv4ThinkingModeV2::Thinking,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: Dsv4ReasoningEffortV2::Low,
        }
    }

    pub const fn chat() -> Self {
        Self {
            thinking_mode: Dsv4ThinkingModeV2::Chat,
            drop_thinking: true,
            add_bos: true,
            reasoning_effort: Dsv4ReasoningEffortV2::Low,
        }
    }
}

pub fn encode_dsv4_messages_v2(
    bytes: &[u8],
    options: Dsv4EncodingOptionsV2,
) -> Result<String, M114EncodingErrorV1> {
    let thinking_mode = match options.thinking_mode {
        Dsv4ThinkingModeV2::Chat => M114ThinkingModeV1::Chat,
        Dsv4ThinkingModeV2::Thinking => M114ThinkingModeV1::Thinking,
    };
    // The frozen V1 `Max` value carries the prefix that 0731 calls `high`.
    // Low and the new 0731 max first render the prefix-free V1 body.
    let reasoning_effort = match options.reasoning_effort {
        Dsv4ReasoningEffortV2::High => Some(M114ReasoningEffortV1::Max),
        Dsv4ReasoningEffortV2::Low | Dsv4ReasoningEffortV2::Max => None,
    };
    let mut prompt = encode_m114_messages_v1(
        bytes,
        M114EncodingOptionsV1 {
            thinking_mode,
            drop_thinking: options.drop_thinking,
            add_bos: options.add_bos,
            reasoning_effort,
        },
    )?;

    if options.thinking_mode == Dsv4ThinkingModeV2::Thinking
        && options.reasoning_effort == Dsv4ReasoningEffortV2::Max
    {
        let prefix_offset = if options.add_bos {
            debug_assert!(prompt.starts_with(M114_BOS));
            M114_BOS.len()
        } else {
            0
        };
        prompt.insert_str(prefix_offset, DSV4_0731_REASONING_EFFORT_MAX);
    }
    Ok(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    const CASE_2: &[u8] = br#"[
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Hello"},
        {"role":"assistant","reasoning_content":"The user said hello, I should greet back.","content":"Hi there! How can I help you?"},
        {"role":"user","content":"What is the capital of France?"},
        {"role":"assistant","reasoning_content":"The user asks about the capital of France. It is Paris.","content":"The capital of France is Paris."}
    ]"#;

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn thinking(effort: Dsv4ReasoningEffortV2) -> String {
        encode_dsv4_messages_v2(
            CASE_2,
            Dsv4EncodingOptionsV2 {
                reasoning_effort: effort,
                ..Dsv4EncodingOptionsV2::thinking()
            },
        )
        .unwrap()
    }

    #[test]
    fn official_0731_reasoning_effort_hashes_match_python() {
        assert_eq!(
            digest(&thinking(Dsv4ReasoningEffortV2::Low)),
            "ca66b01a1ac3a204bb032c928fb607d4877171e998a50f7f89f39fa821b75665"
        );
        assert_eq!(
            digest(&thinking(Dsv4ReasoningEffortV2::High)),
            "abb4285245a5faf0cd5c18b9209a5f990d4a55d0fe5219bce5e8aecd6ebfacde"
        );
        assert_eq!(
            digest(&thinking(Dsv4ReasoningEffortV2::Max)),
            "8b1839206453736cc786798bd59f592591452f8c2c0e38cbc7da347ed4f142e9"
        );
    }

    #[test]
    fn max_prefix_follows_bos_and_high_reuses_the_frozen_prefix() {
        let high = thinking(Dsv4ReasoningEffortV2::High);
        let max = thinking(Dsv4ReasoningEffortV2::Max);
        assert!(high.starts_with(&format!("{M114_BOS}{DSV4_0731_REASONING_EFFORT_HIGH}")));
        assert!(max.starts_with(&format!("{M114_BOS}{DSV4_0731_REASONING_EFFORT_MAX}")));
    }

    #[test]
    fn chat_ignores_reasoning_effort() {
        let low = encode_dsv4_messages_v2(CASE_2, Dsv4EncodingOptionsV2::chat()).unwrap();
        let max = encode_dsv4_messages_v2(
            CASE_2,
            Dsv4EncodingOptionsV2 {
                reasoning_effort: Dsv4ReasoningEffortV2::Max,
                ..Dsv4EncodingOptionsV2::chat()
            },
        )
        .unwrap();
        assert_eq!(low, max);
        assert_eq!(
            digest(&max),
            "35e2e45f736772286fcdafc9b0a9954c69f9b8360b5f143359373a67df0f69d2"
        );
    }

    #[test]
    fn max_without_bos_starts_with_the_new_prefix() {
        let prompt = encode_dsv4_messages_v2(
            CASE_2,
            Dsv4EncodingOptionsV2 {
                add_bos: false,
                reasoning_effort: Dsv4ReasoningEffortV2::Max,
                ..Dsv4EncodingOptionsV2::thinking()
            },
        )
        .unwrap();
        assert!(prompt.starts_with(DSV4_0731_REASONING_EFFORT_MAX));
    }

    #[test]
    fn strict_json_fail_closed_is_inherited() {
        assert!(encode_dsv4_messages_v2(
            br#"[{"role":"user","role":"assistant"}]"#,
            Dsv4EncodingOptionsV2::thinking(),
        )
        .is_err());
    }
}
