//! Product context-budget calibration for Tina chat.
//!
//! The CLI exposes the calibrated usable input budget, not a larger nominal
//! model window. With 262,144 physical positions and a 4,096 output reserve,
//! the user-visible hard capacity is 258,048 input tokens. Handoff begins at
//! 95% of that active input budget; the earlier 272K x 95% calculation remains
//! calibration provenance only.

#[cfg(test)]
pub(crate) const NOMINAL_CONTEXT_TOKENS: usize = 272_000;
#[cfg(test)]
pub(crate) const NOMINAL_NINETY_FIVE_PERCENT: usize = NOMINAL_CONTEXT_TOKENS * 95 / 100;
/// The product's 256K KV capacity expressed as an exact binary position count.
pub(crate) const PRODUCT_KV_POSITIONS: usize = 262_144;
pub(crate) const PRODUCT_OUTPUT_RESERVE: usize = 4_096;
pub(crate) const CALIBRATED_HANDOFF_INPUT_TOKENS: usize =
    PRODUCT_KV_POSITIONS - PRODUCT_OUTPUT_RESERVE;
#[cfg(test)]
pub(crate) const DEFAULT_PREPARE_INPUT_TOKENS: usize = CALIBRATED_HANDOFF_INPUT_TOKENS * 80 / 100;
#[cfg(test)]
pub(crate) const DEFAULT_FREEZE_INPUT_TOKENS: usize = CALIBRATED_HANDOFF_INPUT_TOKENS * 90 / 100;
#[cfg(test)]
pub(crate) const DEFAULT_COMPACT_INPUT_TOKENS: usize = CALIBRATED_HANDOFF_INPUT_TOKENS * 95 / 100;
/// A byte-level denial-of-service guard, not a context-capacity definition.
///
/// The old 1 MiB guard could reject valid 258K-token histories before the
/// tokenizer and physical KV budget were consulted. Thirty-two MiB remains
/// bounded while leaving ample room for UTF-8 and JSON framing overhead.
pub(crate) const MAX_CONVERSATION_DOCUMENT_BYTES: u64 = 32 * 1024 * 1024;
pub(crate) const MAX_INTERACTIVE_MESSAGE_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const CONTEXT_CAPACITY_REJECTION_SCHEMA: &str =
    "crystal-quant.product.context-capacity-rejection.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContextCapacityRejectionKind {
    EmptyInput,
    PhysicalOwnerCapacity,
    OutputReserveExhausted,
    UsableInputLimit,
}

impl ContextCapacityRejectionKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::EmptyInput => "empty-input",
            Self::PhysicalOwnerCapacity => "physical-owner-capacity",
            Self::OutputReserveExhausted => "output-reserve-exhausted",
            Self::UsableInputLimit => "usable-input-limit",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextCapacityRejection {
    pub(crate) schema: &'static str,
    pub(crate) kind: ContextCapacityRejectionKind,
    pub(crate) input_tokens: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) reserved_output_tokens: usize,
    pub(crate) usable_input_limit: usize,
    pub(crate) owner_positions: usize,
    pub(crate) required_positions: usize,
    pub(crate) overflow_positions: usize,
}

impl std::fmt::Display for ContextCapacityRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "context capacity rejected ({kind}): {input} input tokens plus {reserve} reserved output tokens require {required} positions; usable input limit is {limit} and owner capacity is {owner}",
            kind = self.kind.as_str(),
            input = self.input_tokens,
            reserve = self.reserved_output_tokens,
            required = self.required_positions,
            limit = self.usable_input_limit,
            owner = self.owner_positions,
        )
    }
}

impl std::error::Error for ContextCapacityRejection {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ContextPressure {
    Normal,
    Prepare,
    Freeze,
    Handoff,
    HardLimit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextThresholds {
    pub(crate) prepare_at: usize,
    pub(crate) freeze_at: usize,
    pub(crate) handoff_at: usize,
    pub(crate) hard_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContextBudget {
    pub(crate) owner_positions: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) output_reserve: usize,
    pub(crate) thresholds: ContextThresholds,
    /// Compatibility alias for the soft 95% handoff boundary.
    pub(crate) handoff_at: usize,
    pub(crate) product_256k_owner: bool,
}

impl ContextBudget {
    pub(crate) const fn for_owner(owner_positions: usize, requested_output: usize) -> Self {
        let product_256k_owner = owner_positions >= PRODUCT_KV_POSITIONS;
        let output_reserve = if product_256k_owner {
            if requested_output > PRODUCT_OUTPUT_RESERVE {
                requested_output
            } else {
                PRODUCT_OUTPUT_RESERVE
            }
        } else if requested_output == 0 {
            1
        } else {
            requested_output
        };
        let physical_input = owner_positions.saturating_sub(output_reserve);
        let hard_limit = if physical_input < CALIBRATED_HANDOFF_INPUT_TOKENS {
            physical_input
        } else {
            CALIBRATED_HANDOFF_INPUT_TOKENS
        };
        let thresholds = ContextThresholds {
            prepare_at: hard_limit * 80 / 100,
            freeze_at: hard_limit * 90 / 100,
            handoff_at: hard_limit * 95 / 100,
            hard_limit,
        };
        Self {
            owner_positions,
            requested_output_tokens: requested_output,
            output_reserve,
            thresholds,
            handoff_at: thresholds.handoff_at,
            product_256k_owner,
        }
    }

    #[cfg(test)]
    pub(crate) const fn needs_handoff(self, committed_positions: usize) -> bool {
        matches!(
            self.pressure(committed_positions),
            ContextPressure::Handoff | ContextPressure::HardLimit
        )
    }

    pub(crate) const fn pressure(self, input_tokens: usize) -> ContextPressure {
        if input_tokens >= self.thresholds.hard_limit {
            ContextPressure::HardLimit
        } else if input_tokens >= self.thresholds.handoff_at {
            ContextPressure::Handoff
        } else if input_tokens >= self.thresholds.freeze_at {
            ContextPressure::Freeze
        } else if input_tokens >= self.thresholds.prepare_at {
            ContextPressure::Prepare
        } else {
            ContextPressure::Normal
        }
    }

    /// Admit a complete encoded input before an owner is created or appended.
    ///
    /// The usable input limit is inclusive: 258,048 input tokens plus the
    /// mandatory 4,096-token reserve exactly fill the 262,144-position owner.
    /// A prompt that merely fits the physical owner is still rejected when it
    /// consumes the output reserve.
    pub(crate) const fn admit_input(
        self,
        input_tokens: usize,
    ) -> Result<(), ContextCapacityRejection> {
        let required_positions = input_tokens.saturating_add(self.output_reserve);
        let kind = if input_tokens == 0 {
            Some(ContextCapacityRejectionKind::EmptyInput)
        } else if input_tokens > self.owner_positions {
            Some(ContextCapacityRejectionKind::PhysicalOwnerCapacity)
        } else if self.output_reserve >= self.owner_positions {
            Some(ContextCapacityRejectionKind::OutputReserveExhausted)
        } else if input_tokens > self.thresholds.hard_limit {
            Some(ContextCapacityRejectionKind::UsableInputLimit)
        } else {
            None
        };
        match kind {
            Some(kind) => Err(ContextCapacityRejection {
                schema: CONTEXT_CAPACITY_REJECTION_SCHEMA,
                kind,
                input_tokens,
                requested_output_tokens: self.requested_output_tokens,
                reserved_output_tokens: self.output_reserve,
                usable_input_limit: self.thresholds.hard_limit,
                owner_positions: self.owner_positions,
                required_positions,
                overflow_positions: required_positions.saturating_sub(self.owner_positions),
            }),
            None => Ok(()),
        }
    }

    pub(crate) const fn display_capacity(self) -> usize {
        if self.product_256k_owner {
            self.thresholds.hard_limit
        } else {
            self.owner_positions
        }
    }

    pub(crate) fn status_text(self) -> String {
        if self.product_256k_owner {
            format!(
                "{} usable input context with {} tokens reserved for output on the 256K KV owner ({} positions).",
                compact(self.thresholds.hard_limit),
                compact(self.output_reserve),
                self.owner_positions,
            )
        } else {
            format!(
                "Development context owner: {} positions. The calibrated 258K usable-input policy remains inactive until the 262,144-position owner is qualified.",
                compact(self.owner_positions),
            )
        }
    }
}

fn compact(value: usize) -> String {
    if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_kv_and_output_reserve_override_nominal_95_percent() {
        assert_eq!(NOMINAL_NINETY_FIVE_PERCENT, 258_400);
        assert_eq!(CALIBRATED_HANDOFF_INPUT_TOKENS, 258_048);
        assert_eq!(DEFAULT_PREPARE_INPUT_TOKENS, 206_438);
        assert_eq!(DEFAULT_FREEZE_INPUT_TOKENS, 232_243);
        assert_eq!(DEFAULT_COMPACT_INPUT_TOKENS, 245_145);
        const {
            assert!(CALIBRATED_HANDOFF_INPUT_TOKENS < NOMINAL_NINETY_FIVE_PERCENT);
        }
        let budget = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, 1);
        assert_eq!(budget.handoff_at, 245_145);
        assert_eq!(budget.thresholds.prepare_at, 206_438);
        assert_eq!(budget.thresholds.freeze_at, 232_243);
        assert_eq!(budget.thresholds.handoff_at, 245_145);
        assert_eq!(budget.thresholds.hard_limit, 258_048);
        assert_eq!(budget.output_reserve, 4_096);
        assert_eq!(budget.display_capacity(), 258_048);
        assert!(budget.product_256k_owner);
        let larger_output = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, 8_192);
        assert_eq!(larger_output.output_reserve, 8_192);
        assert_eq!(larger_output.thresholds.hard_limit, 253_952);
        assert_eq!(larger_output.handoff_at, 241_254);
    }

    #[test]
    fn byte_guards_do_not_preempt_the_calibrated_token_budget() {
        assert!(
            MAX_CONVERSATION_DOCUMENT_BYTES
                > (CALIBRATED_HANDOFF_INPUT_TOKENS as u64).saturating_mul(4)
        );
        assert_eq!(
            MAX_INTERACTIVE_MESSAGE_BYTES as u64,
            MAX_CONVERSATION_DOCUMENT_BYTES
        );
    }

    #[test]
    fn smaller_development_owner_never_claims_256k_support() {
        let budget = ContextBudget::for_owner(1_024, 64);
        assert_eq!(budget.thresholds.hard_limit, 960);
        assert_eq!(budget.handoff_at, 912);
        assert_eq!(budget.output_reserve, 64);
        assert_eq!(budget.display_capacity(), 1_024);
        assert!(!budget.product_256k_owner);
        assert!(!budget.needs_handoff(911));
        assert!(budget.needs_handoff(912));
        assert!(budget.status_text().contains("remains inactive"));
    }

    #[test]
    fn pressure_state_boundaries_are_exact_and_monotonic() {
        let budget = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, PRODUCT_OUTPUT_RESERVE);
        assert_eq!(budget.pressure(206_437), ContextPressure::Normal);
        assert_eq!(budget.pressure(206_438), ContextPressure::Prepare);
        assert_eq!(budget.pressure(232_242), ContextPressure::Prepare);
        assert_eq!(budget.pressure(232_243), ContextPressure::Freeze);
        assert_eq!(budget.pressure(245_144), ContextPressure::Freeze);
        assert_eq!(budget.pressure(245_145), ContextPressure::Handoff);
        assert_eq!(budget.pressure(258_047), ContextPressure::Handoff);
        assert_eq!(budget.pressure(258_048), ContextPressure::HardLimit);
        assert_eq!(budget.pressure(usize::MAX), ContextPressure::HardLimit);
    }

    #[test]
    fn product_input_admission_preserves_the_exact_output_reserve_boundary() {
        let budget = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, 64);
        assert!(budget.admit_input(258_047).is_ok());
        assert!(budget.admit_input(258_048).is_ok());

        let reserve_conflict = budget.admit_input(262_144).unwrap_err();
        assert_eq!(
            reserve_conflict.kind,
            ContextCapacityRejectionKind::UsableInputLimit
        );
        assert_eq!(reserve_conflict.input_tokens, 262_144);
        assert_eq!(reserve_conflict.requested_output_tokens, 64);
        assert_eq!(reserve_conflict.reserved_output_tokens, 4_096);
        assert_eq!(reserve_conflict.required_positions, 266_240);
        assert_eq!(reserve_conflict.overflow_positions, 4_096);

        let physical_overflow = budget.admit_input(262_145).unwrap_err();
        assert_eq!(
            physical_overflow.kind,
            ContextCapacityRejectionKind::PhysicalOwnerCapacity
        );
        assert_eq!(physical_overflow.required_positions, 266_241);
        assert_eq!(physical_overflow.overflow_positions, 4_097);
    }

    #[test]
    fn requested_output_can_only_shrink_the_usable_input_limit() {
        let larger_output = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, 8_192);
        assert_eq!(larger_output.thresholds.hard_limit, 253_952);
        assert!(larger_output.admit_input(253_952).is_ok());
        let rejection = larger_output.admit_input(253_953).unwrap_err();
        assert_eq!(
            rejection.kind,
            ContextCapacityRejectionKind::UsableInputLimit
        );
        assert_eq!(rejection.requested_output_tokens, 8_192);
        assert_eq!(rejection.reserved_output_tokens, 8_192);
        assert_eq!(rejection.overflow_positions, 1);

        let exhausted = ContextBudget::for_owner(PRODUCT_KV_POSITIONS, PRODUCT_KV_POSITIONS)
            .admit_input(1)
            .unwrap_err();
        assert_eq!(
            exhausted.kind,
            ContextCapacityRejectionKind::OutputReserveExhausted
        );
    }
    #[test]
    fn small_owner_rejects_an_exhausting_output_reserve_without_clamping() {
        for requested in [1024, 2048, usize::MAX] {
            let budget = ContextBudget::for_owner(1024, requested);
            assert_eq!(budget.output_reserve, requested);
            assert_eq!(budget.thresholds.hard_limit, 0);
            assert_eq!(
                budget.admit_input(1).unwrap_err().kind,
                ContextCapacityRejectionKind::OutputReserveExhausted
            );
        }
        assert!(ContextBudget::for_owner(0, 64).admit_input(1).is_err());
        assert!(ContextBudget::for_owner(262144, 4096)
            .admit_input(0)
            .is_err());
    }
}
