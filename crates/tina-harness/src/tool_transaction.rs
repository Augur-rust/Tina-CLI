//! Bounded, approval-aware tool transaction state for the Agent loop.
//!
//! This module deliberately contains no process, filesystem, or CUDA code. It
//! is the protocol-side state machine that a future out-of-process tool host
//! and the CLI archive can share.

use sha2::{Digest, Sha256};
use std::fmt;

pub const DEFAULT_MAX_AGENT_STEPS: usize = 32;
pub const DEFAULT_MAX_TOOL_CALLS_PER_TURN: usize = 8;
pub const DEFAULT_MAX_TOOL_RESULT_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_TOOL_WALL_MS: u64 = 120_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolTransactionState {
    Proposed,
    AwaitingApproval,
    Approved,
    Running,
    Succeeded,
    Failed,
    Denied,
    Cancelled,
}

impl ToolTransactionState {
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Denied | Self::Cancelled
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolResultReceipt {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub output_truncated: bool,
    pub artifact_path: Option<String>,
}

impl ToolResultReceipt {
    pub fn inline_bytes(&self) -> usize {
        self.stdout.len().saturating_add(self.stderr.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolCallTransaction {
    call_id: String,
    tool_name: String,
    arguments_json: String,
    state: ToolTransactionState,
    result: Option<ToolResultReceipt>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolTransactionError {
    EmptyCallId,
    EmptyToolName,
    EmptyArguments,
    ArgumentsTooLarge {
        bytes: usize,
        limit: usize,
    },
    ResultTooLarge {
        bytes: usize,
        limit: usize,
    },
    InvalidTransition {
        state: ToolTransactionState,
        operation: &'static str,
    },
}

impl fmt::Display for ToolTransactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCallId => formatter.write_str("tool call ID cannot be empty"),
            Self::EmptyToolName => formatter.write_str("tool name cannot be empty"),
            Self::EmptyArguments => formatter.write_str("tool arguments cannot be empty"),
            Self::ArgumentsTooLarge { bytes, limit } => {
                write!(
                    formatter,
                    "tool arguments are {bytes} bytes; limit is {limit}"
                )
            }
            Self::ResultTooLarge { bytes, limit } => {
                write!(formatter, "tool result is {bytes} bytes; limit is {limit}")
            }
            Self::InvalidTransition { state, operation } => {
                write!(formatter, "cannot {operation} tool call in state {state:?}")
            }
        }
    }
}

impl std::error::Error for ToolTransactionError {}

impl ToolCallTransaction {
    pub fn new(
        call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments_json: impl Into<String>,
        argument_limit: usize,
    ) -> Result<Self, ToolTransactionError> {
        let call_id = call_id.into();
        let tool_name = tool_name.into();
        let arguments_json = arguments_json.into();
        if call_id.trim().is_empty() {
            return Err(ToolTransactionError::EmptyCallId);
        }
        if tool_name.trim().is_empty() {
            return Err(ToolTransactionError::EmptyToolName);
        }
        if arguments_json.is_empty() {
            return Err(ToolTransactionError::EmptyArguments);
        }
        if arguments_json.len() > argument_limit {
            return Err(ToolTransactionError::ArgumentsTooLarge {
                bytes: arguments_json.len(),
                limit: argument_limit,
            });
        }
        Ok(Self {
            call_id,
            tool_name,
            arguments_json,
            state: ToolTransactionState::Proposed,
            result: None,
        })
    }

    pub fn call_id(&self) -> &str {
        &self.call_id
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn arguments_json(&self) -> &str {
        &self.arguments_json
    }

    pub const fn state(&self) -> ToolTransactionState {
        self.state
    }

    pub fn result(&self) -> Option<&ToolResultReceipt> {
        self.result.as_ref()
    }

    pub fn request_approval(&mut self) -> Result<(), ToolTransactionError> {
        self.transition(
            ToolTransactionState::AwaitingApproval,
            "request approval",
            |state| *state == ToolTransactionState::Proposed,
        )
    }

    pub fn approve(&mut self) -> Result<(), ToolTransactionError> {
        self.transition(ToolTransactionState::Approved, "approve", |state| {
            *state == ToolTransactionState::AwaitingApproval
        })
    }

    pub fn deny(&mut self) -> Result<(), ToolTransactionError> {
        self.transition(ToolTransactionState::Denied, "deny", |state| {
            matches!(
                *state,
                ToolTransactionState::Proposed | ToolTransactionState::AwaitingApproval
            )
        })
    }

    pub fn start(&mut self) -> Result<(), ToolTransactionError> {
        self.transition(ToolTransactionState::Running, "start", |state| {
            *state == ToolTransactionState::Approved
        })
    }

    pub fn complete(
        &mut self,
        result: ToolResultReceipt,
        result_limit: usize,
    ) -> Result<(), ToolTransactionError> {
        if self.state != ToolTransactionState::Running {
            return Err(ToolTransactionError::InvalidTransition {
                state: self.state,
                operation: "complete",
            });
        }
        let bytes = result.inline_bytes();
        if bytes > result_limit {
            return Err(ToolTransactionError::ResultTooLarge {
                bytes,
                limit: result_limit,
            });
        }
        self.result = Some(result);
        self.state = ToolTransactionState::Succeeded;
        Ok(())
    }

    pub fn fail(
        &mut self,
        result: ToolResultReceipt,
        result_limit: usize,
    ) -> Result<(), ToolTransactionError> {
        if self.state != ToolTransactionState::Running {
            return Err(ToolTransactionError::InvalidTransition {
                state: self.state,
                operation: "fail",
            });
        }
        let bytes = result.inline_bytes();
        if bytes > result_limit {
            return Err(ToolTransactionError::ResultTooLarge {
                bytes,
                limit: result_limit,
            });
        }
        self.result = Some(result);
        self.state = ToolTransactionState::Failed;
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<(), ToolTransactionError> {
        if self.state.is_terminal() {
            return Err(ToolTransactionError::InvalidTransition {
                state: self.state,
                operation: "cancel",
            });
        }
        self.state = ToolTransactionState::Cancelled;
        Ok(())
    }

    fn transition(
        &mut self,
        next: ToolTransactionState,
        operation: &'static str,
        allowed: impl FnOnce(&ToolTransactionState) -> bool,
    ) -> Result<(), ToolTransactionError> {
        if !allowed(&self.state) {
            return Err(ToolTransactionError::InvalidTransition {
                state: self.state,
                operation,
            });
        }
        self.state = next;
        Ok(())
    }
}

/// Derive a stable call ID because DSML itself does not carry one.
pub fn derive_tool_call_id(
    turn_id: &str,
    ordinal: usize,
    tool_name: &str,
    arguments_json: &str,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"CQ-AGENT-TOOL-CALL-V1\0");
    for value in [turn_id, tool_name, arguments_json] {
        digest.update((value.len() as u64).to_le_bytes());
        digest.update(value.as_bytes());
    }
    digest.update((ordinal as u64).to_le_bytes());
    format!("tool-{:x}", digest.finalize())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentBudget {
    pub max_steps: usize,
    pub max_tool_calls: usize,
    pub max_result_bytes: usize,
    pub max_tool_wall_ms: u64,
}

impl Default for AgentBudget {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_AGENT_STEPS,
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS_PER_TURN,
            max_result_bytes: DEFAULT_MAX_TOOL_RESULT_BYTES,
            max_tool_wall_ms: DEFAULT_MAX_TOOL_WALL_MS,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentBudgetError {
    Steps,
    ToolCalls,
    ResultBytes,
    ToolWall,
    Overflow,
}

impl fmt::Display for AgentBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Agent budget exceeded: {self:?}")
    }
}

impl std::error::Error for AgentBudgetError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentBudgetLedger {
    budget: AgentBudget,
    steps: usize,
    tool_calls: usize,
    result_bytes: usize,
    tool_wall_ms: u64,
}

impl AgentBudgetLedger {
    pub const fn new(budget: AgentBudget) -> Self {
        Self {
            budget,
            steps: 0,
            tool_calls: 0,
            result_bytes: 0,
            tool_wall_ms: 0,
        }
    }

    pub fn begin_step(&mut self) -> Result<(), AgentBudgetError> {
        let next = self
            .steps
            .checked_add(1)
            .ok_or(AgentBudgetError::Overflow)?;
        if next > self.budget.max_steps {
            return Err(AgentBudgetError::Steps);
        }
        self.steps = next;
        Ok(())
    }

    pub fn register_tool_call(&mut self) -> Result<(), AgentBudgetError> {
        let next = self
            .tool_calls
            .checked_add(1)
            .ok_or(AgentBudgetError::Overflow)?;
        if next > self.budget.max_tool_calls {
            return Err(AgentBudgetError::ToolCalls);
        }
        self.tool_calls = next;
        Ok(())
    }

    pub fn account_result_bytes(&mut self, bytes: usize) -> Result<(), AgentBudgetError> {
        let next = self
            .result_bytes
            .checked_add(bytes)
            .ok_or(AgentBudgetError::Overflow)?;
        if next > self.budget.max_result_bytes {
            return Err(AgentBudgetError::ResultBytes);
        }
        self.result_bytes = next;
        Ok(())
    }

    /// Accounts completed tool wall against the per-turn host budget.
    ///
    /// This is accounting only. The process host must still enforce a hard
    /// deadline and reclaim the complete process tree independently.
    pub fn account_tool_wall_ms(&mut self, wall_ms: u64) -> Result<(), AgentBudgetError> {
        let next = self
            .tool_wall_ms
            .checked_add(wall_ms)
            .ok_or(AgentBudgetError::Overflow)?;
        if next > self.budget.max_tool_wall_ms {
            return Err(AgentBudgetError::ToolWall);
        }
        self.tool_wall_ms = next;
        Ok(())
    }

    pub const fn budget(&self) -> AgentBudget {
        self.budget
    }

    pub const fn steps(&self) -> usize {
        self.steps
    }

    pub const fn tool_calls(&self) -> usize {
        self.tool_calls
    }

    pub const fn result_bytes(&self) -> usize {
        self.result_bytes
    }

    pub const fn tool_wall_ms(&self) -> u64 {
        self.tool_wall_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(text: &str) -> ToolResultReceipt {
        ToolResultReceipt {
            exit_code: Some(0),
            stdout: text.to_string(),
            stderr: String::new(),
            output_truncated: false,
            artifact_path: None,
        }
    }

    #[test]
    fn transaction_requires_approval_before_execution() {
        let mut tx = ToolCallTransaction::new("call-1", "read", "{}", 32).unwrap();
        assert_eq!(
            tx.start(),
            Err(ToolTransactionError::InvalidTransition {
                state: ToolTransactionState::Proposed,
                operation: "start",
            })
        );
        tx.request_approval().unwrap();
        tx.approve().unwrap();
        tx.start().unwrap();
        tx.complete(receipt("ok"), 16).unwrap();
        assert_eq!(tx.state(), ToolTransactionState::Succeeded);
        assert_eq!(tx.result().unwrap().stdout, "ok");
    }

    #[test]
    fn terminal_transactions_cannot_be_reused_or_cancelled() {
        let mut tx = ToolCallTransaction::new("call-1", "read", "{}", 32).unwrap();
        tx.request_approval().unwrap();
        tx.deny().unwrap();
        assert!(tx.cancel().is_err());
        assert!(tx.approve().is_err());
    }

    #[test]
    fn results_and_arguments_are_bounded_before_state_mutation() {
        let mut tx = ToolCallTransaction::new("call-1", "read", "{}", 32).unwrap();
        tx.request_approval().unwrap();
        tx.approve().unwrap();
        tx.start().unwrap();
        assert!(matches!(
            tx.complete(receipt("0123456789"), 4),
            Err(ToolTransactionError::ResultTooLarge { .. })
        ));
        assert_eq!(tx.state(), ToolTransactionState::Running);
    }

    #[test]
    fn call_ids_are_deterministic_and_ordinal_sensitive() {
        let first = derive_tool_call_id("turn", 0, "read", "{\"path\":\"a\"}");
        let second = derive_tool_call_id("turn", 0, "read", "{\"path\":\"a\"}");
        let sibling = derive_tool_call_id("turn", 1, "read", "{\"path\":\"a\"}");
        assert_eq!(first, second);
        assert_ne!(first, sibling);
        assert!(first.starts_with("tool-"));
    }

    #[test]
    fn budget_ledger_rejects_each_limit_without_overcounting() {
        let budget = AgentBudget {
            max_steps: 1,
            max_tool_calls: 1,
            max_result_bytes: 4,
            max_tool_wall_ms: 10,
        };
        let mut ledger = AgentBudgetLedger::new(budget);
        ledger.begin_step().unwrap();
        assert_eq!(ledger.begin_step(), Err(AgentBudgetError::Steps));
        ledger.register_tool_call().unwrap();
        assert_eq!(
            ledger.register_tool_call(),
            Err(AgentBudgetError::ToolCalls)
        );
        ledger.account_result_bytes(4).unwrap();
        assert_eq!(
            ledger.account_result_bytes(1),
            Err(AgentBudgetError::ResultBytes)
        );
        assert_eq!(ledger.result_bytes(), 4);
        ledger.account_tool_wall_ms(10).unwrap();
        assert_eq!(
            ledger.account_tool_wall_ms(1),
            Err(AgentBudgetError::ToolWall)
        );
        assert_eq!(ledger.tool_wall_ms(), 10);
    }

    #[test]
    fn budget_arithmetic_overflow_fails_without_mutation() {
        let budget = AgentBudget {
            max_steps: usize::MAX,
            max_tool_calls: usize::MAX,
            max_result_bytes: usize::MAX,
            max_tool_wall_ms: u64::MAX,
        };
        let mut ledger = AgentBudgetLedger::new(budget);
        ledger.account_result_bytes(usize::MAX).unwrap();
        assert_eq!(
            ledger.account_result_bytes(1),
            Err(AgentBudgetError::Overflow)
        );
        assert_eq!(ledger.result_bytes(), usize::MAX);
        ledger.account_tool_wall_ms(u64::MAX).unwrap();
        assert_eq!(
            ledger.account_tool_wall_ms(1),
            Err(AgentBudgetError::Overflow)
        );
        assert_eq!(ledger.tool_wall_ms(), u64::MAX);
    }
}
