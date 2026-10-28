//! Engine-free extraction of original DSML encoder/parser; no GPU or inference.
#![allow(dead_code)]
mod dsv4_0731_encoding;
mod dsv4_tool_parser;
mod m114_encoding;
mod strict_json;
pub use dsv4_0731_encoding::*;
pub use dsv4_tool_parser::*;
pub use m114_encoding::*;
#[derive(Debug)]
pub enum ProfileError {
    Json(String),
    InvalidField(&'static str),
    MissingField(&'static str),
}
impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ProfileError {}
pub fn validate_strict_json(bytes: &[u8]) -> Result<(), ProfileError> {
    strict_json::StrictValue::parse(bytes).map(|_| ())
}
