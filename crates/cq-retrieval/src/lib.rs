#![deny(unsafe_op_in_unsafe_fn)]

//! External retrieval protocols and CPU vector-search primitives.
//!
//! The crate is disabled by default at the product boundary. It does not alter
//! model-internal attention or the release CLI. `usearch-backend` adds the
//! external ANN adapter; exact reranking and its scalar/AVX2 dispatch remain
//! available without a native dependency. The optional semantic-memory store
//! is a host boundary for a future out-of-process embedding service. It must
//! not be linked into the CUDA inference binary on Windows because the CXX
//! bridge and the current CUDA objects use different MSVC CRT selections.

mod distance;
mod protocol;

#[cfg(feature = "usearch-backend")]
mod semantic_memory;
#[cfg(feature = "usearch-backend")]
mod usearch_backend;

pub use distance::*;
pub use protocol::*;

#[cfg(feature = "usearch-backend")]
pub use semantic_memory::*;
#[cfg(feature = "usearch-backend")]
pub use usearch_backend::*;
