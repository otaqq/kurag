//! The rules of kurag, the same on every backend: how text is cut into terms, how hits are scored and fused, how
//! chunk ids and namespaces work, and the request and answer types of the HTTP API.

pub mod ids;
pub mod score;
pub mod text;
pub mod types;
