//! From-scratch Pure-Rust chess engine core for Saccadion.
//! Provides Magic Bitboards, legal move generation, incremental bilateral HalfKA + T4/T5 ray
//! features, Lateral Horn tactical/fortress descriptors, and 10-20 ply Strategic Pressure metrics.

pub mod attacks;
pub mod bitboard;
pub mod features;
pub mod movegen;
pub mod perft;
pub mod position;
pub mod pressure;
pub mod types;
pub mod zobrist;

pub use bitboard::Bitboard;
pub use position::Position;
pub use types::{CastlingRights, Color, Move, MoveFlags, MoveList, Piece, Square};
