//! Transposition Table (TT) and Dedicated Strategic Transposition Cache (StrategicTT).

use saccadion_chess::types::Move;
use saccadion_nn::stream_b::{PolicyMoveScore, StreamBOutput};

pub const MATE_SCORE: i32 = 30_000;
pub const MATE_THRESHOLD: i32 = 29_000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Exact = 1,
    Lower = 2,
    Upper = 3,
}

#[derive(Copy, Clone, Debug)]
pub struct TtEntry {
    pub key: u64,
    pub best_move: Move,
    pub score: i16,
    pub depth: i8,
    pub bound: Bound,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            key: 0,
            best_move: Move::NULL,
            score: 0,
            depth: -1,
            bound: Bound::None,
        }
    }
}

pub struct TranspositionTable {
    entries: Vec<TtEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new_mb(mb: usize) -> Self {
        let bytes = mb.max(1) * 1024 * 1024;
        let entry_size = std::mem::size_of::<TtEntry>().max(1);
        let count = (bytes / entry_size).next_power_of_two() >> 1;
        let len = count.max(1024);
        Self {
            entries: vec![TtEntry::default(); len],
            mask: len - 1,
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(TtEntry::default());
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let e = self.entries[(key as usize) & self.mask];
        if e.bound != Bound::None && e.key == key {
            Some(e)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn store(&mut self, key: u64, best_move: Move, score: i32, depth: i32, bound: Bound, ply: i32) {
        let idx = (key as usize) & self.mask;
        let existing = &mut self.entries[idx];
        if existing.key != key || depth as i8 >= existing.depth || bound == Bound::Exact {
            let tt_score = score_to_tt(score, ply).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            existing.key = key;
            if !best_move.is_null() {
                existing.best_move = best_move;
            }
            existing.score = tt_score;
            existing.depth = depth as i8;
            existing.bound = bound;
        }
    }
}

#[inline(always)]
pub fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_THRESHOLD {
        score + ply
    } else if score < -MATE_THRESHOLD {
        score - ply
    } else {
        score
    }
}

#[inline(always)]
pub fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score > MATE_THRESHOLD {
        score - ply
    } else if score < -MATE_THRESHOLD {
        score + ply
    } else {
        score
    }
}

/// Compact cached output from Stream B Central Complex so transpositions cost 0 ns.
#[derive(Clone, Debug)]
pub struct StrategicEntry {
    pub key: u64,
    pub deep_cp: f32,
    pub pfl2_sharpness: f32,
    pub ticks_used: usize,
    pub strategic_pressure: [f32; 5],
    pub top_moves: Vec<PolicyMoveScore>,
    pub gamma_seed: Vec<f32>,
    pub centrifugal_kc_mask: Vec<f32>,
    pub committed_goal: Vec<f32>,
}

pub struct StrategicTT {
    slots: Vec<Option<StrategicEntry>>,
    mask: usize,
}

impl StrategicTT {
    pub fn new(capacity_pow2: usize) -> Self {
        let len = capacity_pow2.next_power_of_two().max(256);
        Self {
            slots: vec![None; len],
            mask: len - 1,
        }
    }

    pub fn clear(&mut self) {
        for s in &mut self.slots {
            *s = None;
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<&StrategicEntry> {
        let slot = &self.slots[(key as usize) & self.mask];
        match slot {
            Some(e) if e.key == key => Some(e),
            _ => None,
        }
    }

    pub fn store(&mut self, key: u64, out: &StreamBOutput) {
        let idx = (key as usize) & self.mask;
        let top_k = out.policy_moves.len().min(8);
        self.slots[idx] = Some(StrategicEntry {
            key,
            deep_cp: out.deep_cp,
            pfl2_sharpness: out.pfl2_sharpness,
            ticks_used: out.ticks_used,
            strategic_pressure: out.strategic_pressure,
            top_moves: out.policy_moves[..top_k].to_vec(),
            gamma_seed: out.gamma_seed.clone(),
            centrifugal_kc_mask: out.centrifugal_kc_mask.clone(),
            committed_goal: out.committed_goal.clone(),
        });
    }
}
