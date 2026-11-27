//! Lock-Free Atomic Transposition Table (TT) and Dedicated Strategic Cache (StrategicTT).
//!
//! Uses classic XOR-verified 128-bit atomic slots (`key ^ data`, `data`) so multiple Parallel
//! DNp (Lazy SMP) worker threads can probe and store concurrently with zero mutex contention!

use saccadion_chess::types::Move;
use saccadion_nn::stream_b::{PolicyMoveScore, StreamBOutput};
use std::sync::atomic::{AtomicU64, Ordering};

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

struct AtomicTtSlot {
    key_xor: AtomicU64,
    data: AtomicU64,
}

#[inline(always)]
fn pack_tt_data(best_move: Move, score: i16, depth: i8, bound: Bound) -> u64 {
    (best_move.0 as u64)
        | (((score as u16) as u64) << 16)
        | (((depth as u8) as u64) << 32)
        | (((bound as u8) as u64) << 40)
}

#[inline(always)]
fn unpack_tt_data(key: u64, data: u64) -> Option<TtEntry> {
    let bound = match ((data >> 40) & 0x3) as u8 {
        1 => Bound::Exact,
        2 => Bound::Lower,
        3 => Bound::Upper,
        _ => return None,
    };
    let best_move = Move((data & 0xFFFF) as u16);
    let score = ((data >> 16) & 0xFFFF) as u16 as i16;
    let depth = ((data >> 32) & 0xFF) as u8 as i8;
    Some(TtEntry {
        key,
        best_move,
        score,
        depth,
        bound,
    })
}

pub struct TranspositionTable {
    entries: Vec<AtomicTtSlot>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new_mb(mb: usize) -> Self {
        let bytes = mb.max(1) * 1024 * 1024;
        let entry_size = std::mem::size_of::<AtomicTtSlot>().max(16);
        let count = (bytes / entry_size).next_power_of_two() >> 1;
        let len = count.max(1024);
        let mut entries = Vec::with_capacity(len);
        for _ in 0..len {
            entries.push(AtomicTtSlot {
                key_xor: AtomicU64::new(0),
                data: AtomicU64::new(0),
            });
        }
        Self {
            entries,
            mask: len - 1,
        }
    }

    pub fn clear(&self) {
        for slot in &self.entries {
            slot.key_xor.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let slot = &self.entries[(key as usize) & self.mask];
        let data = slot.data.load(Ordering::Relaxed);
        let key_xor = slot.key_xor.load(Ordering::Relaxed);
        if (key_xor ^ data) != key {
            return None;
        }
        unpack_tt_data(key, data)
    }

    #[inline(always)]
    pub fn store(&self, key: u64, best_move: Move, score: i32, depth: i32, bound: Bound, ply: i32) {
        let idx = (key as usize) & self.mask;
        let slot = &self.entries[idx];
        let old_data = slot.data.load(Ordering::Relaxed);
        let old_key = slot.key_xor.load(Ordering::Relaxed) ^ old_data;
        let old_depth = ((old_data >> 32) & 0xFF) as u8 as i8;
        let same_key = old_key == key && ((old_data >> 40) & 0x3) != 0;

        if !same_key || (depth as i8) >= old_depth || bound == Bound::Exact {
            let tt_score = score_to_tt(score, ply).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            let mv_to_store = if !best_move.is_null() {
                best_move
            } else if same_key {
                Move((old_data & 0xFFFF) as u16)
            } else {
                Move::NULL
            };
            let new_data = pack_tt_data(mv_to_store, tt_score, depth as i8, bound);
            slot.data.store(new_data, Ordering::Relaxed);
            slot.key_xor.store(key ^ new_data, Ordering::Relaxed);
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
    pub retro_goal_saliency: [f32; 64],
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
            retro_goal_saliency: out.retro_goal_saliency,
        });
    }
}
