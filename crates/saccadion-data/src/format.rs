//! Exact 128-byte `.sacc` binary training shard format shared byte-for-byte with JAX TPU dataset.py.

use saccadion_chess::position::Position;
use saccadion_chess::types::{Color, Piece, Square};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const SACC_SHARD_MAGIC: [u8; 8] = *b"SACC0001";
pub const SACC_RECORD_BYTES: usize = 128;

/// Fixed-size 128-byte binary record encoding board state + all 8 multi-sensor training targets.
#[derive(Clone, Debug, PartialEq)]
pub struct SaccRecord {
    pub board_nibbles: [u8; 32],
    pub meta_flags: u32,
    pub wdl_target: [f32; 3],
    pub best_move_u16: u16,
    pub next_move_u16: u16,
    pub top4_moves: [u16; 4],
    pub top4_probs_u16: [u16; 4],
    pub pressure_targets: [f32; 5],
    pub lh_targets: [f32; 4],
    pub volatility: f32,
    pub mlh: f32,
    pub threat_mask_stm: u64,
    pub threat_mask_opp: u64,
}

impl SaccRecord {
    /// Packs a Position's 64 squares into 32 bytes (4-bit nibbles, 0=empty, 1..6=W P..K, 7..12=B P..K).
    pub fn pack_board(pos: &Position) -> ([u8; 32], u32) {
        let mut nibbles = [0u8; 32];
        for sq in 0..64 {
            let code: u8 = match pos.piece_at(Square::new(sq as u8)) {
                None => 0,
                Some((Color::White, p)) => 1 + p.index() as u8,
                Some((Color::Black, p)) => 7 + p.index() as u8,
            };
            if (sq & 1) == 0 {
                nibbles[sq >> 1] |= code & 0x0F;
            } else {
                nibbles[sq >> 1] |= (code & 0x0F) << 4;
            }
        }
        let stm_bit = (pos.side_to_move.index() as u32) & 1;
        let castling_bits = ((pos.castling.0 as u32) & 0x0F) << 1;
        let ep_bits = (pos.ep_square.map(|s| s.file() as u32 + 1).unwrap_or(0) & 0x0F) << 5;
        let halfmove_bits = ((pos.halfmove_clock as u32).min(255)) << 9;
        let phase_bits = ((pos.phase_bucket() as u32) & 0x07) << 17;
        let meta = stm_bit | castling_bits | ep_bits | halfmove_bits | phase_bits;
        (nibbles, meta)
    }

    pub fn to_bytes(&self) -> [u8; SACC_RECORD_BYTES] {
        let mut buf = [0u8; SACC_RECORD_BYTES];
        buf[0..32].copy_from_slice(&self.board_nibbles);
        buf[32..36].copy_from_slice(&self.meta_flags.to_le_bytes());
        for i in 0..3 {
            buf[36 + i * 4..40 + i * 4].copy_from_slice(&self.wdl_target[i].to_le_bytes());
        }
        buf[48..50].copy_from_slice(&self.best_move_u16.to_le_bytes());
        buf[50..52].copy_from_slice(&self.next_move_u16.to_le_bytes());
        for i in 0..4 {
            buf[52 + i * 2..54 + i * 2].copy_from_slice(&self.top4_moves[i].to_le_bytes());
        }
        for i in 0..4 {
            buf[60 + i * 2..62 + i * 2].copy_from_slice(&self.top4_probs_u16[i].to_le_bytes());
        }
        for i in 0..5 {
            buf[68 + i * 4..72 + i * 4].copy_from_slice(&self.pressure_targets[i].to_le_bytes());
        }
        for i in 0..4 {
            buf[88 + i * 4..92 + i * 4].copy_from_slice(&self.lh_targets[i].to_le_bytes());
        }
        buf[104..108].copy_from_slice(&self.volatility.to_le_bytes());
        buf[108..112].copy_from_slice(&self.mlh.to_le_bytes());
        buf[112..120].copy_from_slice(&self.threat_mask_stm.to_le_bytes());
        buf[120..128].copy_from_slice(&self.threat_mask_opp.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: &[u8; SACC_RECORD_BYTES]) -> Self {
        let mut board_nibbles = [0u8; 32];
        board_nibbles.copy_from_slice(&buf[0..32]);
        let meta_flags = u32::from_le_bytes(buf[32..36].try_into().unwrap());
        let wdl_target = [
            f32::from_le_bytes(buf[36..40].try_into().unwrap()),
            f32::from_le_bytes(buf[40..44].try_into().unwrap()),
            f32::from_le_bytes(buf[44..48].try_into().unwrap()),
        ];
        let best_move_u16 = u16::from_le_bytes(buf[48..50].try_into().unwrap());
        let next_move_u16 = u16::from_le_bytes(buf[50..52].try_into().unwrap());
        let mut top4_moves = [0u16; 4];
        let mut top4_probs_u16 = [0u16; 4];
        for i in 0..4 {
            top4_moves[i] = u16::from_le_bytes(buf[52 + i * 2..54 + i * 2].try_into().unwrap());
            top4_probs_u16[i] = u16::from_le_bytes(buf[60 + i * 2..62 + i * 2].try_into().unwrap());
        }
        let mut pressure_targets = [0.0f32; 5];
        for i in 0..5 {
            pressure_targets[i] =
                f32::from_le_bytes(buf[68 + i * 4..72 + i * 4].try_into().unwrap());
        }
        let mut lh_targets = [0.0f32; 4];
        for i in 0..4 {
            lh_targets[i] = f32::from_le_bytes(buf[88 + i * 4..92 + i * 4].try_into().unwrap());
        }
        let volatility = f32::from_le_bytes(buf[104..108].try_into().unwrap());
        let mlh = f32::from_le_bytes(buf[108..112].try_into().unwrap());
        let threat_mask_stm = u64::from_le_bytes(buf[112..120].try_into().unwrap());
        let threat_mask_opp = u64::from_le_bytes(buf[120..128].try_into().unwrap());
        Self {
            board_nibbles,
            meta_flags,
            wdl_target,
            best_move_u16,
            next_move_u16,
            top4_moves,
            top4_probs_u16,
            pressure_targets,
            lh_targets,
            volatility,
            mlh,
            threat_mask_stm,
            threat_mask_opp,
        }
    }

    /// Unpacks a piece at square `sq` (utility for verification tests).
    pub fn piece_at(&self, sq: usize) -> Option<(Color, Piece)> {
        let byte = self.board_nibbles[sq >> 1];
        let code = if (sq & 1) == 0 { byte & 0x0F } else { byte >> 4 };
        match code {
            0 => None,
            1..=6 => Some((Color::White, Piece::from_index((code - 1) as usize)?)),
            7..=12 => Some((Color::Black, Piece::from_index((code - 7) as usize)?)),
            _ => None,
        }
    }
}

/// Writes a slice of SaccRecords into a `.sacc` binary shard with 16-byte header.
pub fn write_sacc_shard(path: &Path, records: &[SaccRecord], is_trajectory: bool) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&SACC_SHARD_MAGIC)?;
    let flags: u32 = if is_trajectory { 1 } else { 0 };
    w.write_all(&flags.to_le_bytes())?;
    w.write_all(&(records.len() as u32).to_le_bytes())?;
    for rec in records {
        w.write_all(&rec.to_bytes())?;
    }
    w.flush()
}

/// Reads all SaccRecords from a `.sacc` binary shard.
pub fn read_sacc_shard(path: &Path) -> std::io::Result<(bool, Vec<SaccRecord>)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if magic != SACC_SHARD_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid SACC0001 shard magic",
        ));
    }
    let mut b4 = [0u8; 4];
    r.read_exact(&mut b4)?;
    let is_traj = (u32::from_le_bytes(b4) & 1) != 0;
    r.read_exact(&mut b4)?;
    let count = u32::from_le_bytes(b4) as usize;
    let mut out = Vec::with_capacity(count);
    let mut rec_buf = [0u8; SACC_RECORD_BYTES];
    for _ in 0..count {
        r.read_exact(&mut rec_buf)?;
        out.push(SaccRecord::from_bytes(&rec_buf));
    }
    Ok((is_traj, out))
}
