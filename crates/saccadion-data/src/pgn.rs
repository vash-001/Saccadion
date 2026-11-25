//! Zero-dependency PGN and Standard Algebraic Notation (SAN) parser with [%eval] comment extraction.

use saccadion_chess::movegen::generate_legal;
use saccadion_chess::position::Position;
use saccadion_chess::types::{Move, MoveList, Piece, Square};

#[derive(Clone, Debug)]
pub struct ParsedGame {
    /// Optional starting FEN for book/Fishtest PGNs (`[FEN "..."]`).
    pub start_fen: Option<String>,
    /// Result from White's perspective: 1.0 = White win, 0.5 = Draw, 0.0 = Black win.
    pub white_result: f32,
    /// Sequence of (legal move, optional centipawn eval from STM perspective BEFORE playing move).
    pub moves: Vec<(Move, Option<f32>)>,
}

/// Parses a single SAN token (e.g. "e4", "Nf3", "Bxc6+", "O-O", "e8=Q#", "Rad1") into a legal Move.
pub fn parse_san_move(pos: &mut Position, raw_san: &str) -> Option<Move> {
    let san = raw_san.trim_end_matches(|c| matches!(c, '+' | '#' | '!' | '?'));
    let mut legal = MoveList::new();
    generate_legal(pos, &mut legal);

    if san == "O-O" || san == "0-0" {
        return legal
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.is_castle() && m.to_sq().file() == 6);
    }
    if san == "O-O-O" || san == "0-0-0" {
        return legal
            .as_slice()
            .iter()
            .copied()
            .find(|m| m.is_castle() && m.to_sq().file() == 2);
    }

    // Check promotion suffix (=Q, =R, =B, =N)
    let (body, promo_piece) = if let Some(eq_idx) = san.find('=') {
        let p = match san.as_bytes().get(eq_idx + 1).copied()? {
            b'Q' => Piece::Queen,
            b'R' => Piece::Rook,
            b'B' => Piece::Bishop,
            b'N' => Piece::Knight,
            _ => return None,
        };
        (&san[..eq_idx], Some(p))
    } else {
        (san, None)
    };

    let clean: String = body.chars().filter(|&c| c != 'x').collect();
    let bytes = clean.as_bytes();
    if bytes.len() < 2 {
        return None;
    }

    let (piece, rest) = match bytes[0] {
        b'K' => (Piece::King, &clean[1..]),
        b'Q' => (Piece::Queen, &clean[1..]),
        b'R' => (Piece::Rook, &clean[1..]),
        b'B' => (Piece::Bishop, &clean[1..]),
        b'N' => (Piece::Knight, &clean[1..]),
        _ => (Piece::Pawn, clean.as_str()),
    };

    if rest.len() < 2 {
        return None;
    }
    let to_str = &rest[rest.len() - 2..];
    let to_sq = Square::from_algebraic(to_str)?;
    let disambig = &rest[..rest.len() - 2];

    let mut req_file: Option<u8> = None;
    let mut req_rank: Option<u8> = None;
    for b in disambig.bytes() {
        if (b'a'..=b'h').contains(&b) {
            req_file = Some(b - b'a');
        } else if (b'1'..=b'8').contains(&b) {
            req_rank = Some(b - b'1');
        }
    }

    for &mv in legal.as_slice() {
        if mv.to_sq() != to_sq {
            continue;
        }
        if mv.promotion_piece() != promo_piece {
            continue;
        }
        let (_, p) = pos.piece_at(mv.from_sq())?;
        if p != piece {
            continue;
        }
        if let Some(f) = req_file {
            if mv.from_sq().file() != f {
                continue;
            }
        }
        if let Some(r) = req_rank {
            if mv.from_sq().rank() != r {
                continue;
            }
        }
        return Some(mv);
    }
    None
}

/// Returns `(eval_cp, is_white_pov)`.
/// Lichess `[%eval ...]` is always White POV; Fishtest/CuteChess `{+1.84/19 1.7s}` is moving-side POV.
fn extract_eval_cp(comment: &str) -> Option<(f32, bool)> {
    // Format 1: Lichess PGN comment `[%eval 0.42]` or `[%eval #5]` (White POV)
    if let Some(idx) = comment.find("[%eval ") {
        let tail = &comment[idx + 7..];
        let end = tail.find(']')?;
        let val_str = tail[..end].trim();
        if let Some(m_str) = val_str.strip_prefix('#') {
            let mate_in = m_str.parse::<i32>().ok()?;
            return Some((if mate_in > 0 { 2500.0 } else { -2500.0 }, true));
        }
        let pawns = val_str.parse::<f32>().ok()?;
        return Some(((pawns * 100.0).clamp(-2500.0, 2500.0), true));
    }
    // Format 2: Fishtest / CuteChess / TCEC comment `{+1.84/19 1.7s}` or `{+M5/30 ...}` (STM POV)
    let first = comment.split_whitespace().next()?;
    if let Some((score_part, _depth_part)) = first.split_once('/') {
        let s = score_part.trim();
        if s.contains('M') || s.contains('m') {
            return Some((if s.starts_with('-') { -2500.0 } else { 2500.0 }, false));
        }
        if let Ok(pawns) = s.parse::<f32>() {
            return Some(((pawns * 100.0).clamp(-2500.0, 2500.0), false));
        }
    }
    None
}

/// Parses a multi-game PGN text string into structured games with [FEN] & eval comment support.
pub fn parse_pgn_text(text: &str) -> Vec<ParsedGame> {
    let mut games = Vec::new();
    let mut in_movetext = false;
    let mut white_result = 0.5f32;
    let mut start_fen: Option<String> = None;
    let mut move_buf = String::new();

    let flush_game = |buf: &str, res: f32, fen_opt: Option<String>, out: &mut Vec<ParsedGame>| {
        if buf.trim().is_empty() {
            return;
        }
        let mut pos = match &fen_opt {
            Some(f) => match Position::from_fen(f) {
                Ok(p) => p,
                Err(_) => return,
            },
            None => Position::startpos(),
        };
        let mut moves: Vec<(Move, Option<f32>)> = Vec::new();
        let mut mover_colors: Vec<saccadion_chess::types::Color> = Vec::new();
        let chars: Vec<char> = buf.chars().collect();
        let mut i = 0usize;
        let mut paren_depth = 0usize;

        while i < chars.len() {
            let c = chars[i];
            if c == '(' {
                paren_depth += 1;
                i += 1;
                continue;
            }
            if c == ')' {
                paren_depth = paren_depth.saturating_sub(1);
                i += 1;
                continue;
            }
            if paren_depth > 0 {
                i += 1;
                continue;
            }
            if c == '{' {
                let start = i + 1;
                while i < chars.len() && chars[i] != '}' {
                    i += 1;
                }
                let comment: String = chars[start..i].iter().collect();
                if let (Some((raw_cp, is_white_pov)), Some(last), Some(&mover_c)) = (
                    extract_eval_cp(&comment),
                    moves.last_mut(),
                    mover_colors.last(),
                ) {
                    let stm_cp = if is_white_pov {
                        if mover_c == saccadion_chess::types::Color::White {
                            raw_cp
                        } else {
                            -raw_cp
                        }
                    } else {
                        raw_cp
                    };
                    last.1 = Some(stm_cp);
                }
                i += 1;
                continue;
            }
            if c.is_whitespace() {
                i += 1;
                continue;
            }
            let start = i;
            while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '{' && chars[i] != '(' {
                i += 1;
            }
            let tok: String = chars[start..i].iter().collect();
            if tok.ends_with('.') || tok.contains("...") || tok.starts_with('$') {
                continue;
            }
            if matches!(tok.as_str(), "1-0" | "0-1" | "1/2-1/2" | "*") {
                break;
            }
            // Strip leading move number if attached like "1.e4"
            let san = tok.rsplit('.').next().unwrap_or(&tok);
            if san.is_empty() {
                continue;
            }
            let cur_stm = pos.side_to_move;
            if let Some(mv) = parse_san_move(&mut pos, san) {
                pos.make_move(mv);
                moves.push((mv, None));
                mover_colors.push(cur_stm);
            } else {
                break;
            }
        }
        if moves.len() >= 6 {
            out.push(ParsedGame {
                start_fen: fen_opt,
                white_result: res,
                moves,
            });
        }
    };

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if in_movetext {
                flush_game(&move_buf, white_result, start_fen.take(), &mut games);
                move_buf.clear();
                white_result = 0.5;
                in_movetext = false;
            }
            if let Some(rest) = trimmed.strip_prefix("[Result \"") {
                if rest.starts_with("1-0") {
                    white_result = 1.0;
                } else if rest.starts_with("0-1") {
                    white_result = 0.0;
                } else {
                    white_result = 0.5;
                }
            } else if let Some(rest) = trimmed.strip_prefix("[FEN \"") {
                if let Some(end_q) = rest.find('"') {
                    start_fen = Some(rest[..end_q].to_string());
                }
            }
        } else if !trimmed.is_empty() {
            in_movetext = true;
            move_buf.push_str(trimmed);
            move_buf.push(' ');
        }
    }
    if in_movetext {
        flush_game(&move_buf, white_result, start_fen.take(), &mut games);
    }

    games
}
