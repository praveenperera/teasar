//! Compatibility mode, `target_order='kimimaro'` (spec addendum KM): the rules both backends share, and the two
//! orderings the mode takes from the running Python environment
//!
//! Every float operation here is one IEEE float32 operation with no fused multiply-add, as KM0 requires; the CPU
//! evaluates them without flushing subnormals

use std::sync::OnceLock;

use crate::Result;

/// The largest maximum boundary distance (physical units) compatibility mode traces (KM1); a call with a piece
/// above it raises `SomaUnsupported`
pub const SOMA_LIMIT: f32 = 1100.0;

/// The penalty multiplier of KM6
pub const PENALTY_SCALE: f32 = 5000.0;

/// The traversal order and the target order compatibility mode takes from villa's Python environment (KM5.1, KM7)
///
/// The Python layer implements it with CPython's built-in set and numpy's default argsort; nothing else may stand in
/// for them, since the reference's output depends on their exact orders
pub trait CompatOrders: Sync {
    /// For each pick list, the positions of its distinct picks in the iteration order of a fresh built-in set to which
    /// the picks are added in order, each as a tuple of three ints
    fn set_orders(&self, lists: &[&[[u32; 3]]]) -> Result<Vec<Vec<u32>>>;

    /// numpy's default ascending argsort of each list
    fn argsorts(&self, lists: &[&[f32]]) -> Result<Vec<Vec<u32>>>;
}

/// The order in which targets are traced (spec part A §2.4)
#[derive(Clone, Copy)]
pub enum TargetOrder<'a> {
    /// Part B's farthest-first order
    Best,
    /// The reference's order (addendum KM), with the orderings it takes from Python
    Kimimaro(&'a dyn CompatOrders),
}

impl std::fmt::Debug for TargetOrder<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            TargetOrder::Best => "Best",
            TargetOrder::Kimimaro(_) => "Kimimaro",
        })
    }
}

/// `Bmax^fl32(1.01)` as the reference platform computes it in float32 (KM6): the correctly rounded power, except at
/// the values of the addendum's table, where the platform's single-precision power is not correctly rounded
pub fn bmax_power(bmax: f32) -> f32 {
    let table = pow_table();
    match table.binary_search_by_key(&bmax.to_bits(), |&(b, _)| b) {
        Ok(k) => f32::from_bits(table[k].1),
        Err(_) => correctly_rounded_pow(bmax, 1.01f32),
    }
}

/// `x^e` rounded once to float32, through a float64 power whose error is far below half a float32 ulp
fn correctly_rounded_pow(x: f32, e: f32) -> f32 {
    f64::from(x).powf(f64::from(e)) as f32
}

/// The addendum's power table as sorted `(Bmax bits, power bits)` pairs
fn pow_table() -> &'static [(u32, u32)] {
    static TABLE: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let text = include_str!("../data/km6_pow.txt");
        let mut rows: Vec<(u32, u32)> = text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|l| {
                let (b, p) = l.split_once(' ').expect("two hex words per row");
                let hex = |s: &str| u32::from_str_radix(s, 16).expect("hex float32 bits");
                (hex(b), hex(p))
            })
            .collect();
        rows.sort_unstable();
        rows
    })
}

/// A piece's penalty field (KM6): `P` for a voxel of boundary distance `b` and geodesic distance `d`, given the
/// piece's `m = fl32(1 / pw)` and `r_d = fl32(1 / maxD)` (`None` when maxD is 0)
#[inline]
pub fn penalty(b: f32, m: f32, d: f32, r_d: Option<f32>) -> f32 {
    let u1 = 1.0 - b * m;
    let u2 = u1 * u1;
    let u4 = u2 * u2;
    let u8 = u4 * u4;
    let u16 = u8 * u8;
    let pen = u16 * PENALTY_SCALE;
    match r_d {
        Some(r) => pen + d * r,
        None => pen,
    }
}

/// The penalty's per-piece reciprocal of the platform power of `bmax`
pub fn penalty_m(bmax: f32) -> f32 {
    1.0 / bmax_power(bmax)
}

/// The float32 lengths of a 26-step changing 1, 2 or 3 coordinates (KM6; index 0 unused)
pub fn step_lengths(a: f32) -> [f32; 4] {
    let a2 = a * a;
    [0.0, a, (a2 + a2).sqrt(), (a2 + a2 + a2).sqrt()]
}

/// The claim radius of a path voxel (KM9): `fl32(fl32(scale × DBF) + const)`, which is `fl32(DBF + 2)` at villa's
/// settings
#[inline]
pub fn claim_radius(dbf: f32, scale: f32, constant: f32) -> f32 {
    scale * dbf + constant
}

/// The KM9 distance between voxels `d2` squared index units apart at pooling `a`; exact under the 2^24 the claim
/// radii keep it below
#[inline]
pub fn claim_dist(d2: [u32; 3], a: f32) -> f32 {
    let sq = |d: u32| {
        let x = a * d as f32;
        x * x
    };
    (sq(d2[0]) + sq(d2[1]) + sq(d2[2])).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_table_rows_differ_from_correct_rounding() {
        // the table lists exactly the values where the platform power is off by one ulp
        for &(b, p) in pow_table() {
            let cr = correctly_rounded_pow(f32::from_bits(b), 1.01);
            assert_eq!(cr.to_bits().abs_diff(p), 1, "Bmax {}", f32::from_bits(b));
        }
        assert_eq!(pow_table().len(), 2018);
    }

    /// On the reference platform, the C library's powf is the platform power the table describes, so it must equal
    /// `bmax_power` on every candidate value `sqrt32(fl32(a² n))` up to the soma limit
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn platform_power_matches_on_every_candidate() {
        for a in [2.0f32, 4.0] {
            let mut n = 0u32;
            loop {
                n += 1;
                let b = (a * a * n as f32).sqrt();
                if b > SOMA_LIMIT {
                    break;
                }
                assert_eq!(bmax_power(b).to_bits(), b.powf(1.01).to_bits(), "a {a} n {n}");
            }
        }
    }

    #[test]
    fn step_lengths_are_rounded_roots() {
        let w = step_lengths(2.0);
        assert_eq!(w[1], 2.0);
        assert_eq!(w[2], 8f32.sqrt());
        assert_eq!(w[3], 12f32.sqrt());
    }
}
