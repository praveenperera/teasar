//! Rules every backend's skeletoniser shares: the step rule, the radius rule, the PDRF in
//! its fixed float order, the invalidation reach, and the open choices
//!
//! The CPU reference and the Metal kernels evaluate these with the same float32 operations in the same order, which
//! is what makes their output bit-identical

use crate::{Error, Result};

/// A cost word that no relaxation has reached; above every float32 bit pattern a cost can take
pub const UNREACHED: u32 = u32::MAX;

/// The PDRF penalty scale of the TEASAR paper, fixed by both settings sets
pub const PDRF_SCALE: f32 = 5000.0;

/// The step rule: the cost of entering a voxel with weight `w` from a voxel of cost `cost` (bit patterns of
/// non-negative float32s, which order like their values)
///
/// Every step strictly raises the cost, even when `w` is lost to rounding or is 0 (joining S), so costs have a
/// unique least fixed point and predecessor walks always terminate
#[inline]
pub fn step(cost: u32, w: f32) -> u32 {
    let sum = (f32::from_bits(cost) + w).to_bits();
    sum.max(cost + 1)
}

/// The output radius of a voxel at squared pooled distance `n` from the boundary: `a * sqrt(n)` rounded once to
/// float32
///
/// Rounding the float64 product to float32 is exact here: `a` is a power of two, and double rounding a square root
/// from float64 to float32 is innocuous
#[inline]
pub fn radius(n: u32, a: f64) -> f32 {
    (a * f64::from(n).sqrt()) as f32
}

/// The float32 lengths of a 26-step that changes 1, 2 or 3 coordinates (index 0 is unused)
pub fn step_lengths(a: f64) -> [f32; 4] {
    [0.0, a as f32, (a * 2f64.sqrt()) as f32, (a * 3f64.sqrt()) as f32]
}

/// The PDRF in its fixed float32 order, with the powers of `u` flushed to zero below the normal range; `r_b` and `r_d`
/// are the piece's reciprocals of max B and max D, each rounded once ([`reciprocal`]). Every addition is inside an
/// explicit fused multiply-add and no other product feeds an addition, so a GPU compiler has nothing to contract and
/// every backend gets the same bits. Without `daf_term`, the penalty alone
#[inline]
pub fn pdrf(b: f32, r_b: f32, d: f32, r_d: f32, form: PenaltyForm, daf_term: bool) -> f32 {
    let t = b * r_b;
    let u = match form {
        PenaltyForm::Squared => (-t).mul_add(t, 1.0),
        PenaltyForm::Unsquared => (-b).mul_add(r_b, 1.0),
    };
    let u2 = ftz(u * u);
    let u4 = ftz(u2 * u2);
    let u8 = ftz(u4 * u4);
    let u16 = ftz(u8 * u8);
    if daf_term { PDRF_SCALE.mul_add(u16, d * r_d) } else { ftz(PDRF_SCALE * u16) }
}

/// `x` with a subnormal flushed to zero: Apple GPUs flush float32 subnormals, and `u^16` reaches them near the piece's
/// largest radius, so every backend flushes there to get the same bits
#[inline]
fn ftz(x: f32) -> f32 {
    if x.is_subnormal() { 0.0 } else { x }
}

/// `1 / x` rounded once to float32, for the PDRF's per-piece reciprocals
#[inline]
pub fn reciprocal(x: f32) -> f32 {
    1.0 / x
}

/// The invalidation ball of a path vertex: the voxels strictly inside radius
/// `r = fma(scale, dbf, const)`, tested in float32 as `a² × |Δ|² < r × r` over the bounding cube of half-width `⌊r / a⌋`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ball {
    /// the bounding cube's half-width in index units
    pub k: u32,
    rr: f32,
    a2: f32,
}

impl Ball {
    pub fn new(dbf: f32, scale: f32, constant: f32, a: f32) -> Self {
        let r = scale.mul_add(dbf, constant);
        // a is a power of two, so the division and a * a are exact
        Self { k: (r / a).floor() as u32, rr: r * r, a2: a * a }
    }

    /// Whether a voxel at squared index distance `d2` from the centre is strictly inside
    #[inline]
    pub fn contains(self, d2: u32) -> bool {
        self.a2 * (d2 as f32) < self.rr
    }
}

/// The penalty DBF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Penalty {
    /// `B = DBF`
    P1,
    /// DBF with the array border counted as boundary along every axis longer than 2 pooled voxels
    P2,
}

/// The penalty term of the PDRF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyForm {
    /// `(1 - (B / max B)^2)^16`, an alternative form
    Squared,
    /// The README's `(1 - B / max B)^16`, the default
    Unsquared,
}

/// The distances the invalidation radius `scale * r + const` uses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inval {
    /// The output DBF
    Dbf,
    /// The penalty DBF
    Penalty,
    /// The DBF with the array border counted along every axis of length 2 or more
    Thin,
}

/// The voxel the root search starts from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootStart {
    /// The piece's first voxel in C order
    COrder,
    /// The piece's first voxel in Fortran order
    FOrder,
    /// The piece's largest DBF, lowest C index among ties
    DbfMax,
    /// The first mandatory face target, when the piece has one; else as `COrder`
    Face,
    /// The last mandatory face target, when the piece has one; else as `COrder`
    FaceLast,
    /// The mandatory face target farthest from the piece's C-order first voxel; else as `COrder`
    FaceFar,
}

/// How a face region's mandatory target is picked among its 2D distance maxima
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacePick {
    /// The README's filters in turn: nearest the region's centroid, the plane's centre, the nearest plane corner and
    /// the nearest plane edge; then the highest C index
    Readme,
    /// Nearest the region's centroid, then the lowest C index
    Centroid,
    /// Nearest the region's centroid, then the highest C index
    CentroidHigh,
    /// The lowest C index
    Lowest,
}

/// The skeletoniser's open choices; [`Choices::default`] is the combination that reproduces the reference skeletons,
/// with the root starting at the first border target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choices {
    pub penalty: Penalty,
    pub penalty_form: PenaltyForm,
    pub root: RootStart,
    pub face_pick: FacePick,
    /// 8-connected face regions, else 4-connected
    pub face_8: bool,
    /// trace the mandatory targets before the ordinary ones
    pub mandatory_first: bool,
    /// count the target's own PDRF in the path cost
    pub count_target: bool,
    /// add `D / max D` to the PDRF; off leaves the penalty term alone
    pub daf_term: bool,
    /// invalidate only the ball voxels connected to its centre inside the ball, else every piece voxel in it
    pub ball_connected: bool,
    pub inval: Inval,
    /// face layers by axis x, y, z instead of z, y, x
    pub faces_xyz: bool,
    /// the N-1 layer of an axis before its 0 layer
    pub faces_high_first: bool,
}

impl Default for Choices {
    fn default() -> Self {
        Self {
            penalty: Penalty::P2,
            penalty_form: PenaltyForm::Unsquared,
            root: RootStart::Face,
            face_pick: FacePick::Readme,
            face_8: true,
            mandatory_first: true,
            count_target: false,
            daf_term: true,
            ball_connected: false,
            inval: Inval::Dbf,
            faces_xyz: false,
            faces_high_first: false,
        }
    }
}

impl Choices {
    /// The defaults changed by a comma-separated list such as `penalty=p1,root=f`, for comparing the open choices
    ///
    /// Keys: `penalty` (p1, p2), `form` (squared, unsquared), `root` (c, f, dbf, face, facelast, facefar), `face`
    /// (readme, centroid, centroidhigh, lowest), `regions` (8, 4), `mandatory` (first, last), `target` (free, counted),
    /// `daf` (on, off), `ballfill` (all, connected), `inval` (dbf, penalty, thin), `faces` (zyx-lo, zyx-hi, xyz-lo, xyz-hi)
    pub fn parse(spec: &str) -> Result<Self> {
        let mut c = Self::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let bad = || Error::Value(format!("unknown open choice {item:?}"));
            let (key, value) = item.split_once('=').ok_or_else(bad)?;
            match (key, value) {
                ("penalty", "p1") => c.penalty = Penalty::P1,
                ("penalty", "p2") => c.penalty = Penalty::P2,
                ("form", "squared") => c.penalty_form = PenaltyForm::Squared,
                ("form", "unsquared") => c.penalty_form = PenaltyForm::Unsquared,
                ("root", "c") => c.root = RootStart::COrder,
                ("root", "f") => c.root = RootStart::FOrder,
                ("root", "dbf") => c.root = RootStart::DbfMax,
                ("root", "face") => c.root = RootStart::Face,
                ("root", "facelast") => c.root = RootStart::FaceLast,
                ("root", "facefar") => c.root = RootStart::FaceFar,
                ("face", "readme") => c.face_pick = FacePick::Readme,
                ("face", "centroid") => c.face_pick = FacePick::Centroid,
                ("face", "centroidhigh") => c.face_pick = FacePick::CentroidHigh,
                ("face", "lowest") => c.face_pick = FacePick::Lowest,
                ("regions", "8") => c.face_8 = true,
                ("regions", "4") => c.face_8 = false,
                ("mandatory", "first") => c.mandatory_first = true,
                ("mandatory", "last") => c.mandatory_first = false,
                ("target", "free") => c.count_target = false,
                ("target", "counted") => c.count_target = true,
                ("daf", "on") => c.daf_term = true,
                ("daf", "off") => c.daf_term = false,
                ("ballfill", "all") => c.ball_connected = false,
                ("ballfill", "connected") => c.ball_connected = true,
                ("inval", "dbf") => c.inval = Inval::Dbf,
                ("inval", "penalty") => c.inval = Inval::Penalty,
                ("inval", "thin") => c.inval = Inval::Thin,
                ("faces", "zyx-lo") => (c.faces_xyz, c.faces_high_first) = (false, false),
                ("faces", "zyx-hi") => (c.faces_xyz, c.faces_high_first) = (false, true),
                ("faces", "xyz-lo") => (c.faces_xyz, c.faces_high_first) = (true, false),
                ("faces", "xyz-hi") => (c.faces_xyz, c.faces_high_first) = (true, true),
                _ => return Err(bad()),
            }
        }
        Ok(c)
    }

    /// The choices a call runs with: the defaults, or `TEASAR_CHOICES` in a `testing` build
    pub fn current() -> Result<Self> {
        #[cfg(feature = "testing")]
        if let Ok(spec) = std::env::var("TEASAR_CHOICES") {
            return Self::parse(&spec);
        }
        Ok(Self::default())
    }
}

/// One slab's fields for the field-level test: DBF, `B`, `D` and `P` as float32 bit patterns
/// indexed by slab voxel, with `UNREACHED` outside every traced piece, and each traced piece's root
#[cfg(feature = "testing")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDump {
    /// the output radius rule applied to the DBF's squared distance
    pub dbf: Vec<u32>,
    /// the penalty's boundary distance
    pub b: Vec<u32>,
    /// the geodesic distance from the root
    pub d: Vec<u32>,
    pub p: Vec<u32>,
    /// slab voxel indices, ascending
    pub roots: Vec<u32>,
}

#[cfg(feature = "testing")]
impl FieldDump {
    /// The dump of a slab of `n` voxels with no traced piece
    pub fn new(n: usize) -> Self {
        let none = vec![UNREACHED; n];
        Self { dbf: none.clone(), b: none.clone(), d: none.clone(), p: none, roots: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_rule_always_increases() {
        // a weight lost to rounding against a large cost still moves to the next float
        let big = 1.0e9f32.to_bits();
        assert_eq!(step(big, 1.0), big + 1);
        assert_eq!(step(0, 0.0), 1);
        assert_eq!(f32::from_bits(step(2.0f32.to_bits(), 0.5)), 2.5);
    }

    #[test]
    fn radius_is_correctly_rounded() {
        // an uncorrected float32 square root returns the wrong neighbour here
        let n = 2 * 2 + 7947 * 7947;
        let exact = (f64::from(n)).sqrt();
        let r = radius(n, 1.0);
        let below = f32::from_bits(r.to_bits() - 1);
        let above = f32::from_bits(r.to_bits() + 1);
        assert!((f64::from(r) - exact).abs() <= (f64::from(below) - exact).abs());
        assert!((f64::from(r) - exact).abs() <= (f64::from(above) - exact).abs());
        assert_eq!(radius(4, 2.0), 4.0);
    }

    #[test]
    fn ball_is_strict() {
        // the synthetic line-stub masks: DBF 2 at a = 2 gives r = 4, so a tip 2 voxels straight off the line (d2 = 4)
        // or at the knight offset (1, 2) pooled (d2 = 5) is outside, and one voxel off is inside
        let b = Ball::new(2.0, 1.0, 2.0, 2.0);
        assert_eq!(b.k, 2);
        assert!(b.contains(1) && b.contains(2) && b.contains(3));
        assert!(!b.contains(4) && !b.contains(5));
        // r / a not whole: half-width 2, and (2, 0, 0) is inside
        let b = Ball::new(2.828_427, 1.0, 2.0, 2.0);
        assert_eq!(b.k, 2);
        assert!(b.contains(4) && !b.contains(6));
    }

    #[test]
    fn choices_parse() {
        let c = Choices::parse("penalty=p1, root=f,mandatory=last,form=unsquared").unwrap();
        assert_eq!((c.penalty, c.root, c.mandatory_first), (Penalty::P1, RootStart::FOrder, false));
        assert_eq!(c.penalty_form, PenaltyForm::Unsquared);
        assert!(Choices::parse("penalty=p3").is_err());
    }
}
