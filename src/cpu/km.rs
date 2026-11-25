//! Compatibility mode on the host: the reference's face picks, root and target orders, path and
//! claim rules, for one call's label arrays
//!
//! Both backends run this on their host threads; the Metal backend's prepare stages feed it. The slab-wide stages
//! (pieces, DBF) run for the whole call first, so `SomaUnsupported` is raised before any path search

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::Instant;

use rayon::prelude::*;

use super::edt::edt_except;
use super::pieces::{Pieces, pieces26, skeleton_ids};
use super::skel::Counts;
use crate::compat::{CompatOrders, SOMA_LIMIT, claim_dist, claim_radius, penalty, penalty_m, step_lengths};
use crate::{Error, LabelsRef, Packed, Result, SkelSettings};

/// Phase timers and counters for `TEASAR_PROFILE=1`: thread CPU nanoseconds summed over threads, the GIL's wait and
/// hold times around the Python orders, and the claims' and searches' heap traffic
pub(crate) mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const FIELDS: usize = 0;
    pub const FACES: usize = 1;
    pub const GRAPH: usize = 2;
    pub const GEODESIC: usize = 3;
    pub const ORDERS: usize = 4;
    pub const SEARCH: usize = 5;
    pub const CLAIM: usize = 6;
    #[cfg(feature = "python")]
    pub const GIL_WAIT: usize = 7;
    #[cfg(feature = "python")]
    pub const GIL_HELD: usize = 8;
    const NAMES: [&str; 9] = [
        "fields",
        "face picks",
        "graphs",
        "geodesics",
        "python orders",
        "searches",
        "claims",
        "GIL wait (wall)",
        "GIL held (wall)",
    ];
    static COUNTERS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

    pub const PUSHES: usize = 0;
    pub const SKIPPED: usize = 1;
    pub const POPS: usize = 2;
    pub const SEARCH_POPS: usize = 3;
    const COUNT_NAMES: [&str; 4] =
        ["claim pushes", "claim pushes skipped (not below the voxel's best key)", "claim pops", "search settles"];
    static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

    /// the call's longest single piece (prepare and finish) and slab, in wall nanoseconds: the serial chains under
    /// its wall
    pub const PIECE_MAX: usize = 0;
    pub const SLAB_MAX: usize = 1;
    static MAXES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

    pub fn on() -> bool {
        super::super::skel::prof::on()
    }

    /// The calling thread's CPU time in nanoseconds, so nested or descheduled work does not count
    fn cpu_ns() -> u64 {
        #[cfg(unix)]
        {
            let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
            // SAFETY: ts is a valid timespec for the call to fill
            unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
            ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
        }
        #[cfg(not(unix))]
        {
            // no thread clock off unix: wall time since the first call
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
        }
    }

    /// Runs `f`, adding the thread CPU time it took to counter `k` when profiling
    #[inline]
    pub fn time<T>(k: usize, f: impl FnOnce() -> T) -> T {
        if !on() {
            return f();
        }
        let t = cpu_ns();
        let out = f();
        COUNTERS[k].fetch_add(cpu_ns().saturating_sub(t), Ordering::Relaxed);
        out
    }

    /// Adds `secs` of wall time to counter `k`
    #[cfg(feature = "python")]
    pub fn add_wall(k: usize, secs: f64) {
        if on() {
            COUNTERS[k].fetch_add((secs * 1e9) as u64, Ordering::Relaxed);
        }
    }

    /// Adds `n` to count `k`
    #[inline]
    pub fn count(k: usize, n: u64) {
        COUNTS[k].fetch_add(n, Ordering::Relaxed);
    }

    pub fn max(k: usize, ns: u64) {
        MAXES[k].fetch_max(ns, Ordering::Relaxed);
    }

    /// Prints and zeroes the counters, with the call's wall
    pub fn print(slabs: usize, wall: f64) {
        if !on() {
            return;
        }
        let parts: Vec<String> = NAMES
            .iter()
            .zip(&COUNTERS)
            .map(|(n, c)| format!("{n} {:.2} s", c.swap(0, Ordering::Relaxed) as f64 * 1e-9))
            .collect();
        let counts: Vec<String> =
            COUNT_NAMES.iter().zip(&COUNTS).map(|(n, c)| format!("{n} {}", c.swap(0, Ordering::Relaxed))).collect();
        let [piece, slab] = [PIECE_MAX, SLAB_MAX].map(|k| MAXES[k].swap(0, Ordering::Relaxed) as f64 * 1e-9);
        eprintln!(
            "teasar kimimaro skeletonize: {slabs} slabs in {wall:.2} s (longest slab {slab:.2} s with its fields, longest \
             piece {piece:.2} s); thread CPU: {}; {}",
            parts.join(", "),
            counts.join(", ")
        );
    }
}

/// The 26 offsets `(Δi0, Δi1, Δi2)` in the join preference N
const JOIN_PREFERENCE: [[i8; 3]; 26] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
    [-1, -1, 0],
    [-1, 1, 0],
    [1, -1, 0],
    [1, 1, 0],
    [0, -1, -1],
    [0, -1, 1],
    [0, 1, -1],
    [0, 1, 1],
    [-1, 0, -1],
    [-1, 0, 1],
    [1, 0, -1],
    [1, 0, 1],
    [-1, -1, -1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1],
];

/// A slab's largest boundary distance, and its skeleton unless the call was refused first
type SlabRun = (f32, Option<Result<(Packed, Counts)>>);

/// The skeletons of each label array in compatibility mode, with the call's trace counts; `fix_borders(shape)` says
/// whether border targets are on for a slab of that pooled shape
pub fn skeletonize(
    labels: &[LabelsRef<'_>],
    skel: &SkelSettings,
    fix_borders: impl Fn([usize; 3]) -> bool + Sync,
    orders: &dyn CompatOrders,
) -> Result<(Vec<Packed>, Counts)> {
    let a = skel.anisotropy[0] as f32;
    let start = Instant::now();
    // each slab runs its fields and then its traces as one task, so no phase barrier idles the threads. The soma limit
    // refuses the whole call, naming the first slab over the limit: every slab still computes its fields, and once any
    // is over, no slab starts its traces. The traces in flight are discarded, so the outcome is the refusal alone
    let refused = AtomicBool::new(false);
    let out: Vec<SlabRun> = labels
        .par_iter()
        .map(|l| {
            let t = Instant::now();
            let f = prof::time(prof::FIELDS, || Fields::new(l, a));
            let max_dbf = f.max_dbf;
            if max_dbf > SOMA_LIMIT {
                refused.store(true, AtomicOrdering::Relaxed);
            }
            if refused.load(AtomicOrdering::Relaxed) {
                return (max_dbf, None);
            }
            let out = Slab::new(l, skel, f, fix_borders(l.shape)).run(orders);
            if prof::on() {
                prof::max(prof::SLAB_MAX, t.elapsed().as_nanos() as u64);
            }
            (max_dbf, Some(out))
        })
        .collect();
    if let Some(slab) = out.iter().position(|&(m, _)| m > SOMA_LIMIT) {
        return Err(Error::SomaUnsupported {
            slab,
            message: format!(
                "a piece's largest boundary distance {} exceeds {SOMA_LIMIT}, the limit of target_order='kimimaro'",
                out[slab].0
            ),
        });
    }

    let mut counts = Counts::default();
    let mut packed = Vec::with_capacity(out.len());
    for (_, r) in out {
        let (p, c) = r.expect("with no slab over the limit, every slab ran its traces")?;
        counts += c;
        packed.push(p);
    }
    prof::print(labels.len(), start.elapsed().as_secs_f64());
    Ok((packed, counts))
}

/// A slab's pieces and boundary distances
struct Fields {
    pieces: Pieces,
    /// squared pooled distance to the nearest voxel outside the voxel's piece
    n_dbf: Vec<u32>,
    max_dbf: f32,
}

impl Fields {
    fn new(labels: &LabelsRef<'_>, a: f32) -> Self {
        let pieces = pieces26(labels);
        // the border is boundary only when one nonzero label fills the array, which is then one piece
        let n = labels.data.len();
        let full = pieces.pieces.len() == 1 && pieces.pieces[0].voxels as usize == n;
        // background distances are never read
        let n_dbf = edt_except(&pieces.of_voxel, labels.shape, [full; 3], Some(u32::MAX));
        let max_n = pieces.of_voxel.iter().zip(&n_dbf).filter(|&(&p, _)| p != u32::MAX).map(|(_, &d)| d).max();
        let max_dbf = max_n.map_or(0.0, |d| dbf(d, a));
        Self { pieces, n_dbf, max_dbf }
    }
}

/// The DBF of a voxel at squared pooled distance `n`: `sqrt32(fl32(a² n))`
#[inline]
fn dbf(n: u32, a: f32) -> f32 {
    (a * a * n as f32).sqrt()
}

/// One slab's stages after the fields: voxel lists, face picks and the traces of its kept pieces
struct Slab<'a> {
    labels: &'a LabelsRef<'a>,
    skel: &'a SkelSettings,
    a: f32,
    fields: Fields,
    /// voxels of piece p, in C order: `voxels[offs[p]..offs[p + 1]]`
    offs: Vec<u32>,
    voxels: Vec<u32>,
    /// each voxel's index within its piece's list
    local: Vec<u32>,
    /// each piece's face picks in the order of `face_picks`, as full-array coordinates; empty with border targets off
    picks: Vec<Vec<[u32; 3]>>,
}

impl<'a> Slab<'a> {
    fn new(labels: &'a LabelsRef<'a>, skel: &'a SkelSettings, fields: Fields, borders: bool) -> Self {
        let a = skel.anisotropy[0] as f32;
        let n = labels.data.len();
        let np = fields.pieces.pieces.len();
        let mut offs = vec![0u32; np + 1];
        for (k, p) in fields.pieces.pieces.iter().enumerate() {
            offs[k + 1] = offs[k] + p.voxels;
        }
        let mut fill = offs.clone();
        let mut voxels = vec![0u32; offs[np] as usize];
        let mut local = vec![u32::MAX; n];
        for (g, &p) in fields.pieces.of_voxel.iter().enumerate() {
            if p == u32::MAX {
                continue;
            }
            let slot = &mut fill[p as usize];
            local[g] = *slot - offs[p as usize];
            voxels[*slot as usize] = g as u32;
            *slot += 1;
        }

        let picks = if borders {
            prof::time(prof::FACES, || face_picks(&fields.pieces.of_voxel, labels.shape, a, np))
        } else {
            vec![Vec::new(); np]
        };
        Self { labels, skel, a, fields, offs, voxels, local, picks }
    }

    fn run(&self, orders: &dyn CompatOrders) -> Result<(Packed, Counts)> {
        let pieces = &self.fields.pieces;
        let ids = skeleton_ids(self.labels, pieces, self.skel.dust_threshold);
        if ids.is_empty() {
            return Ok((Packed::empty(), Counts::default()));
        }

        let kept: Vec<u32> = (0..pieces.pieces.len() as u32)
            .filter(|&p| pieces.pieces[p as usize].voxels > self.skel.dust_threshold)
            .collect();
        // the orders come from Python once per slab for all its pieces: the GIL is taken twice per slab, not per piece
        let picked: Vec<usize> = (0..kept.len()).filter(|&k| !self.picks[kept[k] as usize].is_empty()).collect();
        let lists: Vec<&[[u32; 3]]> = picked.iter().map(|&k| self.picks[kept[k] as usize].as_slice()).collect();
        let set_orders = prof::time(prof::ORDERS, || orders.set_orders(&lists))?;
        if set_orders.len() != lists.len() {
            return Err(Error::Value(format!("set orders: {} lists for {} pieces", set_orders.len(), lists.len())));
        }
        let mut order_of: Vec<Option<Vec<u32>>> = vec![None; kept.len()];
        for (k, o) in picked.into_iter().zip(set_orders) {
            order_of[k] = Some(o);
        }

        let preps: Vec<Prepared> =
            kept.par_iter().zip(order_of).map(|(&p, o)| self.piece(p).prepare(o.as_deref())).collect();
        let sorted = {
            let d: Vec<&[f32]> = preps.iter().map(|p| p.d_f.as_slice()).collect();
            prof::time(prof::ORDERS, || orders.argsorts(&d))?
        };
        if sorted.len() != preps.len() || sorted.iter().zip(&preps).any(|(s, p)| s.len() != p.d_f.len()) {
            return Err(Error::Value("argsort returned the wrong number of indices".into()));
        }
        let skels: Vec<PieceOut> = kept
            .par_iter()
            .zip(preps)
            .zip(sorted)
            .map(|((&p, prep), sorted)| self.piece(p).finish(prep, &sorted))
            .collect();

        // each label's pieces merged and sorted, labels in skeleton order
        let mut by_label: Vec<(u32, usize)> =
            kept.iter().enumerate().map(|(k, &p)| (pieces.pieces[p as usize].label, k)).collect();
        by_label.sort_unstable();
        let mut out =
            Packed { ids: Vec::with_capacity(ids.len()), v_off: vec![0], e_off: vec![0], ..Packed::default() };
        let mut counts = Counts::default();
        let (mut verts, mut edges) = (Vec::new(), Vec::new());
        for &label in &ids {
            verts.clear();
            edges.clear();
            let lo = by_label.partition_point(|&(l, _)| l < label);
            for &(_, k) in by_label[lo..].iter().take_while(|&&(l, _)| l == label) {
                let s = &skels[k];
                verts.extend(s.verts.iter().map(|&g| (g, dbf(self.fields.n_dbf[g as usize], self.a))));
                edges.extend_from_slice(&s.edges);
                counts += s.counts;
            }
            out.push_sorted(label, &mut verts, &mut edges, self.labels.shape, self.skel.anisotropy);
        }
        Ok((out, counts))
    }

    fn piece(&self, pid: u32) -> Piece<'_> {
        let vox = &self.voxels[self.offs[pid as usize] as usize..self.offs[pid as usize + 1] as usize];
        Piece { slab: self, me: pid, vox }
    }
}

/// A piece between its fields and its traces: the root, the mandatory targets, and D in F order with that order
struct Prepared<'s> {
    root: usize,
    mandatory: Vec<usize>,
    /// D, by list entry and in F order (for the argsort)
    d: Vec<f32>,
    d_f: Vec<f32>,
    forder: Vec<u32>,
    /// the piece's coordinates, graph and DBF, kept for the traces
    coords: Vec<[u32; 3]>,
    graph: Graph<'s>,
    dbf: Vec<f32>,
    busy_ns: u64,
}

/// One piece's skeleton: slab-global vertex voxels and edges between them
struct PieceOut {
    verts: Vec<u32>,
    edges: Vec<[u32; 2]>,
    counts: Counts,
}

/// One piece of a slab
struct Piece<'s> {
    slab: &'s Slab<'s>,
    me: u32,
    /// the piece's voxels in C order
    vox: &'s [u32],
}

impl<'s> Piece<'s> {
    fn coords(&self, g: usize) -> [usize; 3] {
        let [_, y, x] = self.slab.labels.shape;
        [g / (y * x), (g / x) % y, g % x]
    }

    /// The piece-local index of the voxel at `c`, if it is in the array and the piece
    fn at(&self, c: [i64; 3]) -> Option<usize> {
        let [z, y, x] = self.slab.labels.shape.map(|s| s as i64);
        if c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= z || c[1] >= y || c[2] >= x {
            return None;
        }
        let g = ((c[0] * y + c[1]) * x + c[2]) as usize;
        (self.slab.fields.pieces.of_voxel[g] == self.me).then(|| self.slab.local[g] as usize)
    }

    /// The piece's coordinates, F-indices, graph and DBF
    fn basics(&self) -> (Vec<[u32; 3]>, Vec<u32>, Graph<'s>, Vec<f32>) {
        let [s0, s1, _] = self.slab.labels.shape;
        let coords: Vec<[u32; 3]> = self.vox.iter().map(|&g| self.coords(g as usize).map(|c| c as u32)).collect();
        let fidx: Vec<u32> = coords.iter().map(|c| c[0] + s0 as u32 * (c[1] + s1 as u32 * c[2])).collect();
        let graph = prof::time(prof::GRAPH, || self.graph(&coords));
        let dbf: Vec<f32> = self.vox.iter().map(|&g| dbf(self.slab.fields.n_dbf[g as usize], self.slab.a)).collect();
        (coords, fidx, graph, dbf)
    }

    /// The root, mandatory targets and geodesic distance D, given the set order of the piece's picks (`None` for a
    /// piece with none)
    fn prepare(&self, set_order: Option<&[u32]>) -> Prepared<'s> {
        let t = Instant::now();
        let m = self.vox.len();
        let (coords, fidx, graph, dbf) = self.basics();
        let lengths = step_lengths(self.slab.a);
        let far = |d: &[f32]| far_point(d, &fidx);

        let picks = &self.slab.picks[self.me as usize];
        let (root, mandatory) = match set_order {
            None => {
                let lowest = (0..m).min_by_key(|&i| fidx[i]).expect("a kept piece has voxels");
                let root = far(&prof::time(prof::GEODESIC, || graph.geodesic(lowest, &lengths)));
                (root, Vec::new())
            }
            Some(order) => {
                let to_local = |k: u32| {
                    let c = picks[k as usize].map(i64::from);
                    self.at(c).expect("a face pick is a voxel of its piece")
                };
                let l: Vec<usize> = order.iter().map(|&k| to_local(k)).collect();
                let (&root, rest) = l.split_last().expect("a set of at least one pick");
                (root, rest.iter().rev().copied().collect())
            }
        };
        let d = prof::time(prof::GEODESIC, || graph.geodesic(root, &lengths));
        // the far-point target, when no pick remains after the root
        let mandatory = if mandatory.is_empty() { vec![far(&d)] } else { mandatory };

        // the ordinary targets sort D taken in F order
        let mut forder: Vec<u32> = (0..m as u32).collect();
        forder.sort_unstable_by_key(|&i| fidx[i as usize]);
        let d_f: Vec<f32> = forder.iter().map(|&i| d[i as usize]).collect();
        let busy_ns = t.elapsed().as_nanos() as u64;
        Prepared { root, mandatory, d, d_f, forder, coords, graph, dbf, busy_ns }
    }

    /// The penalty and the traces, given numpy's argsort of the prepared D list
    fn finish(&self, prep: Prepared<'_>, sorted: &[u32]) -> PieceOut {
        let t = Instant::now();
        let m = self.vox.len();
        let (coords, graph, dbf, d) = (&prep.coords, &prep.graph, &prep.dbf, &prep.d);

        // the penalty
        let bmax = dbf.iter().copied().fold(0.0f32, f32::max);
        let max_d = d.iter().copied().fold(0.0f32, f32::max);
        let r_d = (max_d != 0.0).then(|| 1.0 / max_d);
        let pm = penalty_m(bmax);
        let p: Vec<f32> = (0..m).map(|i| penalty(dbf[i], pm, d[i], r_d)).collect();

        // ordinary targets in the reversed argsort
        let ordinary: Vec<u32> = sorted.iter().rev().map(|&k| prep.forder[k as usize]).collect();
        let (scale, constant) = (self.slab.skel.scale as f32, self.slab.skel.constant as f32);
        let radius: Vec<f32> = dbf.iter().map(|&b| claim_radius(b, scale, constant)).collect();
        let mut run = Run::new(self, graph, coords, &prep.forder, &p, &radius, prep.root);
        for &t in &prep.mandatory {
            run.trace(t);
        }
        let mut traced = 0u64;
        for &t in &ordinary {
            // a trace never makes a voxel valid again, so one pass over R finds every ordinary target in turn
            while run.valid[t as usize] {
                run.trace(t as usize);
                traced += 1;
            }
        }

        let verts: Vec<u32> = (0..m).filter(|&i| run.vertex[i]).map(|i| self.vox[i]).collect();
        let edges = run.edges.iter().map(|&[u, v]| [self.vox[u as usize], self.vox[v as usize]]).collect();
        let busy_ns = prep.busy_ns + t.elapsed().as_nanos() as u64;
        if prof::on() {
            prof::max(prof::PIECE_MAX, busy_ns);
        }
        let counts =
            Counts { pieces: 1, ordinary: traced, mandatory: prep.mandatory.len() as u64, voxels: m as u64, busy_ns };
        PieceOut { verts, edges, counts }
    }

    /// The piece's 26-neighbour lists, each entry `local << 2 | coordinates changed`
    fn graph(&self, coords: &[[u32; 3]]) -> Graph<'s> {
        let shape = self.slab.labels.shape;
        let [z, y, x] = shape.map(|s| s as u32);
        let of_voxel = &self.slab.fields.pieces.of_voxel;
        let doff = neighbour_offsets(shape);
        let mut mask = Vec::with_capacity(coords.len());
        for (&g, c) in self.vox.iter().zip(coords) {
            let g = g as isize;
            let inside = |k: usize| of_voxel[(g + doff[k]) as usize] == self.me;
            let interior = c[0] > 0 && c[1] > 0 && c[2] > 0 && c[0] + 1 < z && c[1] + 1 < y && c[2] + 1 < x;
            let mut m = 0u32;
            if interior {
                // every neighbour is in the array, so only the piece test remains
                for k in 0..26 {
                    m |= u32::from(inside(k)) << k;
                }
            } else {
                for (k, step) in STEPS.iter().enumerate() {
                    let in_array = (0..3).all(|a| {
                        let n = i64::from(c[a]) + i64::from(step[a]);
                        n >= 0 && n < shape[a] as i64
                    });
                    m |= u32::from(in_array && inside(k)) << k;
                }
            }
            mask.push(m);
        }
        Graph { mask, vox: self.vox, local: &self.slab.local, doff }
    }
}

/// The 26 neighbour steps (dz, dy, dx) in the graph's neighbour order: dz, then dy, then dx, each from -1 to 1
const STEPS: [[i8; 3]; 26] = {
    let mut out = [[0i8; 3]; 26];
    let mut k = 0;
    let mut i = 0;
    while i < 27 {
        let step = [(i / 9) as i8 - 1, ((i / 3) % 3) as i8 - 1, (i % 3) as i8 - 1];
        if i != 13 {
            out[k] = step;
            k += 1;
        }
        i += 1;
    }
    out
};

/// Each neighbour step's kind: how many axes it moves along (1 face, 2 edge, 3 corner)
const KINDS: [u8; 26] = {
    let mut out = [0u8; 26];
    let mut k = 0;
    while k < 26 {
        out[k] = (STEPS[k][0] != 0) as u8 + (STEPS[k][1] != 0) as u8 + (STEPS[k][2] != 0) as u8;
        k += 1;
    }
    out
};

/// The C-order index offset of each neighbour step in an array of `shape`
fn neighbour_offsets(shape: [usize; 3]) -> [isize; 26] {
    let (sz, sy) = ((shape[1] * shape[2]) as isize, shape[2] as isize);
    STEPS.map(|[dz, dy, dx]| dz as isize * sz + dy as isize * sy + dx as isize)
}

/// The voxel of largest `d`, lowest F-index among ties
fn far_point(d: &[f32], fidx: &[u32]) -> usize {
    let mut best = 0;
    for i in 1..d.len() {
        if d[i] > d[best] || (d[i] == d[best] && fidx[i] < fidx[best]) {
            best = i;
        }
    }
    best
}

/// A piece's voxel graph: each voxel's neighbours in the piece as a mask over `STEPS`, resolved through the slab's
/// piece-local index; 4 bytes a voxel, so every piece of a slab can keep its graph from the root search to the traces
struct Graph<'s> {
    mask: Vec<u32>,
    /// the piece's voxels (C-order indices) and the slab's voxel-to-list-entry map
    vox: &'s [u32],
    local: &'s [u32],
    doff: [isize; 26],
}

impl Graph<'_> {
    /// The neighbours of list entry `i` as (list entry, step kind), in `STEPS` order
    #[inline]
    fn neighbours(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let g = self.vox[i] as isize;
        let mut m = self.mask[i];
        std::iter::from_fn(move || {
            if m == 0 {
                return None;
            }
            let k = m.trailing_zeros() as usize;
            m &= m - 1;
            Some((self.local[(g + self.doff[k]) as usize] as usize, KINDS[k] as usize))
        })
    }

    /// The geodesic distance D from `src`, every step a plain float32 addition of its length
    fn geodesic(&self, src: usize, lengths: &[f32; 4]) -> Vec<f32> {
        let m = self.mask.len();
        let mut cost = vec![f32::INFINITY; m];
        // keys are (cost bits, list entry) in one u64; non-negative float bits order like the floats
        let key = |c: f32, i: usize| (u64::from(c.to_bits()) << 32) | i as u64;
        let mut heap = BinaryHeap::new();
        cost[src] = 0.0;
        heap.push(Reverse(key(0.0, src)));
        while let Some(Reverse(k)) = heap.pop() {
            let i = (k & 0xFFFF_FFFF) as usize;
            if (k >> 32) as u32 != cost[i].to_bits() {
                continue;
            }
            let c = cost[i];
            for (j, kind) in self.neighbours(i) {
                let nc = c + lengths[kind];
                if nc < cost[j] {
                    cost[j] = nc;
                    heap.push(Reverse(key(nc, j)));
                }
            }
        }
        cost
    }
}

/// The trace loop's state for one piece
struct Run<'r> {
    piece: &'r Piece<'r>,
    graph: &'r Graph<'r>,
    coords: &'r [[u32; 3]],
    /// the piece's list entries in F order, and each entry's rank in it: ranks order like F-indices, so the heaps
    /// key on (value bits, rank) in one u64 and read the voxel back from `forder`
    forder: &'r [u32],
    frank: Vec<u32>,
    p: &'r [f32],
    radius: &'r [f32],
    valid: Vec<bool>,
    in_s: Vec<bool>,
    vertex: Vec<bool>,
    edges: Vec<[u32; 2]>,
    /// path-search state, reset through `touched`
    cost: Vec<f32>,
    settled: Vec<bool>,
    hops: Vec<u32>,
    touched: Vec<usize>,
    /// marks the settled voxels with a cost-tight path to p* during one search's backtrack
    toward: Vec<bool>,
    /// claim state: each voxel's best (d bits, path position) in the current claim, reset through `claim_touched`,
    /// and the claim heap of (d bits, F rank), kept for its allocation
    claim_best: Vec<u64>,
    claim_touched: Vec<u32>,
    claim_heap: BinaryHeap<Reverse<u64>>,
    /// the search heap of (cost bits, F rank), kept for its allocation
    search_heap: BinaryHeap<Reverse<u64>>,
}

impl<'r> Run<'r> {
    fn new(
        piece: &'r Piece<'r>,
        graph: &'r Graph<'r>,
        coords: &'r [[u32; 3]],
        forder: &'r [u32],
        p: &'r [f32],
        radius: &'r [f32],
        root: usize,
    ) -> Self {
        let m = coords.len();
        let mut frank = vec![0u32; m];
        for (r, &i) in forder.iter().enumerate() {
            frank[i as usize] = r as u32;
        }
        let mut in_s = vec![false; m];
        // the root starts in S, but valid: claims run only after a trace
        in_s[root] = true;
        Self {
            piece,
            graph,
            coords,
            forder,
            frank,
            p,
            radius,
            valid: vec![true; m],
            in_s,
            vertex: vec![false; m],
            edges: Vec::new(),
            cost: vec![f32::INFINITY; m],
            settled: vec![false; m],
            hops: vec![u32::MAX; m],
            touched: Vec::new(),
            toward: vec![false; m],
            claim_best: vec![u64::MAX; m],
            claim_touched: Vec::new(),
            claim_heap: BinaryHeap::new(),
            search_heap: BinaryHeap::new(),
        }
    }

    fn trace(&mut self, t: usize) {
        let path = if self.in_s[t] { vec![t] } else { prof::time(prof::SEARCH, || self.search(t)) };
        for &v in &path {
            self.in_s[v] = true;
        }
        if path.len() >= 2 {
            for &v in &path {
                self.vertex[v] = true;
            }
            for w in path.windows(2) {
                self.edges.push([w[0] as u32, w[1] as u32]);
            }
        }
        prof::time(prof::CLAIM, || self.claim(&path));
    }

    /// The path from `t` (outside S) to its join in S, as `t, …, p*, j`
    fn search(&mut self, t: usize) -> Vec<usize> {
        for &i in &self.touched {
            self.cost[i] = f32::INFINITY;
            self.settled[i] = false;
            self.hops[i] = u32::MAX;
        }
        self.touched.clear();

        // keys are (cost bits, F rank), unique per voxel, so the pop order is the (cost, F-index) order of the search
        let key = |c: f32, f: u32| (u64::from(c.to_bits()) << 32) | u64::from(f);
        let mut heap = std::mem::take(&mut self.search_heap);
        heap.clear();
        self.cost[t] = 0.0;
        self.touched.push(t);
        heap.push(Reverse(key(0.0, self.frank[t])));
        // the join candidate of smallest (cost, F-index), as its key
        let mut best: Option<(u64, usize)> = None;
        let mut settled = 0u64;
        while let Some(&Reverse(k)) = heap.peek() {
            if best.is_some_and(|(bk, _)| k >> 32 > bk >> 32) {
                break;
            }
            heap.pop();
            let i = self.forder[(k & 0xFFFF_FFFF) as usize] as usize;
            if self.settled[i] || (k >> 32) as u32 != self.cost[i].to_bits() {
                continue;
            }
            self.settled[i] = true;
            settled += 1;
            let ci = self.cost[i];
            let mut joins = false;
            for (j, _) in self.graph.neighbours(i) {
                if self.in_s[j] {
                    joins = true;
                    continue;
                }
                if self.settled[j] {
                    continue;
                }
                let nc = ci + self.p[j];
                if nc < self.cost[j] {
                    if self.cost[j] == f32::INFINITY {
                        self.touched.push(j);
                    }
                    self.cost[j] = nc;
                    heap.push(Reverse(key(nc, self.frank[j])));
                }
            }
            if joins && best.is_none_or(|(bk, _)| k < bk) {
                best = Some((k, i));
            }
        }
        self.search_heap = heap;
        if prof::on() {
            prof::count(prof::SEARCH_POPS, settled);
        }
        let (_, pstar) = best.expect("a connected piece's search reaches S");

        // the settled voxels (all those of cost up to c(p*)) with a cost-tight path to p*. The backtrack reads hop
        // counts only on these, and every tight path from t to one of them stays among them, so counting hops within
        // them gives the same counts as over every settled voxel
        self.toward[pstar] = true;
        let mut toward = vec![pstar];
        let mut k = 0;
        while k < toward.len() {
            let v = toward[k];
            k += 1;
            for (u, _) in self.graph.neighbours(v) {
                if self.settled[u] && !self.toward[u] && self.tight(u, v) {
                    self.toward[u] = true;
                    toward.push(u);
                }
            }
        }

        // hop counts over cost-tight steps from t
        self.hops[t] = 0;
        let mut queue = VecDeque::from([t]);
        while let Some(u) = queue.pop_front() {
            for (v, _) in self.graph.neighbours(u) {
                if self.toward[v] && self.hops[v] == u32::MAX && self.tight(u, v) {
                    self.hops[v] = self.hops[u] + 1;
                    queue.push_back(v);
                }
            }
        }

        let mut path = vec![pstar];
        let mut v = pstar;
        while v != t {
            v = self
                .graph
                .neighbours(v)
                .map(|(u, _)| u)
                .filter(|&u| self.settled[u] && self.hops[u] != u32::MAX && self.tight(u, v))
                .min_by(|&x, &y| {
                    (self.hops[x], self.cost[x], self.frank[x])
                        .partial_cmp(&(self.hops[y], self.cost[y], self.frank[y]))
                        .unwrap_or(Ordering::Equal)
                })
                .expect("every settled voxel but the target has a cost-tight predecessor");
            path.push(v);
        }
        for v in toward {
            self.toward[v] = false;
        }
        path.reverse();
        path.push(self.join(pstar));
        path
    }

    /// Whether the step from `u` to `v` reproduces `v`'s cost
    #[inline]
    fn tight(&self, u: usize, v: usize) -> bool {
        self.cost[u] + self.p[v] == self.cost[v]
    }

    /// The S-neighbour of `q` first in the join preference N
    fn join(&self, q: usize) -> usize {
        let c = self.coords[q].map(i64::from);
        JOIN_PREFERENCE
            .iter()
            .find_map(|o| {
                self.piece.at([c[0] + o[0] as i64, c[1] + o[1] as i64, c[2] + o[2] as i64]).filter(|&j| self.in_s[j])
            })
            .expect("p* has a neighbour in S")
    }

    /// The path's valid voxels claim the valid voxels around them, nearest claim first
    ///
    /// Claims pop in (d, F-index, path position) order and a voxel is claimed by its first pop, so only its smallest
    /// (d, position) can win: a push that does not beat the voxel's best key so far is skipped. The heap then orders
    /// by (d, F-index), unique per voxel, and the winner's position comes from the best key
    fn claim(&mut self, path: &[usize]) {
        let a = self.piece.slab.a;
        let (mut pushes, mut skipped, mut pops) = (0u64, 0u64, 0u64);
        let mut heap = std::mem::take(&mut self.claim_heap);
        for (pos, &s) in path.iter().enumerate() {
            if self.valid[s] && self.offer(s, 0, pos as u32, &mut heap) {
                pushes += 1;
            }
        }
        while let Some(Reverse(key)) = heap.pop() {
            pops += 1;
            let q = self.forder[(key & 0xFFFF_FFFF) as usize] as usize;
            if !self.valid[q] {
                continue;
            }
            self.valid[q] = false;
            debug_assert_eq!(self.claim_best[q] >> 32, key >> 32, "a voxel's first pop carries its best d");
            let pos = (self.claim_best[q] & 0xFFFF_FFFF) as u32;
            let s = path[pos as usize];
            let (cs, r) = (self.coords[s], self.radius[s]);
            for (j, _) in self.graph.neighbours(q) {
                if !self.valid[j] {
                    continue;
                }
                let cj = self.coords[j];
                let d = claim_dist([0, 1, 2].map(|k| cs[k].abs_diff(cj[k])), a);
                if d < r {
                    if self.offer(j, d.to_bits(), pos, &mut heap) {
                        pushes += 1;
                    } else {
                        skipped += 1;
                    }
                }
            }
        }
        self.claim_heap = heap;
        for &v in &self.claim_touched {
            self.claim_best[v as usize] = u64::MAX;
        }
        self.claim_touched.clear();
        if prof::on() {
            prof::count(prof::PUSHES, pushes);
            prof::count(prof::SKIPPED, skipped);
            prof::count(prof::POPS, pops);
        }
    }

    /// Pushes the claim of `j` at distance bits `d` by the source at path position `pos` when it beats `j`'s best
    #[inline]
    fn offer(&mut self, j: usize, d: u32, pos: u32, heap: &mut BinaryHeap<Reverse<u64>>) -> bool {
        let dp = (u64::from(d) << 32) | u64::from(pos);
        let best = &mut self.claim_best[j];
        if dp >= *best {
            return false;
        }
        if *best == u64::MAX {
            self.claim_touched.push(j as u32);
        }
        *best = dp;
        heap.push(Reverse((u64::from(d) << 32) | u64::from(self.frank[j])));
        true
    }
}

/// Every piece's face picks, in pick order (faces in order, then regions by their first voxel's plane F-index),
/// as full-array coordinates; picks may repeat across faces
fn face_picks(of_voxel: &[u32], shape: [usize; 3], a: f32, pieces: usize) -> Vec<Vec<[u32; 3]>> {
    let mut out = vec![Vec::new(); pieces];
    for axis in [2, 1, 0] {
        let mut idx = vec![0, shape[axis] - 1];
        // an axis of length 1 has one face layer, whose picks count once
        idx.dedup();
        for index in idx {
            for (pid, pick) in face_layer_picks(of_voxel, shape, a, axis, index) {
                out[pid as usize].push(pick);
            }
        }
    }
    out
}

/// One face region's running sums and best voxel
struct Region {
    piece: u32,
    su: f32,
    sv: f32,
    cnt: u32,
    /// the largest squared plane distance among the region's voxels
    best_n: u32,
    first_f: usize,
}

/// The picks of one face layer, one per region, as `(piece, coordinates)` in increasing plane F-index of each
/// region's first voxel
fn face_layer_picks(of_voxel: &[u32], shape: [usize; 3], a: f32, axis: usize, index: usize) -> Vec<(u32, [u32; 3])> {
    let (u_ax, v_ax) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let (p0, p1) = (shape[u_ax], shape[v_ax]);
    let full = |u: usize, v: usize| {
        let mut c = [0usize; 3];
        c[axis] = index;
        c[u_ax] = u;
        c[v_ax] = v;
        c
    };
    let ids: Vec<u32> = (0..p0 * p1)
        .map(|k| {
            let c = full(k / p1, k % p1);
            of_voxel[(c[0] * shape[1] + c[1]) * shape[2] + c[2]]
        })
        .collect();
    // positions off the plane count as outside every region
    let m2 = edt_except(&ids, [1, p0, p1], [false, true, true], Some(u32::MAX));

    // 8-connected regions of one piece, merged in C order (v fastest); each set's root is its smallest index.
    // Earlier voxels were merged with their own earlier neighbours, so a matching up-neighbour already joins the
    // up-left and up-right ones, and a matching left one the up-left one
    let mut parent: Vec<usize> = (0..ids.len()).collect();
    let union = |parent: &mut [usize], k: usize, j: usize| {
        let (ra, rb) = (find(parent, k), find(parent, j));
        parent[ra.max(rb)] = ra.min(rb);
    };
    for u in 0..p0 {
        for v in 0..p1 {
            let k = u * p1 + v;
            let id = ids[k];
            if id == u32::MAX {
                continue;
            }
            let same = |j: Option<usize>| j.filter(|&j| ids[j] == id);
            let up = (u > 0).then(|| k - p1);
            let left = (v > 0).then(|| k - 1);
            let up_left = (u > 0 && v > 0).then(|| k - p1 - 1);
            let up_right = (u > 0 && v + 1 < p1).then(|| k - p1 + 1);
            if let Some(j) = same(up) {
                union(&mut parent, k, j);
                continue;
            }
            if let Some(j) = same(left).or_else(|| same(up_left)) {
                union(&mut parent, k, j);
            }
            if let Some(j) = same(up_right) {
                union(&mut parent, k, j);
            }
        }
    }

    // dt is monotone in the squared distance, so a region's largest dt is the dt of its largest squared distance
    let dt = |n: u32| (a * a * n as f32).sqrt();
    let mut slot = vec![usize::MAX; ids.len()];
    let mut regions: Vec<Region> = Vec::new();
    let mut of_pos = vec![usize::MAX; ids.len()];
    for k in 0..ids.len() {
        if ids[k] == u32::MAX {
            continue;
        }
        let r = find(&mut parent, k);
        if slot[r] == usize::MAX {
            slot[r] = regions.len();
            regions.push(Region { piece: ids[k], su: 0.0, sv: 0.0, cnt: 0, best_n: 0, first_f: usize::MAX });
        }
        let reg = &mut regions[slot[r]];
        let (u, v) = (k / p1, k % p1);
        // the coordinate sums accumulate in the plane's C order, rounded after every addition
        reg.su += u as f32;
        reg.sv += v as f32;
        reg.cnt += 1;
        reg.best_n = reg.best_n.max(m2[k]);
        reg.first_f = reg.first_f.min(u + p0 * v);
        of_pos[k] = slot[r];
    }

    let dsq = |pu: f32, pv: f32, qu: f32, qv: f32| {
        let x = a * (pu - qu);
        let y = a * (pv - qv);
        x * x + y * y
    };
    let g = |s: f32, cnt: u32, plane: usize| {
        let p = (a * s) / cnt as f32;
        let p2 = if p - (a * plane as f32) / 2.0 >= 0.0 { p } else { p + a };
        (p2 / a).trunc()
    };
    let (fp0, fp1) = (p0 as f32, p1 as f32);
    let centre = ((a * fp0) / 2.0, (a * fp1) / 2.0);
    let corners = [(-0.5f32, -0.5f32), (fp0 - 0.5, -0.5), (fp0 - 0.5, fp1 - 0.5), (-0.5, fp0 - 0.5)];
    let centroids: Vec<(f32, f32)> = regions.iter().map(|r| (g(r.su, r.cnt, p0), g(r.sv, r.cnt, p1))).collect();
    let key = |k: usize| -> [f32; 4] {
        let (u, v) = ((k / p1) as f32, (k % p1) as f32);
        let (cu, cv) = centroids[of_pos[k]];
        let k1 = dsq(u, v, cu, cv);
        let k2 = dsq(u, v, centre.0, centre.1);
        let k3 = corners.iter().map(|&(qu, qv)| dsq(u, v, qu, qv)).fold(f32::INFINITY, f32::min);
        let k4 = (a * (u - 0.5)).min(a * ((fp0 - 0.5) - u)).min(a * (v - 0.5)).min(a * ((fp1 - 0.5) - v));
        [k1, k2, k3, k4]
    };

    // the pick: largest dt, then the smallest key, then the earliest plane F-index. The voxels at a region's
    // largest dt are those whose squared distance is at least the smallest one with that same dt
    let at_best: Vec<u32> = regions
        .iter()
        .map(|r| {
            let mut lo = r.best_n;
            while lo > 0 && dt(lo - 1) == dt(r.best_n) {
                lo -= 1;
            }
            lo
        })
        .collect();
    let mut best: Vec<Option<([f32; 4], usize, usize)>> = vec![None; regions.len()];
    for (k, &r) in of_pos.iter().enumerate() {
        if r == usize::MAX || m2[k] < at_best[r] {
            continue;
        }
        let cand = (key(k), (k / p1) + p0 * (k % p1), k);
        let better = match &best[r] {
            None => true,
            Some(b) => match cand.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal) {
                Ordering::Less => true,
                Ordering::Equal => cand.1 < b.1,
                Ordering::Greater => false,
            },
        };
        if better {
            best[r] = Some(cand);
        }
    }

    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_unstable_by_key(|&r| regions[r].first_f);
    order
        .into_iter()
        .map(|r| {
            let (_, _, k) = best[r].expect("every region has a voxel at its largest dt");
            let c = full(k / p1, k % p1);
            (regions[r].piece, c.map(|x| x as u32))
        })
        .collect()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Orders standing in for Python's in the unit tests: insertion order and a stable ascending sort, which agree
    /// with the reference's only when nothing ties
    struct Plain;

    impl CompatOrders for Plain {
        fn set_orders(&self, lists: &[&[[u32; 3]]]) -> Result<Vec<Vec<u32>>> {
            let one = |picks: &[[u32; 3]]| {
                let mut out: Vec<u32> = Vec::new();
                for (k, p) in picks.iter().enumerate() {
                    if !out.iter().any(|&j| picks[j as usize] == *p) {
                        out.push(k as u32);
                    }
                }
                out
            };
            Ok(lists.iter().map(|p| one(p)).collect())
        }

        fn argsorts(&self, ds: &[&[f32]]) -> Result<Vec<Vec<u32>>> {
            let one = |d: &[f32]| {
                let mut idx: Vec<u32> = (0..d.len() as u32).collect();
                idx.sort_by(|&x, &y| d[x as usize].total_cmp(&d[y as usize]));
                idx
            };
            Ok(ds.iter().map(|d| one(d)).collect())
        }
    }

    fn villa() -> SkelSettings {
        crate::SettingsSet::Published0211.skel_settings(true)
    }

    fn run(shape: [usize; 3], data: &[u32]) -> Result<Packed> {
        let l = LabelsRef { shape, data };
        let fix = |s: [usize; 3]| crate::effective_fix_borders(true, s);
        skeletonize(&[l], &villa(), fix, &Plain).map(|(mut p, _)| p.remove(0))
    }

    #[test]
    fn bar_is_a_tree_with_dbf_radii() {
        let (z, y, x) = (5, 5, 80);
        let p = run([z, y, x], &vec![1u32; z * y * x]).unwrap();
        assert_eq!(p.ids, vec![1]);
        assert_eq!(p.edges.len() + 1, p.vertices.len());
        // one label fills the array, so the border is boundary
        assert!(p.radii.iter().all(|r| (2.0..=6.0).contains(r)), "{:?}", p.radii);
    }

    #[test]
    fn two_labels_covering_the_array_keep_radius_a() {
        // a 1×2×60 array of two labels, on which the reference fails; this mode keeps every radius a
        let mut two = vec![1u32; 120];
        two[60..].fill(2);
        let p = run([1, 2, 60], &two).unwrap();
        assert_eq!(p.ids.len(), 2);
        assert!(p.radii.iter().all(|&r| r == 2.0));
    }

    #[test]
    fn soma_limit_refuses_the_call() {
        let line = |n: usize| {
            let mut d = vec![1u32; n];
            d[0] = 0;
            d
        };
        let fix = |s: [usize; 3]| crate::effective_fix_borders(true, s);
        let ok = line(551);
        let bad = line(552);
        let ls = [LabelsRef { shape: [1, 1, 551], data: &ok }, LabelsRef { shape: [1, 1, 552], data: &bad }];
        match skeletonize(&ls, &villa(), fix, &Plain) {
            Err(Error::SomaUnsupported { slab, .. }) => assert_eq!(slab, 1),
            other => panic!("{:?}", other.map(|_| ())),
        }
        assert!(skeletonize(&ls[..1], &villa(), fix, &Plain).is_ok());

        // slabs run their fields and traces as one task each, so slabs under the limit may finish their traces
        // before a later slab is found over it: the call still refuses, naming the first slab over the limit
        let mut many = vec![LabelsRef { shape: [1, 1, 551], data: &ok }; 16];
        many.extend([LabelsRef { shape: [1, 1, 552], data: &bad }; 2]);
        match skeletonize(&many, &villa(), fix, &Plain) {
            Err(Error::SomaUnsupported { slab, .. }) => assert_eq!(slab, 16),
            other => panic!("{:?}", other.map(|_| ())),
        }
    }
}
