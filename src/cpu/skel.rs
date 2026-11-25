//! The CPU reference skeletoniser (spec part B §2): pieces, DBF, root, fields, mandatory face targets, the path loop
//! and the packed output of one slab
//!
//! Shortest paths are plain Dijkstra under the step rule of §3.2. The step rule gives every field a unique least
//! fixed point, so these costs equal the GPU's bit for bit whatever order either relaxes in

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use rayon::prelude::*;

use super::edt::{FAR, edt};
use super::pieces::{Pieces, pieces26, skeleton_ids};
use crate::skel::{
    Ball, Choices, FacePick, Inval, Penalty, RootStart, UNREACHED, pdrf, radius, reciprocal, step, step_lengths,
};
use crate::{LabelsRef, Packed, SkelSettings};

/// Phase timers for `TEASAR_PROFILE=1`: nanoseconds summed over threads, and the voxels the path searches settled
#[cfg(feature = "testing")]
pub mod fields;

pub mod prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub const SLAB: usize = 0;
    pub const GRAPH: usize = 1;
    pub const FIELDS: usize = 2;
    pub const FACES: usize = 3;
    pub const SEARCH: usize = 4;
    pub const INVALIDATE: usize = 5;
    pub const SETTLED: usize = 6;
    pub const NAMES: [&str; 7] = ["slab EDTs", "graphs", "fields", "faces", "searches", "invalidation", "settled"];
    pub static COUNTERS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("TEASAR_PROFILE").is_some_and(|v| v == "1"))
    }

    /// Runs `f`, adding its wall time to counter `k` when profiling
    #[inline]
    pub fn time<T>(k: usize, f: impl FnOnce() -> T) -> T {
        if !on() {
            return f();
        }
        let t = Instant::now();
        let out = f();
        COUNTERS[k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }

    pub fn add(k: usize, n: u64) {
        if on() {
            COUNTERS[k].fetch_add(n, Ordering::Relaxed);
        }
    }

    /// The counters since the last call, zeroing them
    pub fn take() -> [u64; 7] {
        std::array::from_fn(|k| COUNTERS[k].swap(0, Ordering::Relaxed))
    }
}

/// Trace counts of one call, for the profile (§3.3)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub pieces: u64,
    pub ordinary: u64,
    pub mandatory: u64,
    /// the pieces' pooled voxels, and the nanoseconds their traces took on one thread
    pub voxels: u64,
    pub busy_ns: u64,
}

impl std::ops::AddAssign for Counts {
    fn add_assign(&mut self, o: Self) {
        self.pieces += o.pieces;
        self.ordinary += o.ordinary;
        self.mandatory += o.mandatory;
        self.voxels += o.voxels;
        self.busy_ns += o.busy_ns;
    }
}

/// The skeletons of one slab's pooled labels
pub fn skeletonize_slab(
    labels: &LabelsRef<'_>,
    skel: &SkelSettings,
    fix_borders: bool,
    choices: &Choices,
) -> (Packed, Counts) {
    let slab = prof::time(prof::SLAB, || Slab::new(labels, skel, fix_borders, choices));
    let ids = skeleton_ids(labels, &slab.pieces, skel.dust_threshold);
    if ids.is_empty() {
        return (Packed::empty(), Counts::default());
    }

    let kept: Vec<u32> = (0..slab.pieces.pieces.len() as u32)
        .filter(|&p| slab.pieces.pieces[p as usize].voxels > skel.dust_threshold)
        .collect();
    let skels: Vec<PieceSkel> = kept
        .par_iter()
        .map(|&p| {
            let t = Instant::now();
            let mut s = slab.piece(p);
            s.counts.busy_ns = t.elapsed().as_nanos() as u64;
            s
        })
        .collect();

    // labels in skeleton order (§2.8); each label's pieces are merged into the sorted form of §2.7
    let mut by_label: Vec<(u32, usize)> =
        kept.iter().enumerate().map(|(k, &p)| (slab.pieces.pieces[p as usize].label, k)).collect();
    by_label.sort_unstable();

    let mut out = Packed { ids: Vec::with_capacity(ids.len()), v_off: vec![0], e_off: vec![0], ..Packed::default() };
    let mut counts = Counts::default();
    let (mut verts, mut edges) = (Vec::new(), Vec::new());
    for &label in &ids {
        verts.clear();
        edges.clear();
        let lo = by_label.partition_point(|&(l, _)| l < label);
        for &(_, k) in by_label[lo..].iter().take_while(|&&(l, _)| l == label) {
            let s = &skels[k];
            verts.extend(s.verts.iter().map(|&g| (g, radius(slab.n_dbf[g as usize], skel.anisotropy[0]))));
            edges.extend(s.edges.iter().map(|&[u, v]| [s.verts[u as usize], s.verts[v as usize]]));
            counts += s.counts;
        }
        out.push_sorted(label, &mut verts, &mut edges, labels.shape, skel.anisotropy);
    }
    (out, counts)
}

/// One piece's skeleton, as slab-global voxel indices in vertex order and piece-local edges
pub struct PieceSkel {
    pub verts: Vec<u32>,
    pub edges: Vec<[u32; 2]>,
    pub counts: Counts,
}

/// One face layer of the array: the layer at `index` along `axis`, with its 2D distances
struct Layer {
    axis: usize,
    index: usize,
    dist: Vec<u32>,
}

/// The slab-wide stages: pieces, voxel lists, the two distance transforms and the face layers
struct Slab<'a> {
    shape: [usize; 3],
    skel: &'a SkelSettings,
    choices: &'a Choices,
    pieces: Pieces,
    /// voxels of piece p, in C order: `voxels[offs[p]..offs[p + 1]]`
    offs: Vec<u32>,
    voxels: Vec<u32>,
    /// each voxel's index within its piece's list
    local: Vec<u32>,
    n_dbf: Vec<u32>,
    n_pen: Vec<u32>,
    /// the squared distances the invalidation radius uses, when not the DBF's
    n_inv: Option<Vec<u32>>,
    layers: Vec<Layer>,
}

impl<'a> Slab<'a> {
    fn new(labels: &LabelsRef<'_>, skel: &'a SkelSettings, fix_borders: bool, choices: &'a Choices) -> Self {
        let shape = labels.shape;
        let n = labels.data.len();
        let pieces = pieces26(labels);

        // a piece that fills the array measures to the positions just outside it (§2.2)
        let full = pieces.pieces.len() == 1 && pieces.pieces[0].voxels as usize == n;
        let n_dbf = edt(&pieces.of_voxel, shape, [full; 3]);
        let pen_border = penalty_border(shape, full);
        let n_pen = match choices.penalty {
            Penalty::P2 if pen_border != [full; 3] => edt(&pieces.of_voxel, shape, pen_border),
            _ => n_dbf.clone(),
        };
        let n_inv = match choices.inval {
            Inval::Dbf => None,
            Inval::Penalty => Some(n_pen.clone()),
            Inval::Thin => Some(edt(&pieces.of_voxel, shape, shape.map(|s| full || s >= 2))),
        };

        let np = pieces.pieces.len();
        let mut offs = vec![0u32; np + 1];
        for (k, p) in pieces.pieces.iter().enumerate() {
            offs[k + 1] = offs[k] + p.voxels;
        }
        let mut fill = offs.clone();
        let mut voxels = vec![0u32; offs[np] as usize];
        let mut local = vec![u32::MAX; n];
        for (g, &p) in pieces.of_voxel.iter().enumerate() {
            if p == u32::MAX {
                continue;
            }
            let slot = &mut fill[p as usize];
            local[g] = *slot - offs[p as usize];
            voxels[*slot as usize] = g as u32;
            *slot += 1;
        }

        let mut layers = Vec::new();
        if fix_borders {
            let axes = if choices.faces_xyz { [2, 1, 0] } else { [0, 1, 2] };
            for axis in axes {
                let mut idx =
                    if choices.faces_high_first { vec![shape[axis] - 1, 0] } else { vec![0, shape[axis] - 1] };
                idx.dedup();
                for index in idx {
                    layers.push(Layer { axis, index, dist: face_edt(&pieces.of_voxel, shape, axis, index) });
                }
            }
        }

        Self { shape, skel, choices, pieces, offs, voxels, local, n_dbf, n_pen, n_inv, layers }
    }

    fn coords(&self, g: usize) -> [usize; 3] {
        let [_, y, x] = self.shape;
        [g / (y * x), (g / x) % y, g % x]
    }

    fn piece(&self, pid: u32) -> PieceSkel {
        trace_piece(&self.ctx(pid), |g| self.local[g])
    }

    /// Voxels of piece `pid`, in C order
    fn vox(&self, pid: u32) -> &[u32] {
        &self.voxels[self.offs[pid as usize] as usize..self.offs[pid as usize + 1] as usize]
    }

    /// What tracing piece `pid` needs, its mandatory targets included
    fn ctx(&self, pid: u32) -> PieceCtx<'_> {
        let vox = self.vox(pid);
        let mandatory =
            prof::time(prof::FACES, || if self.layers.is_empty() { Vec::new() } else { self.face_targets(pid, vox) });
        PieceCtx {
            shape: self.shape,
            skel: self.skel,
            choices: self.choices,
            of_voxel: &self.pieces.of_voxel,
            me: pid,
            vox,
            n_dbf: &self.n_dbf,
            n_pen: &self.n_pen,
            n_inv: self.n_inv.as_deref().unwrap_or(&self.n_dbf),
            mandatory,
        }
    }

    /// The mandatory targets of §2.4, in tracing order, without duplicates
    fn face_targets(&self, pid: u32, vox: &[u32]) -> Vec<usize> {
        let mut out = Vec::new();
        for layer in &self.layers {
            let members: Vec<usize> =
                (0..vox.len()).filter(|&i| self.coords(vox[i] as usize)[layer.axis] == layer.index).collect();
            if members.is_empty() {
                continue;
            }
            for region in self.regions(pid, vox, layer, &members) {
                let t = self.pick(vox, layer, &region);
                if !out.contains(&t) {
                    out.push(t);
                }
            }
        }
        out
    }

    /// The face regions of `members` (piece-local indices in one layer, increasing), as lists ordered by their lowest
    /// index
    fn regions(&self, pid: u32, vox: &[u32], layer: &Layer, members: &[usize]) -> Vec<Vec<usize>> {
        let (u_ax, v_ax) = in_plane(layer.axis);
        let mut parent: Vec<usize> = (0..members.len()).collect();
        let preceding: &[(i64, i64)] =
            if self.choices.face_8 { &[(-1, -1), (-1, 0), (-1, 1), (0, -1)] } else { &[(-1, 0), (0, -1)] };
        for (k, &i) in members.iter().enumerate() {
            let c = self.coords(vox[i] as usize);
            for &(du, dv) in preceding {
                let (u, v) = (c[u_ax] as i64 + du, c[v_ax] as i64 + dv);
                if u < 0 || v < 0 || u >= self.shape[u_ax] as i64 || v >= self.shape[v_ax] as i64 {
                    continue;
                }
                let mut nc = c;
                nc[u_ax] = u as usize;
                nc[v_ax] = v as usize;
                let h = (nc[0] * self.shape[1] + nc[1]) * self.shape[2] + nc[2];
                if self.pieces.of_voxel[h] != pid {
                    continue;
                }
                let other = members.binary_search(&(self.local[h] as usize)).expect("a layer neighbour is a member");
                let (ra, rb) = (find(&mut parent, k), find(&mut parent, other));
                parent[ra.max(rb)] = ra.min(rb);
            }
        }

        let mut regions: Vec<Vec<usize>> = Vec::new();
        let mut slot = vec![usize::MAX; members.len()];
        for (k, &member) in members.iter().enumerate() {
            let r = find(&mut parent, k);
            if slot[r] == usize::MAX {
                slot[r] = regions.len();
                regions.push(Vec::new());
            }
            regions[slot[r]].push(member);
        }
        regions
    }

    /// A region's mandatory target: a 2D distance maximum, chosen among ties as `face_pick` says
    fn pick(&self, vox: &[u32], layer: &Layer, region: &[usize]) -> usize {
        let (u_ax, v_ax) = in_plane(layer.axis);
        let plane = |i: usize| {
            let c = self.coords(vox[i] as usize);
            (c[u_ax], c[v_ax])
        };
        let width = self.shape[v_ax];
        let dist = |i: usize| {
            let (u, v) = plane(i);
            layer.dist[u * width + v]
        };
        let best = region.iter().map(|&i| dist(i)).max().unwrap_or(0);
        let n = region.len() as i128;
        let (su, sv) =
            region.iter().fold((0i128, 0i128), |(a, b), &i| (a + plane(i).0 as i128, b + plane(i).1 as i128));
        let (h, w) = (self.shape[u_ax] as i64 - 1, self.shape[v_ax] as i64 - 1);
        let key = |i: usize| -> FaceKey {
            let (u, v) = plane(i);
            let (u, v) = (u as i64, v as i64);
            // squared distance to the centroid, times n^2, so it stays exact
            let (du, dv) = (n * u as i128 - su, n * v as i128 - sv);
            let centroid = (du * du + dv * dv) as u128;
            // to the plane's centre, times 4
            let centre = ((2 * u - h).pow(2) + (2 * v - w).pow(2)) as u64;
            let corner = (u.min(h - u).pow(2) + v.min(w - v).pow(2)) as u64;
            let edge = u.min(h - u).min(v).min(w - v) as u64;
            match self.choices.face_pick {
                FacePick::Readme => (centroid, centre, corner, edge, Reverse(i)),
                FacePick::Centroid => (centroid, 0, 0, 0, Reverse(usize::MAX - i)),
                FacePick::CentroidHigh => (centroid, 0, 0, 0, Reverse(i)),
                FacePick::Lowest => (0, 0, 0, 0, Reverse(usize::MAX - i)),
            }
        };
        region.iter().copied().filter(|&i| dist(i) == best).min_by_key(|&i| key(i)).unwrap_or(region[0])
    }
}

/// The lexicographic order of a face pick: the README's four nearness filters, then the C index (smallest key wins)
type FaceKey = (u128, u64, u64, u64, Reverse<usize>);

/// The axes whose array border is penalty boundary (§2.5): those longer than 2; with none, the DBF's own rule, which
/// counts every axis only for a piece that fills the array
pub fn penalty_border(shape: [usize; 3], full: bool) -> [bool; 3] {
    let long = shape.map(|s| s > 2);
    if long == [false; 3] { [full; 3] } else { long }
}

/// One piece's inputs to the path loop: its slab's piece ids and squared distances (slab-local voxel indices), its
/// voxel list in C order and its border targets
///
/// The CPU reference builds these from its own slab stages; the Metal backend builds them from the GPU's, to run
/// large pieces on host threads
pub struct PieceCtx<'a> {
    pub shape: [usize; 3],
    pub skel: &'a SkelSettings,
    pub choices: &'a Choices,
    pub of_voxel: &'a [u32],
    pub me: u32,
    /// the piece's voxels, increasing
    pub vox: &'a [u32],
    pub n_dbf: &'a [u32],
    pub n_pen: &'a [u32],
    pub n_inv: &'a [u32],
    /// border targets as list indices, in tracing order, without duplicates
    pub mandatory: Vec<usize>,
}

impl PieceCtx<'_> {
    fn coords(&self, g: usize) -> [usize; 3] {
        let [_, y, x] = self.shape;
        [g / (y * x), (g / x) % y, g % x]
    }

    /// The piece's 26-neighbour lists, neighbours in increasing index, each tagged with how many coordinates the
    /// step changes
    fn graph(&self, local: &impl Fn(usize) -> u32) -> Graph {
        let [z, y, x] = self.shape.map(|s| s as i64);
        let mut off = Vec::with_capacity(self.vox.len() + 1);
        let mut adj = Vec::with_capacity(self.vox.len() * 8);
        off.push(0u32);
        for &g in self.vox {
            let [gz, gy, gx] = self.coords(g as usize).map(|c| c as i64);
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (nz, ny, nx) = (gz + dz, gy + dy, gx + dx);
                        if (dz, dy, dx) == (0, 0, 0) || nz < 0 || ny < 0 || nx < 0 || nz >= z || ny >= y || nx >= x {
                            continue;
                        }
                        let h = ((nz * y + ny) * x + nx) as usize;
                        if self.of_voxel[h] != self.me {
                            continue;
                        }
                        let kind = (dz != 0) as u32 + (dy != 0) as u32 + (dx != 0) as u32;
                        adj.push(local(h) << 2 | kind);
                    }
                }
            }
            off.push(adj.len() as u32);
        }
        Graph { off, adj }
    }
}

/// A piece's root and fields (spec part B §2.3 to §2.5), indexed by list entry; `d` holds cost bit patterns
pub struct PieceFields {
    pub root: usize,
    pub pen: Vec<f32>,
    pub d: Vec<u32>,
    pub p: Vec<f32>,
}

/// The root, `B`, `D` and `P` of the piece of `ctx`, whose graph is `graph`
fn piece_fields(ctx: &PieceCtx<'_>, graph: &Graph) -> PieceFields {
    let vox = ctx.vox;
    let m = vox.len();
    let a = ctx.skel.anisotropy[0];
    let pen: Vec<f32> = vox.iter().map(|&g| radius(ctx.n_pen[g as usize], a)).collect();
    let lengths = step_lengths(a);
    let mandatory = &ctx.mandatory;

    let fields_start = prof::on().then(std::time::Instant::now);
    let start = match ctx.choices.root {
        RootStart::COrder | RootStart::Face | RootStart::FaceLast | RootStart::FaceFar => 0,
        RootStart::FOrder => {
            let [z, y, _] = ctx.shape;
            (0..m)
                .min_by_key(|&i| {
                    let [zz, yy, xx] = ctx.coords(vox[i] as usize);
                    zz + z * (yy + y * xx)
                })
                .unwrap_or(0)
        }
        RootStart::DbfMax => argmax(&vox.iter().map(|&g| ctx.n_dbf[g as usize]).collect::<Vec<_>>()),
    };
    let root = match ctx.choices.root {
        RootStart::Face if !mandatory.is_empty() => mandatory[0],
        RootStart::FaceLast if !mandatory.is_empty() => mandatory[mandatory.len() - 1],
        RootStart::FaceFar if !mandatory.is_empty() => {
            let d0 = graph.geodesic(0, &lengths);
            // farthest, lowest index among ties
            *mandatory.iter().max_by_key(|&&t| (d0[t], Reverse(t))).expect("not empty")
        }
        _ => argmax(&graph.geodesic(start, &lengths)),
    };
    let d = graph.geodesic(root, &lengths);

    let r_d = reciprocal(f32::from_bits(d.iter().copied().max().unwrap_or(0)));
    let r_b = reciprocal(pen.iter().copied().fold(0.0f32, f32::max));
    let (form, daf) = (ctx.choices.penalty_form, ctx.choices.daf_term);
    let p: Vec<f32> = (0..m).map(|i| pdrf(pen[i], r_b, f32::from_bits(d[i]), r_d, form, daf)).collect();
    if let Some(t) = fields_start {
        prof::add(prof::FIELDS, t.elapsed().as_nanos() as u64);
    }

    PieceFields { root, pen, d, p }
}

/// One piece's skeleton (spec part B §2.3 to §2.7); `local` maps a voxel of the piece to its list index
pub fn trace_piece(ctx: &PieceCtx<'_>, local: impl Fn(usize) -> u32) -> PieceSkel {
    let vox = ctx.vox;
    let m = vox.len();
    let a = ctx.skel.anisotropy[0];
    let graph = prof::time(prof::GRAPH, || ctx.graph(&local));
    let PieceFields { root, d, p, .. } = piece_fields(ctx, &graph);
    let mandatory = &ctx.mandatory;

    let inv: Vec<f32> = vox.iter().map(|&g| radius(ctx.n_inv[g as usize], a)).collect();
    let mut run = Run::new(ctx, &local, &graph, &inv, &p);
    run.add_root(root);

    if ctx.choices.mandatory_first {
        for &t in mandatory {
            run.trace(t);
        }
    }

    // ordinary targets: valid voxels by largest D, then lowest index
    let mut order: Vec<u32> = (0..m as u32).collect();
    order.sort_unstable_by_key(|&i| (Reverse(d[i as usize]), i));
    let mut ordinary = 0u64;
    for &t in &order {
        if run.valid[t as usize] {
            run.trace(t as usize);
            ordinary += 1;
        }
    }

    if !ctx.choices.mandatory_first {
        for &t in mandatory {
            run.trace(t);
        }
    }

    let verts = run.verts.iter().map(|&i| vox[i as usize]).collect();
    let counts = Counts { pieces: 1, ordinary, mandatory: mandatory.len() as u64, voxels: m as u64, busy_ns: 0 };
    PieceSkel { verts, edges: run.edges, counts }
}

/// The two in-plane axes of a layer normal to `axis`, in array order
fn in_plane(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// The 2D distances of the face layer at `index` along `axis`: the squared distance to the nearest voxel of another
/// piece or background, with the plane's border counted as background (§2.4)
fn face_edt(of_voxel: &[u32], shape: [usize; 3], axis: usize, index: usize) -> Vec<u32> {
    let (u_ax, v_ax) = in_plane(axis);
    let (h, w) = (shape[u_ax], shape[v_ax]);
    let mut ids = vec![0u32; h * w];
    for u in 0..h {
        for v in 0..w {
            let mut c = [0usize; 3];
            c[axis] = index;
            c[u_ax] = u;
            c[v_ax] = v;
            ids[u * w + v] = of_voxel[(c[0] * shape[1] + c[1]) * shape[2] + c[2]];
        }
    }
    let mut dist = edt(&ids, [1, h, w], [false, true, true]);
    debug_assert!(dist.iter().all(|&d| d != FAR));
    dist.shrink_to_fit();
    dist
}

/// A piece's voxel graph in compressed rows; each entry is `local << 2 | coordinates changed`
struct Graph {
    off: Vec<u32>,
    adj: Vec<u32>,
}

impl Graph {
    fn neighbours(&self, i: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adj[self.off[i] as usize..self.off[i + 1] as usize].iter().map(|&e| ((e >> 2) as usize, (e & 3) as usize))
    }

    /// Geodesic distances from `src` as float32 bit patterns, steps costing their physical length under the step rule
    fn geodesic(&self, src: usize, lengths: &[f32; 4]) -> Vec<u32> {
        let m = self.off.len() - 1;
        let mut cost = vec![UNREACHED; m];
        let mut heap = BinaryHeap::new();
        cost[src] = 0;
        heap.push(Reverse(key(0, src)));
        while let Some(Reverse(k)) = heap.pop() {
            let (c, i) = unkey(k);
            if c != cost[i] {
                continue;
            }
            for (j, kind) in self.neighbours(i) {
                let nc = step(c, lengths[kind]);
                if nc < cost[j] {
                    cost[j] = nc;
                    heap.push(Reverse(key(nc, j)));
                }
            }
        }
        cost
    }
}

#[inline]
fn key(cost: u32, i: usize) -> u64 {
    u64::from(cost) << 32 | i as u64
}

#[inline]
fn unkey(k: u64) -> (u32, usize) {
    ((k >> 32) as u32, (k & 0xffff_ffff) as usize)
}

/// The index of the largest value, lowest index among ties
fn argmax(v: &[u32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The path loop's state for one piece (§2.6)
struct Run<'s, L: Fn(usize) -> u32> {
    shape: [usize; 3],
    pid: u32,
    vox: &'s [u32],
    of_voxel: &'s [u32],
    local: &'s L,
    graph: &'s Graph,
    dbf: &'s [f32],
    p: &'s [f32],
    a: f32,
    scale: f32,
    constant: f32,
    count_target: bool,
    /// invalidate only the ball voxels connected to its centre inside the ball
    ball_connected: bool,
    /// the ball search's visit stamps and stack, for `ball_connected`
    seen: Vec<u32>,
    stamp: u32,
    stack: Vec<usize>,
    valid: Vec<bool>,
    /// vertex number of each voxel of S, else `u32::MAX`
    vid: Vec<u32>,
    verts: Vec<u32>,
    edges: Vec<[u32; 2]>,
    cost: Vec<u32>,
    touched: Vec<usize>,
    heap: BinaryHeap<Reverse<u64>>,
}

impl<'s, L: Fn(usize) -> u32> Run<'s, L> {
    fn new(ctx: &'s PieceCtx<'_>, local: &'s L, graph: &'s Graph, dbf: &'s [f32], p: &'s [f32]) -> Self {
        let m = ctx.vox.len();
        Self {
            shape: ctx.shape,
            pid: ctx.me,
            vox: ctx.vox,
            of_voxel: ctx.of_voxel,
            local,
            graph,
            dbf,
            p,
            a: ctx.skel.anisotropy[0] as f32,
            scale: ctx.skel.scale as f32,
            constant: ctx.skel.constant as f32,
            count_target: ctx.choices.count_target,
            ball_connected: ctx.choices.ball_connected,
            seen: if ctx.choices.ball_connected { vec![0; m] } else { Vec::new() },
            stamp: 0,
            stack: Vec::new(),
            valid: vec![true; m],
            vid: vec![u32::MAX; m],
            verts: Vec::new(),
            edges: Vec::new(),
            cost: vec![UNREACHED; m],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn in_s(&self, i: usize) -> bool {
        self.vid[i] != u32::MAX
    }

    fn add_root(&mut self, root: usize) {
        self.vid[root] = 0;
        self.verts.push(root as u32);
        self.invalidate(root);
    }

    /// Traces target `t`: its cheapest path into S, recorded and invalidated
    fn trace(&mut self, t: usize) {
        if self.in_s(t) {
            self.invalidate(t);
            return;
        }
        let path = prof::time(prof::SEARCH, || self.search(t));
        prof::time(prof::INVALIDATE, || {
            for &v in &path {
                self.invalidate(v);
            }
        });
        // new vertices: every path voxel but the join, in order from t, each joined to the next path voxel
        for &v in &path[..path.len() - 1] {
            self.vid[v] = self.verts.len() as u32;
            self.verts.push(v as u32);
        }
        for w in path.windows(2) {
            self.edges.push([self.vid[w[0]], self.vid[w[1]]]);
        }
    }

    /// The cheapest path from `t` (not in S) to S, as voxels from `t` to the join
    fn search(&mut self, t: usize) -> Vec<usize> {
        for &i in &self.touched {
            self.cost[i] = UNREACHED;
        }
        self.touched.clear();
        self.heap.clear();

        let c0 = if self.count_target { self.p[t].to_bits() } else { 0 };
        self.cost[t] = c0;
        self.touched.push(t);
        self.heap.push(Reverse(key(c0, t)));
        let mut join = None;
        let mut settled = 0u64;
        while let Some(Reverse(k)) = self.heap.pop() {
            let (c, i) = unkey(k);
            if c != self.cost[i] {
                continue;
            }
            settled += 1;
            // the first voxel of S popped has the smallest (cost, index): later costs are strictly larger
            if self.in_s(i) {
                join = Some(i);
                break;
            }
            for (j, _) in self.graph.neighbours(i) {
                let nc = step(c, self.weight(j));
                if nc < self.cost[j] {
                    if self.cost[j] == UNREACHED {
                        self.touched.push(j);
                    }
                    self.cost[j] = nc;
                    self.heap.push(Reverse(key(nc, j)));
                }
            }
        }

        prof::add(prof::SETTLED, settled);

        // S is never empty and the piece is connected, so a join exists
        let mut q = join.expect("a path search reaches S");
        let mut path = vec![q];
        while q != t {
            q = self.predecessor(q);
            path.push(q);
        }
        path.reverse();
        path
    }

    /// The weight of entering `q`: its PDRF outside S, 0 into S
    fn weight(&self, q: usize) -> f32 {
        if self.in_s(q) { 0.0 } else { self.p[q] }
    }

    /// The lowest-index expanded neighbour whose step reproduces `q`'s final cost (§3.2 phase 2)
    fn predecessor(&self, q: usize) -> usize {
        let w = self.weight(q);
        self.graph
            .neighbours(q)
            .map(|(p, _)| p)
            .find(|&p| !self.in_s(p) && self.cost[p] != UNREACHED && step(self.cost[p], w) == self.cost[q])
            .expect("every reached voxel but the target has a predecessor")
    }

    /// Marks invalid every piece voxel strictly inside the ball of radius `scale * DBF + const` around `v` (§2.6)
    fn invalidate(&mut self, v: usize) {
        let ball = Ball::new(self.dbf[v], self.scale, self.constant, self.a);
        let k = ball.k as usize;
        let g = self.vox[v] as usize;
        let [z, y, x] = self.shape;
        let c = [g / (y * x), (g / x) % y, g % x];
        if self.ball_connected {
            self.invalidate_connected(v, c, ball);
            return;
        }
        let span = |c: usize, n: usize| c.saturating_sub(k)..(c + k + 1).min(n);
        for zz in span(c[0], z) {
            let dz = zz.abs_diff(c[0]).pow(2);
            for yy in span(c[1], y) {
                let dzy = dz + yy.abs_diff(c[1]).pow(2);
                let row = (zz * y + yy) * x;
                for xx in span(c[2], x) {
                    let inside = ball.contains((dzy + xx.abs_diff(c[2]).pow(2)) as u32);
                    if inside && self.of_voxel[row + xx] == self.pid {
                        self.valid[(self.local)(row + xx) as usize] = false;
                    }
                }
            }
        }
    }

    /// The ball voxels reachable from `v` through 26-steps that stay inside the ball around `c`
    fn invalidate_connected(&mut self, v: usize, c: [usize; 3], ball: Ball) {
        let [_, y, x] = self.shape;
        let (graph, vox) = (self.graph, self.vox);
        self.stamp += 1;
        let stamp = self.stamp;
        self.seen[v] = stamp;
        self.stack.clear();
        self.stack.push(v);
        while let Some(i) = self.stack.pop() {
            self.valid[i] = false;
            for (j, _) in graph.neighbours(i) {
                if self.seen[j] == stamp {
                    continue;
                }
                let h = vox[j] as usize;
                let d2 = (h / (y * x)).abs_diff(c[0]).pow(2)
                    + ((h / x) % y).abs_diff(c[1]).pow(2)
                    + (h % x).abs_diff(c[2]).pow(2);
                let inside = ball.contains(d2 as u32);
                if inside {
                    self.seen[j] = stamp;
                    self.stack.push(j);
                }
            }
        }
    }
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
    use crate::SettingsSet;

    fn published() -> SkelSettings {
        SettingsSet::Published0211.skel_settings(true)
    }

    fn run(shape: [usize; 3], data: &[u32]) -> Packed {
        let l = LabelsRef { shape, data };
        skeletonize_slab(&l, &published(), crate::effective_fix_borders(true, shape), &Choices::default()).0
    }

    #[test]
    fn synthetic_radii() {
        // spec part B §4.3
        let mut two = vec![1u32; 120];
        two[60..].fill(2);
        let p = run([1, 2, 60], &two);
        assert!(p.radii.iter().all(|&r| r == 2.0));

        let p = run([1, 5, 200], &vec![1u32; 1000]);
        assert!(p.radii.iter().all(|&r| r == 2.0));

        let p = run([3, 9, 60], &vec![1u32; 3 * 9 * 60]);
        for (v, &r) in p.vertices.iter().zip(&p.radii) {
            let [z, y, x] = v.map(|c| (c / 2.0) as usize);
            let edge = z == 0 || z == 2 || y == 0 || y == 8 || x == 0 || x == 59;
            assert_eq!(r, if edge { 2.0 } else { 4.0 }, "at {z} {y} {x}");
        }
    }

    #[test]
    fn tree_per_piece() {
        // an L of width 5
        let (h, w) = (40, 40);
        let mut d = vec![0u32; h * w];
        for y in 5..10 {
            for x in 5..35 {
                d[y * w + x] = 7;
            }
        }
        for y in 5..35 {
            for x in 30..35 {
                d[y * w + x] = 7;
            }
        }
        let p = run([1, h, w], &d);
        assert_eq!(p.ids, vec![7]);
        assert_eq!(p.edges.len() + 1, p.vertices.len());
        // every edge joins 26-neighbours
        for &[u, v] in &p.edges {
            let (a, b) = (p.vertices[u as usize], p.vertices[v as usize]);
            assert!((0..3).all(|k| (a[k] - b[k]).abs() <= 2.0), "{a:?} {b:?}");
        }
    }
}
