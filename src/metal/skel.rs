//! Host side of the part B kernels (`shaders/metal/skel.metal`): batch planning, the piece table, border-target
//! order, the resume loop and the packed output
//!
//! A call's slabs are split into internal batches of at most `batch_voxels` voxels (a larger slab is a batch of its
//! own). Each batch waits on the GPU at three points:
//!
//! 1. after the pieces, to read the piece table and plan each kept piece's workspace
//! 2. after the slab-wide stages, to put each piece's border targets in tracing order
//! 3. after every path-loop dispatch, to drop finished pieces; a piece that spends its work budget resumes in the
//!    next dispatch from the state it left in device memory (spec part B §3.3)
//!
//! The output is assembled on the host from each piece's vertex list, a deviation from §3.1's GPU scatter: with
//! shared memory the host reads the lists in place

use std::cmp::Reverse;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use color_eyre::eyre::{Result, bail, ensure};
use rayon::prelude::*;

use super::gpu::{Arg, Buf, Gpu};
use crate::cpu::skel::{PieceCtx, PieceSkel, trace_piece};
use crate::skel::{Choices, FacePick, Inval, Penalty, PenaltyForm, RootStart, radius, step_lengths};
use crate::{CallStats, LabelsRef, Packed, SkelSettings};

#[cfg(feature = "testing")]
mod fields;

/// Voxels of one internal batch, by default (`TEASAR_SKEL_BATCH_VOXELS`)
pub const BATCH_VOXELS: usize = 1 << 23;

/// Work units a piece may spend in one path-loop dispatch, by default (`TEASAR_GPU_BUDGET`)
pub const BUDGET: u32 = 1 << 16;

const NONE: u32 = u32::MAX;
const SCAN_BLOCK: usize = 1024;
const SCAN_THREADS: usize = 256;

/// Pieces over this many voxels are traced on host threads, by default (`TEASAR_CPU_PIECE`): one GPU lane per piece is
/// latency-bound, so a large piece's lane runs far longer than a CPU core takes for it
pub const CPU_PIECE: u32 = 1000;

/// Lanes per threadgroup of the path loop
const RUN_GROUP: usize = 32;

/// Arrays of `m` words in each piece's workspace (WS_* in skel.metal)
const WS_ARRAYS: usize = 11;
const WS_VOX: usize = 0;
const WS_VERT: usize = 9;
const WS_NEXT: usize = 10;

const PH_DONE: u32 = 13;
const PH_FAIL: u32 = 14;
/// The path loop's phases, in the shader's numbering (`PH_*` in `skel.metal`)
const PHASES: [&str; 13] = [
    "start",
    "largest radius",
    "geodesic",
    "reset",
    "order",
    "order ties",
    "fields",
    "root",
    "target pick",
    "search",
    "walk-back",
    "clean",
    "invalidation",
];

const F_COUNT_TARGET: u32 = 1;
const F_MAND_FIRST: u32 = 4;
const F_DAF: u32 = 8;
const F_ROOT_SHIFT: u32 = 4;
const F_UNSQUARED: u32 = 128;
#[cfg(feature = "testing")]
const F_FIELDS_STOP: u32 = 256;

/// Words of a face pick key, one per `sk_face_pick` pass (skel.metal `PICK_WORDS`)
const PICK_WORDS: u32 = 6;

/// Mirrors `SkSlab` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkSlab {
    z: u32,
    y: u32,
    x: u32,
    base: u32,
}

/// Mirrors `SkPiece` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SkPiece {
    pid: u32,
    slab: u32,
    m: u32,
    ws: u32,
    valid: u32,
    mand_off: u32,
    mand_n: u32,
    first: u32,
    ffirst: u32,
    /// 1: traced on host threads, so its workspace is its voxel list only
    host: u32,
    pad: [u32; 2],
}

/// Mirrors `SkView` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SkView {
    id_base: u32,
    s: [u32; 3],
    n: [u32; 3],
    out: u32,
    border: u32,
    vox_off: u32,
    line_off: u32,
    pad: u32,
}

/// Mirrors `SkLayer` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkLayer {
    off: u32,
    h: u32,
    w: u32,
    vbase: u32,
    su: u32,
    sv: u32,
    order: u32,
    pad: u32,
}

/// Mirrors `SkParams` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SkParams {
    a: f32,
    inv_a: f32,
    scale: f32,
    constant: f32,
    len: [f32; 4],
    flags: u32,
    budget: u32,
    zero: u32,
    nslabs: u32,
    inv_field: u32,
    pen_field: u32,
    dbf_field: u32,
    pad: u32,
}

/// Mirrors `SkState` in skel.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SkState {
    phase: u32,
    i: u32,
    hn: u32,
    nt: u32,
    src: u32,
    best: u32,
    bestc: u32,
    root: u32,
    geo: u32,
    maxd: u32,
    nverts: u32,
    ord: u32,
    mand: u32,
    len: u32,
    join: u32,
    ordinary: u32,
    steps: u32,
    runs: u32,
}

/// Call-wide profile counters (spec part B §3.3)
#[derive(Debug, Default, Clone, Copy)]
pub struct SkelProfile {
    pub batches: u64,
    pub pieces: u64,
    pub resume_dispatches: u64,
    pub paused: u64,
    pub ordinary: u64,
    pub mandatory: u64,
    pub steps: u64,
    pub host_pieces: u64,
    /// host pieces sent there by the fallback above `TEASAR_LARGE_PIECE`, not by the hybrid split
    pub fallback_pieces: u64,
    /// dispatches that ended with a piece unfinished, by the phase it paused in
    pub phase_pauses: [u64; PHASES.len()],
    /// thread-seconds the host pieces took, and the wall time from the GPU loop's start until they were done
    pub host_thread_secs: f64,
    pub host_wall_secs: f64,
    /// host wall seconds per stage: pieces, slab stages, path loop, assembly
    pub secs: [f64; 4],
    /// path-loop wall seconds for pieces of at most the median size, up to p90, up to p99, and the largest 1%
    /// (`TEASAR_PROFILE_BUCKETS=1`, which runs the buckets one after another)
    pub bucket_secs: [f64; 4],
    pub bucket_pieces: [u64; 4],
}

impl SkelProfile {
    /// Prints the counters; `host_threads` is the pool the host pieces ran on
    pub fn print(&self, host_threads: usize) {
        let s = self.secs;
        eprintln!(
            "TEASAR_PROFILE: skeletonize: {} batches, {} pieces, {} resume dispatches, {} pieces paused, {} ordinary \
             and {} mandatory traces, {} work units",
            self.batches, self.pieces, self.resume_dispatches, self.paused, self.ordinary, self.mandatory, self.steps
        );
        eprintln!(
            "TEASAR_PROFILE: host pieces: {} pieces on {} host threads ({} by the hybrid split, {} by the fallback above \
             TEASAR_LARGE_PIECE), {:.3} thread-s, done {:.3} s after the path loop started",
            self.host_pieces,
            host_threads,
            self.host_pieces - self.fallback_pieces,
            self.fallback_pieces,
            self.host_thread_secs,
            self.host_wall_secs
        );
        let pauses: Vec<String> = PHASES
            .iter()
            .zip(self.phase_pauses)
            .filter(|&(_, n)| n > 0)
            .map(|(name, n)| format!("{name} {n}"))
            .collect();
        eprintln!(
            "TEASAR_PROFILE: pauses by phase: {}",
            if pauses.is_empty() { "none".into() } else { pauses.join(", ") }
        );
        eprintln!(
            "TEASAR_PROFILE: skeletonize wall: pieces {:.3} s, slab stages {:.3} s, path loop {:.3} s, assembly {:.3} s",
            s[0], s[1], s[2], s[3]
        );
        if self.bucket_pieces.iter().any(|&n| n > 0) {
            let b = self.bucket_secs;
            let n = self.bucket_pieces;
            eprintln!(
                "TEASAR_PROFILE: path loop by piece size: <= median {:.3} s ({} pieces), <= p90 {:.3} s ({}), <= p99 \
                 {:.3} s ({}), largest 1% {:.3} s ({})",
                b[0], n[0], b[1], n[1], b[2], n[2], b[3], n[3]
            );
        }
    }
}

/// The part B pipeline on one device
pub struct Skel {
    pub batch_voxels: usize,
    pub budget: u32,
    pub cpu_piece: u32,
    /// pieces over this many voxels fall back to the host (`TEASAR_LARGE_PIECE`); unset, the call's size-filter maximum
    large_piece: Option<u32>,
    /// this call's limit: `large_piece`, or the call's default
    large_limit: u32,
    /// `TEASAR_CPU_FALLBACK=0` makes the fallback an error, for tests
    cpu_fallback: bool,
    pub profile: SkelProfile,
    buckets: bool,
    /// the threads that trace host pieces beside the GPU loop, apart from the thread driving the GPU
    host: Arc<rayon::ThreadPool>,
    /// where this call's pieces ran (spec part A §1.4), always collected
    pub stats: CallStats,
}

/// One kept piece as the host plans it
#[derive(Clone, Copy)]
struct Kept {
    pid: u32,
    slab: u32,
    m: u32,
}

/// Where each distance field lives in the batch's field buffer
#[derive(Clone, Copy)]
struct Fields {
    dbf: u32,
    pen: u32,
    inv: u32,
    layers: u32,
    total: usize,
}

/// One batch: its slabs, piece table and the device arrays its path loop reads
struct Batch {
    slabs: Vec<SkSlab>,
    /// per piece: label and smallest slab-local Fortran index
    plabel: Vec<u32>,
    pff: Vec<u32>,
    psize: Vec<u32>,
    kidx: Vec<u32>,
    kept: Vec<Kept>,
    pieces: Vec<SkPiece>,
    /// first piece of each slab, and one past the last
    slab_pieces: Vec<u32>,
    valid_words: usize,
    slab_buf: Buf,
    pid: Buf,
    ws: Buf,
}

/// The arrays the slab stages add for the path loop
struct Staged {
    local: Buf,
    valid: Buf,
    fields: Buf,
    f: Fields,
    maxpen: Buf,
    mand: Buf,
    pieces: Buf,
    pieces_host: Vec<SkPiece>,
    mand_host: Vec<u32>,
}

impl Skel {
    /// A pipeline whose host pieces run on `host_threads` threads
    pub fn new(host_threads: usize) -> Result<Self> {
        fn env<T: std::str::FromStr>(k: &str) -> Option<T> {
            std::env::var(k).ok().and_then(|v| v.parse().ok())
        }
        Ok(Self {
            batch_voxels: env("TEASAR_SKEL_BATCH_VOXELS").unwrap_or(BATCH_VOXELS),
            budget: env("TEASAR_GPU_BUDGET").unwrap_or(BUDGET),
            cpu_piece: env("TEASAR_CPU_PIECE").unwrap_or(CPU_PIECE),
            large_piece: env("TEASAR_LARGE_PIECE"),
            large_limit: u32::MAX,
            cpu_fallback: std::env::var_os("TEASAR_CPU_FALLBACK").is_none_or(|v| v != "0"),
            profile: SkelProfile::default(),
            buckets: std::env::var_os("TEASAR_PROFILE_BUCKETS").is_some_and(|v| v == "1"),
            host: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(host_threads)
                    .thread_name(|i| format!("teasar-host-{i}"))
                    .build()?,
            ),
            stats: CallStats::default(),
        })
    }

    /// The host pool large pieces and compatibility mode run on
    pub fn host_pool(&self) -> &rayon::ThreadPool {
        &self.host
    }

    /// The threads host pieces run on
    pub fn host_threads(&self) -> usize {
        self.host.current_num_threads()
    }

    /// The skeletons of every slab of `labels`, each with its own effective `fix_borders`; pieces over `large_default`
    /// voxels fall back to the host when `TEASAR_LARGE_PIECE` is unset (spec part B §3.3)
    pub fn run(
        &mut self,
        gpu: &Gpu,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        choices: &Choices,
        large_default: u32,
    ) -> Result<Vec<Packed>> {
        self.large_limit = self.large_piece.unwrap_or(large_default);
        let a = skel.anisotropy;
        ensure!(
            a[0] == a[1] && a[1] == a[2] && a[0] > 0.0 && (a[0].log2().fract() == 0.0),
            "the Metal skeletoniser takes an isotropic power-of-two anisotropy, got {a:?}"
        );
        // a CPU-only exploration choice; host pieces would follow it and GPU lanes would not
        ensure!(
            !choices.ball_connected,
            "the Metal skeletoniser has no connected-ball invalidation (ballfill=connected)"
        );
        let mut out = Vec::with_capacity(labels.len());
        let mut start = 0;
        while start < labels.len() {
            let mut end = start + 1;
            let mut voxels = labels[start].data.len();
            while end < labels.len() && voxels + labels[end].data.len() <= self.batch_voxels {
                voxels += labels[end].data.len();
                end += 1;
            }
            ensure!(voxels < (u32::MAX / 4) as usize, "a slab of {voxels} pooled voxels is too large for the GPU path");
            out.extend(self.batch(gpu, &labels[start..end], skel, choices)?);
            self.profile.batches += 1;
            start = end;
        }
        Ok(out)
    }

    fn batch(
        &mut self,
        gpu: &Gpu,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        choices: &Choices,
    ) -> Result<Vec<Packed>> {
        let t0 = Instant::now();
        let b = self.pieces(gpu, labels, skel)?;
        if b.kept.is_empty() {
            return Ok(labels.iter().map(|_| Packed::empty()).collect());
        }
        let t1 = Instant::now();
        let st = self.slab_stages(gpu, labels, &b, skel, choices)?;
        let t2 = Instant::now();
        let pool = Arc::clone(&self.host);
        let jobs = pool.install(|| host_jobs(&b, &st));
        let (states, host) = std::thread::scope(|sc| {
            let worker = sc.spawn(|| {
                let start = Instant::now();
                let out = pool.install(|| run_host(&b, &st, &jobs, skel, choices));
                (out, start.elapsed().as_secs_f64())
            });
            let states = self.path_loop(gpu, &b, &st, skel, choices, 0);
            (states, worker.join())
        });
        let states = states?;
        let Ok((host, host_wall)) = host else { bail!("a host piece thread panicked") };
        self.profile.host_wall_secs += host_wall;
        self.profile.host_pieces += host.len() as u64;
        let host_voxels: usize = jobs.iter().map(|j| j.vox.len()).sum();
        self.stats.add_cpu(host.len() as u64, host_voxels as u64, host.iter().map(|h| h.secs).sum());
        let gpu = b.pieces.iter().filter(|p| p.host == 0);
        self.stats.add_gpu(gpu.clone().count() as u64, gpu.map(|p| u64::from(p.m)).sum());
        self.profile.host_thread_secs += host.iter().map(|h| h.secs).sum::<f64>();
        for h in &host {
            self.profile.ordinary += h.skel.counts.ordinary;
            self.profile.mandatory += h.skel.counts.mandatory;
        }
        let t3 = Instant::now();
        let out = assemble(&b, &st, &states, &host, skel);
        let t4 = Instant::now();
        for (k, (a, z)) in [(t0, t1), (t1, t2), (t2, t3), (t3, t4)].into_iter().enumerate() {
            self.profile.secs[k] += (z - a).as_secs_f64();
        }
        Ok(out)
    }

    /// Wait 1: the 26-connected pieces of the batch and its piece table
    fn pieces(&mut self, gpu: &Gpu, labels: &[LabelsRef<'_>], skel: &SkelSettings) -> Result<Batch> {
        let mut slabs = Vec::with_capacity(labels.len());
        let mut n = 0usize;
        for l in labels {
            let [z, y, x] = l.shape.map(|s| s as u32);
            slabs.push(SkSlab { z, y, x, base: n as u32 });
            n += l.data.len();
        }
        let lab = gpu.alloc(n * 4)?;
        // SAFETY: a fresh buffer no command uses
        let host = unsafe { &mut lab.host_mut::<u32>()[..n] };
        let mut chunks = Vec::with_capacity(labels.len());
        let mut rest = host;
        for l in labels {
            let (head, tail) = rest.split_at_mut(l.data.len());
            chunks.push((head, l.data));
            rest = tail;
        }
        let nz: usize = chunks
            .into_par_iter()
            .map(|(dst, src)| {
                dst.copy_from_slice(src);
                src.iter().filter(|&&v| v != 0).count()
            })
            .sum();
        // pieces are at most the labelled voxels
        let cap = nz.max(1);

        let slab_buf = gpu.upload(&slabs)?;
        let par = gpu.alloc(n * 4)?;
        let num = gpu.alloc((n + 1) * 4)?;
        let pid = gpu.alloc(n * 4)?;
        let size = gpu.alloc(cap * 4)?;
        let ffirst = gpu.alloc(cap * 4)?;
        let plabel = gpu.alloc(cap * 4)?;
        let pfirst = gpu.alloc(cap * 4)?;
        let ns = Arg::U32(slabs.len() as u32);

        gpu.dispatch_1d("sk_ccl_init", n, 256, &[(&par).into()])?;
        gpu.dispatch_1d("sk_ccl_union", n, 256, &[(&lab).into(), (&slab_buf).into(), (&par).into(), ns])?;
        gpu.dispatch_1d("sk_ccl_flag", n, 256, &[(&lab).into(), (&par).into(), (&num).into()])?;
        scan(gpu, &num, n, true)?;
        fill(gpu, &size, cap, 0)?;
        fill(gpu, &ffirst, cap, NONE)?;
        gpu.dispatch_1d(
            "sk_ccl_pid",
            n,
            256,
            &[
                (&lab).into(),
                (&par).into(),
                (&num).into(),
                (&slab_buf).into(),
                (&pid).into(),
                (&size).into(),
                (&ffirst).into(),
                (&plabel).into(),
                (&pfirst).into(),
                ns,
            ],
        )?;
        gpu.synchronize()?;

        // SAFETY: the batch's work so far has completed
        let np = unsafe { num.host::<u32>()[n] } as usize;
        // SAFETY: as above
        let (size, ff, plab, pfirst) = unsafe {
            (
                &size.host::<u32>()[..np],
                &ffirst.host::<u32>()[..np],
                &plabel.host::<u32>()[..np],
                &pfirst.host::<u32>()[..np],
            )
        };

        // pieces are numbered by first voxel, so each slab's pieces are a contiguous run
        let mut slab_pieces: Vec<u32> =
            slabs.iter().map(|sl| pfirst.partition_point(|&f| f < sl.base) as u32).collect();
        slab_pieces.push(np as u32);

        let mut kidx = vec![NONE; np];
        let mut kept: Vec<Kept> = Vec::new();
        for s in 0..slabs.len() {
            for p in slab_pieces[s]..slab_pieces[s + 1] {
                let m = size[p as usize];
                if m > skel.dust_threshold {
                    kept.push(Kept { pid: p, slab: s as u32, m });
                }
            }
        }
        // largest first, so the big pieces start early and a dispatch's lanes have similar work
        kept.sort_by_key(|k| (Reverse(k.m), k.pid));
        let mut pieces = Vec::with_capacity(kept.len());
        let (mut ws, mut valid) = (0usize, 0usize);
        for (k, kp) in kept.iter().enumerate() {
            kidx[kp.pid as usize] = k as u32;
            let fallback = kp.m > self.large_limit;
            ensure!(
                !fallback || self.cpu_fallback,
                "TEASAR_CPU_FALLBACK=0: a piece of {} voxels is over TEASAR_LARGE_PIECE ({})",
                kp.m,
                self.large_limit
            );
            self.profile.fallback_pieces += u64::from(fallback);
            let host = fallback || kp.m > self.cpu_piece;
            let sl = slabs[kp.slab as usize];
            let f = ff[kp.pid as usize];
            let (fz, fy, fx) = (f % sl.z, (f / sl.z) % sl.y, f / (sl.z * sl.y));
            pieces.push(SkPiece {
                pid: kp.pid,
                slab: kp.slab,
                m: kp.m,
                ws: ws as u32,
                valid: valid as u32,
                first: pfirst[kp.pid as usize],
                ffirst: sl.base + (fz * sl.y + fy) * sl.x + fx,
                host: u32::from(host),
                ..SkPiece::default()
            });
            if host {
                ws += kp.m as usize;
            } else {
                ws += WS_ARRAYS * kp.m as usize;
                valid += (kp.m as usize).div_ceil(32);
            }
        }
        ensure!(ws < u32::MAX as usize, "a batch's piece workspace of {ws} words overflows its 32-bit offsets");
        let ws_buf = gpu.alloc(ws * 4)?;
        self.profile.pieces += kept.len() as u64;

        Ok(Batch {
            plabel: plab.to_vec(),
            psize: size.to_vec(),
            pff: ff.to_vec(),
            kidx,
            kept,
            pieces,
            slab_pieces,
            valid_words: valid,
            slabs,
            slab_buf,
            pid,
            ws: ws_buf,
        })
    }

    /// Wait 2: voxel lists, distance transforms and border targets
    fn slab_stages(
        &mut self,
        gpu: &Gpu,
        labels: &[LabelsRef<'_>],
        b: &Batch,
        skel: &SkelSettings,
        choices: &Choices,
    ) -> Result<Staged> {
        let n: usize = labels.iter().map(|l| l.data.len()).sum();
        let nk = b.kept.len();
        let local = gpu.alloc(n * 4)?;
        let valid = gpu.alloc(b.valid_words.max(1) * 4)?;
        let fillc = gpu.alloc(nk * 4)?;
        let maxpen = gpu.alloc(nk * 4)?;
        let kidx = gpu.upload(&b.kidx)?;
        let pieces = gpu.upload(&b.pieces)?;
        fill(gpu, &valid, b.valid_words, NONE)?;
        fill(gpu, &fillc, nk, 0)?;
        fill(gpu, &maxpen, nk, 0)?;
        gpu.dispatch_1d(
            "sk_scatter",
            n,
            256,
            &[(&b.pid).into(), (&kidx).into(), (&pieces).into(), (&fillc).into(), (&b.ws).into(), (&local).into()],
        )?;

        let (views, layers, f) = plan_views(b, labels, skel, choices);
        let fields = gpu.alloc(f.total * 4)?;
        let temp = gpu.alloc(f.total * 4)?;
        let last = views.last().expect("every slab has a view");
        let nlines = (last.line_off + last.n[0] * last.n[1]) as usize;
        let nvox = (last.vox_off + last.n[0] * last.n[1] * last.n[2]) as usize;
        let vbuf = gpu.upload(&views)?;
        let nv = Arg::U32(views.len() as u32);
        gpu.dispatch_1d("sk_edt_x", nlines, 64, &[(&vbuf).into(), (&b.pid).into(), (&fields).into(), nv])?;
        gpu.dispatch_1d(
            "sk_edt_y",
            nvox,
            256,
            &[(&vbuf).into(), (&b.pid).into(), (&fields).into(), (&temp).into(), nv],
        )?;
        gpu.dispatch_1d(
            "sk_edt_z",
            nvox,
            256,
            &[(&vbuf).into(), (&b.pid).into(), (&temp).into(), (&fields).into(), nv],
        )?;
        gpu.dispatch_1d(
            "sk_maxpen",
            n,
            256,
            &[(&b.pid).into(), (&kidx).into(), fields.at::<u32>(f.pen as usize), (&maxpen).into()],
        )?;

        let records = if layers.is_empty() {
            gpu.synchronize()?;
            Vec::new()
        } else {
            face_records(gpu, b, &layers, &kidx, &fields, f, choices)?
        };

        // each piece's targets in tracing order: layers in order, regions by lowest index, duplicates dropped
        let mut recs = records;
        recs.sort_unstable_by_key(|r| (r[0], r[1], r[2]));
        let mut pieces_host = b.pieces.clone();
        let mut mand: Vec<u32> = Vec::with_capacity(recs.len());
        let mut at = 0;
        while at < recs.len() {
            let k = recs[at][0] as usize;
            let start = mand.len();
            while at < recs.len() && recs[at][0] as usize == k {
                let t = recs[at][3];
                if !mand[start..].contains(&t) {
                    mand.push(t);
                }
                at += 1;
            }
            pieces_host[k].mand_off = start as u32;
            pieces_host[k].mand_n = (mand.len() - start) as u32;
        }
        mand.push(NONE);
        drop(temp);

        Ok(Staged {
            local,
            valid,
            fields,
            f,
            maxpen,
            mand: gpu.upload(&mand)?,
            pieces: gpu.upload(&pieces_host)?,
            pieces_host,
            mand_host: mand,
        })
    }

    /// Wait 3: the path loop of every kept piece, resumed until all are done
    fn path_loop(
        &mut self,
        gpu: &Gpu,
        b: &Batch,
        st: &Staged,
        skel: &SkelSettings,
        choices: &Choices,
        extra_flags: u32,
    ) -> Result<Vec<SkState>> {
        let nk = b.kept.len();
        let states = gpu.upload(&vec![SkState::default(); nk])?;
        let a = skel.anisotropy[0];
        let root = match choices.root {
            RootStart::COrder => 0,
            RootStart::FOrder => 1,
            RootStart::DbfMax => 2,
            RootStart::Face => 3,
            RootStart::FaceLast => 4,
            RootStart::FaceFar => 5,
        };
        let bit = |on: bool, f: u32| if on { f } else { 0 };
        let flags = bit(choices.count_target, F_COUNT_TARGET)
            | bit(choices.mandatory_first, F_MAND_FIRST)
            | bit(choices.daf_term, F_DAF)
            | bit(choices.penalty_form == PenaltyForm::Unsquared, F_UNSQUARED)
            | (root << F_ROOT_SHIFT)
            | extra_flags;
        let prm = SkParams {
            a: a as f32,
            inv_a: (1.0 / a) as f32,
            scale: skel.scale as f32,
            constant: skel.constant as f32,
            len: step_lengths(a),
            flags,
            budget: self.budget,
            zero: 0,
            nslabs: b.slabs.len() as u32,
            inv_field: st.f.inv,
            pen_field: st.f.pen,
            dbf_field: st.f.dbf,
            pad: 0,
        };

        // pieces are sorted by size, largest first; with buckets on, each size bucket runs to completion alone
        let lanes: Vec<u32> = (0..nk as u32).filter(|&k| b.pieces[k as usize].host == 0).collect();
        let groups: Vec<(usize, Vec<u32>)> = if self.buckets && !lanes.is_empty() {
            let sizes: Vec<u32> = lanes.iter().map(|&k| b.kept[k as usize].m).collect();
            bucket_split(&sizes)
                .into_iter()
                .map(|(g, v)| (g, v.into_iter().map(|i| lanes[i as usize]).collect()))
                .collect()
        } else {
            vec![(usize::MAX, lanes)]
        };
        let mut dispatches = 0u64;
        for (bucket, mut active) in groups.iter().cloned() {
            let t = Instant::now();
            if bucket < 4 {
                self.profile.bucket_pieces[bucket] += active.len() as u64;
            }
            while !active.is_empty() {
                let abuf = gpu.upload(&active)?;
                gpu.dispatch_1d(
                    "sk_run",
                    active.len(),
                    RUN_GROUP,
                    &[
                        (&st.pieces).into(),
                        (&states).into(),
                        (&abuf).into(),
                        (&b.ws).into(),
                        (&st.valid).into(),
                        (&b.pid).into(),
                        (&st.local).into(),
                        (&st.fields).into(),
                        (&st.mand).into(),
                        (&b.slab_buf).into(),
                        (&st.maxpen).into(),
                        Arg::Bytes(bytemuck::bytes_of(&prm)),
                        Arg::U32(active.len() as u32),
                    ],
                )?;
                gpu.synchronize()?;
                dispatches += 1;
                // SAFETY: the dispatch has completed
                let s = unsafe { states.host::<SkState>() };
                if let Some(&k) = active.iter().find(|&&k| s[k as usize].phase == PH_FAIL) {
                    let kp = b.kept[k as usize];
                    bail!("the GPU path loop broke an invariant on piece {} ({} voxels)", kp.pid, kp.m);
                }
                for &k in &active {
                    let ph = s[k as usize].phase;
                    if ph != PH_DONE {
                        self.profile.phase_pauses[ph as usize] += 1;
                    }
                }
                active.retain(|&k| s[k as usize].phase != PH_DONE);
            }
            if bucket < 4 {
                self.profile.bucket_secs[bucket] += t.elapsed().as_secs_f64();
            }
        }
        // each group's first dispatch is not a resume
        self.profile.resume_dispatches += dispatches.saturating_sub(groups.len() as u64);

        // SAFETY: every dispatch has completed
        let out = unsafe { states.host::<SkState>()[..nk].to_vec() };
        for (k, s) in out.iter().enumerate() {
            if b.pieces[k].host != 0 {
                continue;
            }
            self.profile.paused += u64::from(s.runs > 1);
            self.profile.ordinary += u64::from(s.ordinary);
            self.profile.steps += u64::from(s.steps);
            self.profile.mandatory += u64::from(st.pieces_host[k].mand_n);
        }
        Ok(out)
    }
}

/// The kept pieces (indices into a size-descending list) split at the median, p90 and p99 sizes, largest bucket
/// first
fn bucket_split(sizes: &[u32]) -> Vec<(usize, Vec<u32>)> {
    let n = sizes.len();
    // sizes descend, so the q quantile is at index (1 - q) n
    let at = |q: f64| sizes[((1.0 - q) * n as f64).floor().min((n - 1) as f64) as usize];
    let (med, p90, p99) = (at(0.5), at(0.9), at(0.99));
    let mut out: Vec<(usize, Vec<u32>)> = (0..4).rev().map(|b| (b, Vec::new())).collect();
    for (k, &m) in sizes.iter().enumerate() {
        let b = if m <= med {
            0
        } else if m <= p90 {
            1
        } else if m <= p99 {
            2
        } else {
            3
        };
        out[3 - b].1.push(k as u32);
    }
    out.retain(|(_, v)| !v.is_empty());
    out
}

/// The views of the distance transforms and the face layers of a batch, and where each field lives
fn plan_views(
    b: &Batch,
    labels: &[LabelsRef<'_>],
    skel: &SkelSettings,
    choices: &Choices,
) -> (Vec<SkView>, Vec<SkLayer>, Fields) {
    let n: usize = labels.iter().map(|l| l.data.len()).sum();
    let p2 = choices.penalty == Penalty::P2;
    let thin = choices.inval == Inval::Thin;
    let mut next = n as u32;
    let pen = if p2 {
        next += n as u32;
        n as u32
    } else {
        0
    };
    let inv = match choices.inval {
        Inval::Dbf => 0,
        Inval::Penalty => pen,
        Inval::Thin => {
            next += n as u32;
            next - n as u32
        }
    };
    let layer_field = next;

    let mut views = Vec::new();
    let mut layers = Vec::new();
    let (mut vox, mut lines, mut loff) = (0u32, 0u32, 0u32);
    let mut push = |views: &mut Vec<SkView>, mut v: SkView| {
        v.vox_off = vox;
        v.line_off = lines;
        vox += v.n[0] * v.n[1] * v.n[2];
        lines += v.n[0] * v.n[1];
        views.push(v);
    };
    for (s, sl) in b.slabs.iter().enumerate() {
        let shape = [sl.z, sl.y, sl.x];
        let strides = [sl.y * sl.x, sl.x, 1];
        let np = b.slab_pieces[s + 1] - b.slab_pieces[s];
        let total = sl.z * sl.y * sl.x;
        // a piece that fills the array measures to the positions just outside it (§2.2)
        let full = np == 1 && b.psize[b.slab_pieces[s] as usize] == total;
        // bit k of a view's border is axis k
        let bits = |f: &dyn Fn(u32) -> bool| (0..3).fold(0u32, |acc, k| acc | (u32::from(f(shape[k])) << k));
        let whole = |out: u32, border: u32| SkView {
            id_base: sl.base,
            s: strides,
            n: shape,
            out: out + sl.base,
            border,
            ..SkView::default()
        };
        let dbf_border = if full { 7 } else { 0 };
        push(&mut views, whole(0, dbf_border));
        if p2 {
            // the axes longer than 2 (§2.5); with none, the DBF's own border
            let long = bits(&|len| len > 2);
            push(&mut views, whole(pen, if long == 0 { dbf_border } else { long }));
        }
        if thin {
            push(&mut views, whole(inv, bits(&|len| full || len >= 2)));
        }

        if !crate::effective_fix_borders(skel.fix_borders, labels[s].shape) {
            continue;
        }
        let axes = if choices.faces_xyz { [2, 1, 0] } else { [0, 1, 2] };
        for axis in axes {
            let mut idx = if choices.faces_high_first { vec![shape[axis] - 1, 0] } else { vec![0, shape[axis] - 1] };
            idx.dedup();
            for index in idx {
                let (u_ax, v_ax) = match axis {
                    0 => (1, 2),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                let (h, w) = (shape[u_ax], shape[v_ax]);
                let vbase = sl.base + index * strides[axis];
                let layer = SkLayer {
                    off: loff,
                    h,
                    w,
                    vbase,
                    su: strides[u_ax],
                    sv: strides[v_ax],
                    order: layers.len() as u32,
                    pad: 0,
                };
                push(
                    &mut views,
                    SkView {
                        id_base: vbase,
                        s: [0, strides[u_ax], strides[v_ax]],
                        n: [1, h, w],
                        out: layer_field + loff,
                        border: 0b110,
                        ..SkView::default()
                    },
                );
                layers.push(layer);
                loff += h * w;
            }
        }
    }
    let f = Fields { dbf: 0, pen, inv, layers: layer_field, total: (layer_field + loff) as usize };
    (views, layers, f)
}

/// The border-target records of every face region: (kept piece, layer order, root's layer position, target voxel)
fn face_records(
    gpu: &Gpu,
    b: &Batch,
    layers: &[SkLayer],
    kidx: &Buf,
    fields: &Buf,
    f: Fields,
    choices: &Choices,
) -> Result<Vec<[u32; 4]>> {
    let last = layers.last().expect("not empty");
    let total = (last.off + last.h * last.w) as usize;
    let lbuf = gpu.upload(layers)?;
    let par = gpu.alloc(total * 4)?;
    let stats = gpu.alloc((4 + PICK_WORDS as usize) * total * 4)?;
    let count = gpu.upload(&[0u32])?;
    let nl = Arg::U32(layers.len() as u32);
    let tot = Arg::U32(total as u32);
    let dist = fields.at::<u32>(f.layers as usize);
    gpu.dispatch_1d(
        "sk_face_init",
        total,
        256,
        &[(&lbuf).into(), (&b.pid).into(), kidx.into(), (&par).into(), (&stats).into(), nl, tot],
    )?;
    gpu.dispatch_1d(
        "sk_face_union",
        total,
        256,
        &[(&lbuf).into(), (&b.pid).into(), kidx.into(), (&par).into(), nl, Arg::U32(u32::from(choices.face_8))],
    )?;
    gpu.dispatch_1d("sk_face_stats", total, 256, &[(&lbuf).into(), (&par).into(), dist, (&stats).into(), nl, tot])?;
    // skel.metal `PICK_*`
    let mode = Arg::U32(match choices.face_pick {
        FacePick::Readme => 0,
        FacePick::Centroid => 1,
        FacePick::CentroidHigh => 2,
        FacePick::Lowest => 3,
    });
    for pass in 0..PICK_WORDS {
        gpu.dispatch_1d(
            "sk_face_pick",
            total,
            256,
            &[(&lbuf).into(), (&par).into(), dist, (&stats).into(), nl, tot, Arg::U32(pass), mode],
        )?;
    }
    // regions are far fewer than layer voxels; a batch that has more is emitted again with room
    let mut cap = total / 8 + 1024;
    loop {
        let recs = gpu.alloc(cap * 16)?;
        // SAFETY: no pending command uses the counter between waits
        unsafe { count.host_mut::<u32>()[0] = 0 };
        gpu.dispatch_1d(
            "sk_face_emit",
            total,
            256,
            &[
                (&lbuf).into(),
                (&par).into(),
                (&b.pid).into(),
                kidx.into(),
                (&stats).into(),
                (&count).into(),
                (&recs).into(),
                nl,
                tot,
                mode,
            ],
        )?;
        gpu.synchronize()?;
        // SAFETY: completed
        let got = unsafe { count.host::<u32>()[0] } as usize;
        if got <= cap {
            // SAFETY: completed
            return Ok(unsafe { recs.host::<[u32; 4]>()[..got].to_vec() });
        }
        cap = got;
    }
}

/// Encodes an exclusive prefix sum of `data[0, n)` in place; with `with_total`, `data[n]` gets the total
fn scan(gpu: &Gpu, data: &Buf, n: usize, with_total: bool) -> Result<()> {
    let flag = Arg::U32(u32::from(with_total));
    let groups = n.div_ceil(SCAN_BLOCK);
    if groups <= 1 {
        let zero = gpu.upload(&[0u32])?;
        return gpu.dispatch_groups(
            "scan_apply",
            1,
            SCAN_THREADS,
            &[data.into(), (&zero).into(), Arg::U32(n as u32), flag],
        );
    }
    let sums = gpu.alloc(groups * 4)?;
    gpu.dispatch_groups("scan_reduce", groups, SCAN_THREADS, &[data.into(), (&sums).into(), Arg::U32(n as u32)])?;
    scan(gpu, &sums, groups, false)?;
    gpu.dispatch_groups("scan_apply", groups, SCAN_THREADS, &[data.into(), (&sums).into(), Arg::U32(n as u32), flag])
}

fn fill(gpu: &Gpu, data: &Buf, n: usize, value: u32) -> Result<()> {
    gpu.dispatch_1d("fill_u32", n, 256, &[data.into(), Arg::U32(value)])
}

/// A host piece's inputs, copied out of device memory before the GPU loop writes the workspace again
struct HostJob {
    k: usize,
    /// slab-local voxel indices, increasing
    vox: Vec<u32>,
    /// for each GPU list slot (the `local` the scatter wrote), the voxel's rank in `vox`
    rank: Vec<u32>,
    /// border targets as ranks, in tracing order
    mandatory: Vec<usize>,
}

/// A host piece's skeleton, with slab-local vertices
struct HostSkel {
    k: usize,
    skel: PieceSkel,
    secs: f64,
}

fn host_jobs(b: &Batch, st: &Staged) -> Vec<HostJob> {
    // SAFETY: the slab stages have completed and no command is pending
    let ws = unsafe { b.ws.host::<u32>() };
    // SAFETY: as above
    let local = unsafe { st.local.host::<u32>() };
    (0..b.kept.len())
        .into_par_iter()
        .filter(|&k| b.pieces[k].host != 0)
        .map(|k| {
            let pc = st.pieces_host[k];
            let base = b.slabs[pc.slab as usize].base;
            let list = &ws[pc.ws as usize..pc.ws as usize + pc.m as usize];
            let mut slots: Vec<u32> = (0..pc.m).collect();
            slots.sort_unstable_by_key(|&s| list[s as usize]);
            let mut rank = vec![0u32; pc.m as usize];
            for (r, &s) in slots.iter().enumerate() {
                rank[s as usize] = r as u32;
            }
            let vox = slots.iter().map(|&s| list[s as usize] - base).collect();
            let mand = &st.mand_host[pc.mand_off as usize..(pc.mand_off + pc.mand_n) as usize];
            let mandatory = mand.iter().map(|&g| rank[local[g as usize] as usize] as usize).collect();
            HostJob { k, vox, rank, mandatory }
        })
        .collect()
}

/// The host pieces' path loops, on the rayon pool; reads only device arrays the GPU loop does not write
fn run_host(b: &Batch, st: &Staged, jobs: &[HostJob], skel: &SkelSettings, choices: &Choices) -> Vec<HostSkel> {
    // SAFETY: the GPU loop only reads these buffers while this runs
    let (pid, local, fields) = unsafe { (b.pid.host::<u32>(), st.local.host::<u32>(), st.fields.host::<u32>()) };
    jobs.par_iter()
        .map(|j| {
            let t = Instant::now();
            let pc = st.pieces_host[j.k];
            let sl = b.slabs[pc.slab as usize];
            let (base, n) = (sl.base as usize, (sl.z * sl.y * sl.x) as usize);
            let field = |f: u32| &fields[f as usize + base..f as usize + base + n];
            let ctx = PieceCtx {
                shape: [sl.z, sl.y, sl.x].map(|s| s as usize),
                skel,
                choices,
                of_voxel: &pid[base..base + n],
                me: pc.pid,
                vox: &j.vox,
                n_dbf: field(st.f.dbf),
                n_pen: field(st.f.pen),
                n_inv: field(st.f.inv),
                mandatory: j.mandatory.clone(),
            };
            let local = &local[base..base + n];
            let skel = trace_piece(&ctx, |h| j.rank[local[h] as usize]);
            HostSkel { k: j.k, skel, secs: t.elapsed().as_secs_f64() }
        })
        .collect()
}

/// Every slab's packed skeletons from the pieces' vertex lists (spec part B §2.8)
fn assemble(b: &Batch, st: &Staged, states: &[SkState], host: &[HostSkel], skel: &SkelSettings) -> Vec<Packed> {
    let mut by_k = vec![usize::MAX; b.kept.len()];
    for (i, h) in host.iter().enumerate() {
        by_k[h.k] = i;
    }

    // SAFETY: the batch's work has completed
    let ws = unsafe { b.ws.host::<u32>() };
    // SAFETY: as above
    let fields = unsafe { st.fields.host::<u32>() };
    let a = skel.anisotropy[0];
    (0..b.slabs.len())
        .into_par_iter()
        .map(|s| {
            let sl = b.slabs[s];
            let range = b.slab_pieces[s] as usize..b.slab_pieces[s + 1] as usize;
            // labels with a kept piece, ordered by the first voxel of a kept piece in a Fortran-order scan: dust
            // pieces do not count (the fixtures' order on 5 of the 144 villa_default slabs; spec §2.8 is silent)
            let mut first: Vec<(u32, u32)> =
                range.clone().filter(|&p| b.kidx[p] != NONE).map(|p| (b.plabel[p], b.pff[p])).collect();
            first.sort_unstable();
            let mut by_label: Vec<(u32, u32)> = Vec::new();
            for &(l, f) in &first {
                if by_label.last().is_none_or(|&(pl, _)| pl != l) {
                    by_label.push((l, f));
                }
            }
            let mut kept: Vec<(u32, u32)> =
                range.clone().filter(|&p| b.kidx[p] != NONE).map(|p| (b.plabel[p], p as u32)).collect();
            kept.sort_unstable();
            let mut ids: Vec<(u32, u32)> = by_label
                .into_iter()
                .filter(|&(l, _)| kept.binary_search_by_key(&l, |&(kl, _)| kl).is_ok())
                .map(|(l, f)| (f, l))
                .collect();
            ids.sort_unstable();
            if ids.is_empty() {
                return Packed::empty();
            }

            let mut out =
                Packed { ids: Vec::with_capacity(ids.len()), v_off: vec![0], e_off: vec![0], ..Packed::default() };
            let shape = [sl.z as usize, sl.y as usize, sl.x as usize];
            let (mut verts, mut edges) = (Vec::new(), Vec::new());
            for &(_, label) in &ids {
                verts.clear();
                edges.clear();
                let lo = kept.partition_point(|&(l, _)| l < label);
                for &(_, p) in kept[lo..].iter().take_while(|&&(l, _)| l == label) {
                    let k = b.kidx[p as usize] as usize;
                    let pc = b.pieces[k];
                    let m = pc.m as usize;
                    let dbf = |g: u32| radius(fields[(st.f.dbf + sl.base + g) as usize], a);
                    if by_k[k] != usize::MAX {
                        let h = &host[by_k[k]].skel;
                        verts.extend(h.verts.iter().map(|&g| (g, dbf(g))));
                        edges.extend(h.edges.iter().map(|&[u, v]| [h.verts[u as usize], h.verts[v as usize]]));
                        continue;
                    }
                    let w = &ws[pc.ws as usize..pc.ws as usize + WS_ARRAYS * m];
                    let nverts = states[k].nverts as usize;
                    // slab-local voxel of vertex v
                    let vox = |v: usize| w[WS_VOX * m + w[WS_VERT * m + v] as usize] - sl.base;
                    verts.extend((0..nverts).map(|v| (vox(v), dbf(vox(v)))));
                    edges.extend((1..nverts).map(|v| [vox(v), vox(w[WS_NEXT * m + v] as usize)]));
                }
                out.push_sorted(label, &mut verts, &mut edges, shape, [a; 3]);
            }
            out
        })
        .collect()
}
