//! Host side of the pre-skeleton kernels (`shaders/metal/prep.metal`): batch planning, scratch, encoding, one wait per
//! call
//!
//! A call's slabs are split into internal batches of at most `batch_cells` cells (a larger slab is a batch of its
//! own). Every batch runs the same ~17 dispatches over all its slabs at once, reusing one set of scratch buffers;
//! the queue's in-order execution keeps batches from overlapping, so the host waits once for the whole call. Outputs
//! go to one call-wide buffer, copied out after the wait

use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use color_eyre::eyre::{Result, ensure};
use rayon::prelude::*;

use super::gpu::{Arg, Buf, Gpu};
use crate::geom::SlabGeom;
use crate::{Labels, PrepSettings};

/// Cells of one internal batch, by default; ~16 bytes of scratch each
pub const BATCH_CELLS: usize = 1 << 23;

/// Elements one scan threadgroup covers (SCAN_BLOCK in common.metal)
const SCAN_BLOCK: usize = 1024;
const SCAN_THREADS: usize = 256;

/// Slab-local linear indices per rank bucket (BUCKET_SHIFT in common.metal)
const BUCKET_VOXELS: u64 = 1 << 9;

/// Mirrors `SlabDesc` in common.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SlabDesc {
    z0: u32,
    y0: u32,
    x0: u32,
    zs: u32,
    ys: u32,
    xs: u32,
    cz: u32,
    cy: u32,
    cx: u32,
    pz: u32,
    py: u32,
    px: u32,
    cell_base: u32,
    out_base: u32,
    bucket_base: u32,
    pad: u32,
}

/// Mirrors `PrepParams` in common.metal
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PrepParams {
    can_xp: u64,
    can_xm: u64,
    can_yp: u64,
    can_ym: u64,
    face_xp: u64,
    face_yp: u64,
    face_zp: u64,
    ds: u32,
    ds2: u32,
    nslabs: u32,
    cells: u32,
    outs: u32,
    buckets: u32,
    cap: u32,
    row_bytes: u32,
    block_rows: u32,
    min_count: u32,
    max_count: u32,
    pad: u32,
}

/// The bits of a ds^3 cell mask whose voxel (dz, dy, dx) satisfies `pred`
fn cell_bits(ds: u32, pred: impl Fn(u32, u32, u32) -> bool) -> u64 {
    let mut m = 0u64;
    for dz in 0..ds {
        for dy in 0..ds {
            for dx in 0..ds {
                if pred(dz, dy, dx) {
                    m |= 1 << ((dz * ds + dy) * ds + dx);
                }
            }
        }
    }
    m
}

/// The packed block as the kernels see it
pub struct BlockArg<'a> {
    pub buf: &'a Buf,
    pub offset: usize,
    pub rows: usize,
    pub row_bytes: usize,
}

/// One internal batch: a run of the call's slabs
struct Batch {
    slabs: Range<usize>,
    cells: usize,
    outs: usize,
    buckets: usize,
    /// first pooled voxel of the batch in the call's output
    out_at: usize,
}

/// Scratch for the largest batch, reused by every batch of a call and across calls
struct Scratch {
    cells: usize,
    nodes: usize,
    buckets: usize,
    off: Buf,
    parent: Buf,
    key: Buf,
    sorted: Buf,
    bucket: Buf,
    /// block sums for each level of a scan
    levels: Vec<Buf>,
    zero: Buf,
}

impl Scratch {
    fn new(gpu: &Gpu, cells: usize, nodes: usize, buckets: usize) -> Result<Self> {
        let mut levels = Vec::new();
        let mut n = (cells + 1).max(buckets);
        while n > SCAN_BLOCK {
            n = n.div_ceil(SCAN_BLOCK);
            levels.push(gpu.alloc(n * 4)?);
        }
        Ok(Self {
            cells,
            nodes,
            buckets,
            off: gpu.alloc((cells + 1) * 4)?,
            parent: gpu.alloc(nodes * 4)?,
            key: gpu.alloc(nodes * 4)?,
            sorted: gpu.alloc(nodes * 4)?,
            bucket: gpu.alloc(buckets * 4)?,
            levels,
            zero: gpu.upload(&[0u32])?,
        })
    }

    fn fits(&self, cells: usize, nodes: usize, buckets: usize) -> bool {
        cells <= self.cells && nodes <= self.nodes && buckets <= self.buckets
    }
}

/// The pre-skeleton pipeline on one device
pub struct Prep {
    scratch: Option<Scratch>,
    pub batch_cells: usize,
}

impl Prep {
    pub fn new() -> Self {
        Self { scratch: None, batch_cells: BATCH_CELLS }
    }

    /// The pooled labels of each slab; one wait for the whole call, plus one more only if a batch overflowed its
    /// node capacity
    pub fn run(
        &mut self,
        gpu: &Gpu,
        block: &BlockArg<'_>,
        slabs: &[SlabGeom],
        prep: &PrepSettings,
    ) -> Result<Vec<Labels>> {
        let ds = prep.downsample;
        ensure!((1..=4).contains(&ds), "the Metal pre-skeleton kernels take downsample 1 to 4, got {ds}");
        let (descs, batches) = self.plan(slabs, ds)?;
        let outs: usize = batches.iter().map(|b| b.outs).sum();
        let max_cells = batches.iter().map(|b| b.cells).max().unwrap_or(0);
        let max_buckets = batches.iter().map(|b| b.buckets).max().unwrap_or(0);
        // one node per cell covers real data by a wide margin (pieces sit in under half of the cells); a batch that
        // needs more is rerun below with exactly what it needs
        self.ensure_scratch(gpu, max_cells, max_cells.max(1024), max_buckets)?;

        let desc_buf = gpu.upload(&descs)?;
        let out = gpu.alloc(outs * 4)?;
        let totals = gpu.alloc(batches.len() * 4)?;
        for (i, b) in batches.iter().enumerate() {
            self.encode(gpu, block, &desc_buf, &out, &totals, i, b, prep)?;
        }
        gpu.synchronize()?;

        // SAFETY: the call's work has completed
        let got: Vec<u32> = unsafe { totals.host::<u32>() }[..batches.len()].to_vec();
        let nodes = self.scratch.as_ref().expect("allocated above").nodes;
        let redo: Vec<usize> = (0..batches.len()).filter(|&i| got[i] as usize > nodes).collect();
        if !redo.is_empty() {
            let need = redo.iter().map(|&i| got[i] as usize).max().expect("non-empty");
            self.ensure_scratch(gpu, max_cells, need, max_buckets)?;
            for &i in &redo {
                self.encode(gpu, block, &desc_buf, &out, &totals, i, &batches[i], prep)?;
            }
            gpu.synchronize()?;
        }
        if gpu.profiling() {
            eprintln!(
                "TEASAR_PROFILE: prepare: {} slabs in {} batches ({} rerun for node capacity), {} dispatches",
                slabs.len(),
                batches.len(),
                redo.len(),
                gpu.take_dispatch_count()
            );
            gpu.print_profile("prepare")?;
        }

        // SAFETY: the call's work has completed and nothing else uses `out`
        let all = unsafe { out.host::<u32>() };
        Ok(descs
            .par_iter()
            .zip(slabs.par_iter())
            .enumerate()
            .map(|(i, (d, g))| {
                let b = batches.iter().find(|b| b.slabs.contains(&i)).expect("every slab is in a batch");
                let at = b.out_at + d.out_base as usize;
                let n = (d.pz * d.py * d.px) as usize;
                Labels { shape: g.pooled_shape(ds), data: all[at..at + n].to_vec() }
            })
            .collect())
    }

    /// Slab descriptors, relative to their batch, and the batches
    fn plan(&self, slabs: &[SlabGeom], ds: u32) -> Result<(Vec<SlabDesc>, Vec<Batch>)> {
        let ds = ds as usize;
        let mut descs = Vec::with_capacity(slabs.len());
        let mut batches: Vec<Batch> = Vec::new();
        let mut out_at = 0;
        for (i, g) in slabs.iter().enumerate() {
            let c = g.shape.map(|s| s.div_ceil(ds));
            let p = g.shape.map(|s| s / ds);
            let cells: usize = c.iter().product();
            let outs: usize = p.iter().product();
            let buckets = g.voxels().div_ceil(BUCKET_VOXELS) as usize;
            ensure!(cells > 0 && outs > 0, "slab {g:?} is thin");
            ensure!(cells < u32::MAX as usize / 2, "slab {g:?} has too many cells for 32-bit indexing");
            let start_new = batches.last().is_none_or(|b| b.cells + cells > self.batch_cells);
            if start_new {
                batches.push(Batch { slabs: i..i, cells: 0, outs: 0, buckets: 0, out_at });
            }
            let b = batches.last_mut().expect("pushed above");
            let u = |v: usize| u32::try_from(v).expect("checked against the limits");
            descs.push(SlabDesc {
                z0: u(g.origin[0]),
                y0: u(g.origin[1]),
                x0: u(g.origin[2]),
                zs: u(g.shape[0]),
                ys: u(g.shape[1]),
                xs: u(g.shape[2]),
                cz: u(c[0]),
                cy: u(c[1]),
                cx: u(c[2]),
                pz: u(p[0]),
                py: u(p[1]),
                px: u(p[2]),
                cell_base: u(b.cells),
                out_base: u(b.outs),
                bucket_base: u(b.buckets),
                pad: 0,
            });
            b.slabs.end = i + 1;
            b.cells += cells;
            b.outs += outs;
            b.buckets += buckets;
            out_at += outs;
        }
        for b in &batches {
            ensure!(b.cells < u32::MAX as usize / 2, "a batch of {} cells overflows 32-bit indexing", b.cells);
        }
        Ok((descs, batches))
    }

    fn ensure_scratch(&mut self, gpu: &Gpu, cells: usize, nodes: usize, buckets: usize) -> Result<()> {
        if self.scratch.as_ref().is_some_and(|s| s.fits(cells, nodes, buckets)) {
            return Ok(());
        }
        let (c, n, b) = match &self.scratch {
            Some(s) => (s.cells.max(cells), s.nodes.max(nodes), s.buckets.max(buckets)),
            None => (cells, nodes, buckets),
        };
        // the old scratch may still be read by queued work, which Metal keeps alive until it completes
        self.scratch = Some(Scratch::new(gpu, c, n, b)?);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        gpu: &Gpu,
        block: &BlockArg<'_>,
        descs: &Buf,
        out: &Buf,
        totals: &Buf,
        index: usize,
        b: &Batch,
        prep: &PrepSettings,
    ) -> Result<()> {
        let sc = self.scratch.as_ref().expect("allocated before encoding");
        let ds = prep.downsample;
        let prm = PrepParams {
            can_xp: cell_bits(ds, |_, _, dx| dx != ds - 1),
            can_xm: cell_bits(ds, |_, _, dx| dx != 0),
            can_yp: cell_bits(ds, |_, dy, _| dy != ds - 1),
            can_ym: cell_bits(ds, |_, dy, _| dy != 0),
            face_xp: cell_bits(ds, |_, _, dx| dx == ds - 1),
            face_yp: cell_bits(ds, |_, dy, _| dy == ds - 1),
            face_zp: cell_bits(ds, |dz, _, _| dz == ds - 1),
            ds,
            ds2: ds * ds,
            nslabs: b.slabs.len() as u32,
            cells: b.cells as u32,
            outs: b.outs as u32,
            buckets: b.buckets as u32,
            cap: sc.nodes as u32,
            row_bytes: block.row_bytes as u32,
            block_rows: block.rows as u32,
            min_count: prep.min_count,
            max_count: prep.max_count,
            pad: 0,
        };
        let prm = Arg::Bytes(bytemuck::bytes_of(&prm));
        let blk = block.buf.at_byte(block.offset);
        let sl = descs.at::<SlabDesc>(b.slabs.start);
        let off = Arg::from(&sc.off);
        let parent = Arg::from(&sc.parent);
        let key = Arg::from(&sc.key);
        let bucket = Arg::from(&sc.bucket);
        let sorted = Arg::from(&sc.sorted);
        let outb = out.at::<u32>(b.out_at);
        let (cells, nodes) = (b.cells, sc.nodes);
        const G: usize = 256;

        gpu.dispatch_1d("prep_count", cells, G, &[blk, sl, prm, off])?;
        self.scan(gpu, &sc.off, cells, true, 0)?;
        gpu.dispatch_1d("prep_init", cells, G, &[blk, sl, prm, off, parent, key])?;
        gpu.dispatch_1d("prep_union", cells, G, &[blk, sl, prm, off, parent])?;
        gpu.dispatch_1d("prep_compress", nodes, G, &[prm, off, parent])?;
        gpu.dispatch_1d("prep_keymin", nodes, G, &[prm, off, parent, key])?;
        gpu.dispatch_1d("fill_u32", b.buckets, G, &[bucket, Arg::U32(0)])?;
        gpu.dispatch_1d("prep_bcount", nodes, G, &[sl, prm, off, parent, key, bucket])?;
        self.scan(gpu, &sc.bucket, b.buckets, false, 0)?;
        gpu.dispatch_1d("prep_scatter", nodes, G, &[sl, prm, off, parent, key, bucket, sorted])?;
        gpu.dispatch_1d("prep_rank", nodes, G, &[sl, prm, off, parent, key, bucket, sorted])?;
        gpu.dispatch_1d("prep_label", nodes, G, &[prm, off, parent, key])?;
        gpu.dispatch_1d("prep_pool", b.outs, G, &[sl, prm, off, key, parent, outb])?;
        gpu.dispatch_1d("prep_filter", b.outs, G, &[sl, prm, off, parent, outb])?;
        gpu.dispatch_1d("prep_total", 1, 1, &[prm, off, totals.into(), Arg::U32(index as u32)])?;
        Ok(())
    }

    /// Exclusive prefix sum of `n` uints of `data` in place; with `with_total`, `data[n]` gets the total
    fn scan(&self, gpu: &Gpu, data: &Buf, n: usize, with_total: bool, level: usize) -> Result<()> {
        let sc = self.scratch.as_ref().expect("allocated before encoding");
        let groups = n.div_ceil(SCAN_BLOCK);
        let flag = Arg::U32(u32::from(with_total));
        if groups <= 1 {
            return gpu.dispatch_groups(
                "scan_apply",
                1,
                SCAN_THREADS,
                &[data.into(), (&sc.zero).into(), Arg::U32(n as u32), flag],
            );
        }
        let sums = &sc.levels[level];
        gpu.dispatch_groups("scan_reduce", groups, SCAN_THREADS, &[data.into(), sums.into(), Arg::U32(n as u32)])?;
        self.scan(gpu, sums, groups, false, level + 1)?;
        gpu.dispatch_groups("scan_apply", groups, SCAN_THREADS, &[data.into(), sums.into(), Arg::U32(n as u32), flag])
    }
}
