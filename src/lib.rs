//! teasar: TEASAR skeletonisation of villa's surface-track slabs, with a Metal backend for Apple GPUs and a pure-Rust
//! CPU reference behind the same API
//!
//! The front end ([`Device`]) validates settings, limits and arrays, then hands validated work to a [`Backend`]:
//! [`cpu::CpuBackend`] always, and `metal::MetalBackend` on macOS with the `metal` feature. Every backend returns the
//! same [`Labels`] and [`Packed`] types, and the part A stages give identical results on every backend

pub mod compat;
pub mod cpu;
pub mod error;
pub mod geom;
pub mod limits;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod packed;
#[cfg(feature = "python")]
mod python;
pub mod settings;
pub mod skel;
pub mod stats;

pub use compat::{CompatOrders, TargetOrder};
pub use error::{Error, Result};
pub use geom::{Block, SlabGeom, SlabKind, SlabSpec};
pub use limits::{Limit, Limits};
pub use packed::Packed;
pub use settings::{Field, Method, PrepSettings, SettingsSet, SkelSettings, Value};
pub use stats::CallStats;

/// A slab's pooled labels in C order
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Labels {
    pub shape: [usize; 3],
    pub data: Vec<u32>,
}

/// A borrowed label array, as `skeletonize` takes them
#[derive(Debug, Clone, Copy)]
pub struct LabelsRef<'a> {
    pub shape: [usize; 3],
    pub data: &'a [u32],
}

/// One implementation of the three operations; the front end validates everything before it calls one
pub trait Backend: Send {
    /// "metal" or "cpu"
    fn backend(&self) -> &'static str;

    /// The device's name, such as "Apple M4 Pro"
    fn name(&self) -> String;

    /// The CPU threads the device computes on: the CPU reference's pool, or the GPU backend's host-piece pool
    fn host_threads(&self) -> usize;

    /// The statistics gathered since the last take, without the wall time, which the device measures
    fn take_stats(&mut self) -> CallStats;

    /// Bytes this process has allocated on the device, and the device's recommended working set
    fn memory(&self) -> (u64, u64);

    /// The pooled labels of each slab of `block`; no slab is thin
    fn prepare_labels(&mut self, block: &Block<'_>, slabs: &[SlabGeom], prep: &PrepSettings) -> Result<Vec<Labels>>;

    /// The pooled labels of one dense (z, y, x) array of 0/nonzero bytes; the default packs it into a block
    fn prepare_dense(&mut self, binary: &[u8], shape: [usize; 3], prep: &PrepSettings) -> Result<Labels> {
        let packed = geom::pack_bits(binary, shape);
        let block = Block::new(&packed, [shape[0], shape[1], shape[2].div_ceil(8)], shape[2])?;
        let mut out = self.prepare_labels(&block, &[SlabGeom::whole(&block)], prep)?;
        Ok(out.pop().expect("one slab in, one label array out"))
    }

    /// The skeletons of each label array
    fn skeletonize(&mut self, labels: &[LabelsRef<'_>], set: SettingsSet, skel: &SkelSettings) -> Result<Vec<Packed>>;

    /// The skeletons in compatibility mode (addendum KM), on this backend's host threads
    fn skeletonize_compat(
        &mut self,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        orders: &dyn CompatOrders,
    ) -> Result<Vec<Packed>>;

    /// `prepare_labels` then `skeletonize`, for slabs none of which is thin
    fn extract(
        &mut self,
        block: &Block<'_>,
        slabs: &[SlabGeom],
        set: SettingsSet,
        skel: &SkelSettings,
    ) -> Result<Vec<Packed>> {
        let labels = self.prepare_labels(block, slabs, &set.prep())?;
        let refs: Vec<LabelsRef<'_>> = labels.iter().map(|l| LabelsRef { shape: l.shape, data: &l.data }).collect();
        self.skeletonize(&refs, set, skel)
    }

    /// Caps the pooled cells of one internal batch, so tests can show batching changes nothing
    #[cfg(feature = "testing")]
    fn set_batch_cells(&mut self, _cells: usize) {}

    /// Makes the next GPU submission report failure
    #[cfg(feature = "testing")]
    fn inject_gpu_failure(&mut self) {}

    /// Runs a kernel that writes out of bounds, as a negative control for the validation layers
    #[cfg(feature = "testing")]
    fn oob_probe(&mut self) -> Result<()> {
        Err(Error::NotImplemented("no GPU on this backend".into()))
    }

    /// Runs the GPU self-test `name` (`atomic64`, `float_rules`), returning a one-line report; fails on a mismatch
    #[cfg(feature = "testing")]
    fn selftest(&mut self, _name: &str) -> Result<String> {
        Err(Error::NotImplemented("no GPU on this backend".into()))
    }

    /// B7's field-level dump of each label array (spec part B §4.7), with every piece traced on this backend's own
    /// path (no host pieces on a GPU backend)
    #[cfg(feature = "testing")]
    fn fields(&mut self, _labels: &[LabelsRef<'_>], _skel: &SkelSettings) -> Result<Vec<skel::FieldDump>> {
        Err(Error::NotImplemented("no field dump on this backend".into()))
    }

    /// Copy-kernel bandwidth in GB/s (bytes read plus written), over `iters` copies of `bytes`
    #[cfg(feature = "testing")]
    fn bench_copy(&mut self, _bytes: usize, _iters: usize) -> Result<f64> {
        Err(Error::NotImplemented("no copy benchmark on this backend".into()))
    }
}

/// villa's rule: border targets are off for a slab whose pooled shape has two or more axes of length 1
pub fn effective_fix_borders(fix_borders: bool, pooled_shape: [usize; 3]) -> bool {
    fix_borders && pooled_shape.iter().filter(|&&s| s == 1).count() < 2
}

/// A device: settings and limit checks in front of one backend
pub struct Device {
    backend: Box<dyn Backend>,
    limits: Limits,
    last: CallStats,
}

impl Device {
    /// The CPU reference on `threads` threads, 0 meaning all cores
    pub fn cpu(threads: usize) -> Result<Self> {
        Ok(Self::with(Box::new(cpu::CpuBackend::new(threads)?)))
    }

    /// The platform's GPU backend; its host pieces run on `TEASAR_CPU_THREADS` threads (spec part B §3.1)
    pub fn gpu() -> Result<Self> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            Ok(Self::with(Box::new(metal::MetalBackend::open()?)))
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            Err(Error::BackendUnavailable("this build has no GPU backend for this platform (Metal needs macOS)".into()))
        }
    }

    /// Whether the GPU backend can open, with the device name or the reason it can't; never fails
    pub fn check_available() -> (bool, String) {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match metal::probe() {
                Ok(name) => (true, name),
                Err(e) => (false, e.to_string()),
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            (false, "this build has no GPU backend for this platform (Metal needs macOS)".into())
        }
    }

    pub fn backend(&self) -> &'static str {
        self.backend.backend()
    }

    pub fn name(&self) -> String {
        self.backend.name()
    }

    fn with(backend: Box<dyn Backend>) -> Self {
        Self { backend, limits: Limits::default(), last: CallStats::default() }
    }

    /// The statistics of the most recent call (spec part A §1.4), all zero before the first
    pub fn last_stats(&self) -> CallStats {
        self.last
    }

    /// Runs one public call and records its statistics, whether or not it succeeds
    fn timed<T>(&mut self, call: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        self.backend.take_stats();
        let t = std::time::Instant::now();
        let out = call(self);
        self.last = CallStats { wall_s: t.elapsed().as_secs_f64(), ..self.backend.take_stats() };
        out
    }

    pub fn memory(&self) -> (u64, u64) {
        self.backend.memory()
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Replaces the limits, for tests that check each one without huge inputs
    pub fn set_limits(&mut self, limits: Limits) {
        self.limits = limits;
    }

    #[cfg(feature = "testing")]
    pub fn backend_mut(&mut self) -> &mut dyn Backend {
        self.backend.as_mut()
    }

    /// The pre-skeleton stages for one dense slab (spec §2.1)
    pub fn prepare_labels(&mut self, binary: &[u8], shape: [usize; 3], set: SettingsSet) -> Result<Labels> {
        self.timed(|d| d.prepare_labels_call(binary, shape, set))
    }

    /// The part A stages of `extract`: one label array per slab, `None` for a thin slab (spec §2.3)
    pub fn prepare_from_block(
        &mut self,
        block: &Block<'_>,
        slabs: &[SlabSpec],
        set: SettingsSet,
    ) -> Result<Vec<Option<Labels>>> {
        self.timed(|d| d.prepare_from_block_call(block, slabs, set))
    }

    /// The skeletons of each label array (spec §2.2), tracing targets in `order`
    pub fn skeletonize(
        &mut self,
        labels: &[LabelsRef<'_>],
        set: SettingsSet,
        fix_borders: bool,
        order: TargetOrder<'_>,
    ) -> Result<Vec<Packed>> {
        self.timed(|d| d.skeletonize_call(labels, set, fix_borders, order))
    }

    /// The fused path: pre-skeleton stages then skeletons, for each slab of `block` (spec §2.3)
    pub fn extract(
        &mut self,
        block: &Block<'_>,
        slabs: &[SlabSpec],
        set: SettingsSet,
        fix_borders: bool,
        order: TargetOrder<'_>,
    ) -> Result<Vec<Packed>> {
        self.timed(|d| d.extract_call(block, slabs, set, fix_borders, order))
    }

    fn prepare_labels_call(&mut self, binary: &[u8], shape: [usize; 3], set: SettingsSet) -> Result<Labels> {
        let prep = set.prep();
        ensure_value!(binary.len() == shape.iter().product::<usize>(), "binary does not match its shape {shape:?}");
        let geom = SlabGeom { origin: [0; 3], shape };
        ensure_value!(
            !geom.is_thin(prep.downsample),
            "binary of shape {shape:?} has an axis shorter than downsample {}",
            prep.downsample
        );
        self.check_slab(&geom, prep.downsample, || format!("binary of shape {shape:?}"))?;
        // the output's size is known before any work here, so the output check runs first
        self.limits.check_output([geom.pooled_voxels(prep.downsample) * 4])?;
        self.backend.prepare_dense(binary, shape, &prep)
    }

    fn prepare_from_block_call(
        &mut self,
        block: &Block<'_>,
        slabs: &[SlabSpec],
        set: SettingsSet,
    ) -> Result<Vec<Option<Labels>>> {
        let prep = set.prep();
        let ds = prep.downsample;
        let geoms = self.slab_geoms(Method::PrepareFromBlock, block, slabs, set)?;
        self.check_block(block, &geoms, ds)?;
        self.limits.check_output(geoms.iter().map(|g| if g.is_thin(ds) { 0 } else { g.pooled_voxels(ds) * 4 }))?;

        let work: Vec<SlabGeom> = geoms.iter().filter(|g| !g.is_thin(ds)).copied().collect();
        let mut done = if work.is_empty() { Vec::new() } else { self.backend.prepare_labels(block, &work, &prep)? };
        ensure_gpu(done.len() == work.len(), "the backend returned the wrong number of label arrays")?;
        done.reverse();
        Ok(geoms.iter().map(|g| if g.is_thin(ds) { None } else { done.pop() }).collect())
    }

    fn skeletonize_call(
        &mut self,
        labels: &[LabelsRef<'_>],
        set: SettingsSet,
        fix_borders: bool,
        order: TargetOrder<'_>,
    ) -> Result<Vec<Packed>> {
        for (i, l) in labels.iter().enumerate() {
            ensure_value!(l.data.len() == l.shape.iter().product::<usize>(), "labels do not match their shape");
            self.check_pooled(l.shape, || format!("label array {i} of shape {:?}", l.shape))?;
        }
        let total: u64 = labels.iter().map(|l| l.data.len() as u64).sum();
        self.check_call_pooled(total)?;

        let skel = set.skel_settings(fix_borders);
        let out = match order {
            TargetOrder::Best => self.backend.skeletonize(labels, set, &skel)?,
            TargetOrder::Kimimaro(orders) => self.backend.skeletonize_compat(labels, &skel, orders)?,
        };
        ensure_gpu(out.len() == labels.len(), "the backend returned the wrong number of skeleton sets")?;
        self.limits.check_output(out.iter().map(Packed::nbytes))?;
        Ok(out)
    }

    fn extract_call(
        &mut self,
        block: &Block<'_>,
        slabs: &[SlabSpec],
        set: SettingsSet,
        fix_borders: bool,
        order: TargetOrder<'_>,
    ) -> Result<Vec<Packed>> {
        let ds = set.downsample();
        let geoms = self.slab_geoms(Method::Extract, block, slabs, set)?;
        self.check_block(block, &geoms, ds)?;

        let work: Vec<SlabGeom> = geoms.iter().filter(|g| !g.is_thin(ds)).copied().collect();
        let skel = set.skel_settings(fix_borders);
        let mut done = match order {
            _ if work.is_empty() => Vec::new(),
            TargetOrder::Best => self.backend.extract(block, &work, set, &skel)?,
            // compatibility mode: the prepare stages, then the host skeletoniser
            TargetOrder::Kimimaro(orders) => {
                let labels = self.backend.prepare_labels(block, &work, &set.prep())?;
                let refs: Vec<LabelsRef<'_>> =
                    labels.iter().map(|l| LabelsRef { shape: l.shape, data: &l.data }).collect();
                self.backend.skeletonize_compat(&refs, &skel, orders).map_err(|e| match e {
                    // the error names a slab of the call's list, which also holds the thin slabs
                    Error::SomaUnsupported { slab, message } => {
                        let slab = (0..geoms.len()).filter(|&i| !geoms[i].is_thin(ds)).nth(slab).unwrap_or(slab);
                        Error::SomaUnsupported { slab, message }
                    }
                    e => e,
                })?
            }
        };
        ensure_gpu(done.len() == work.len(), "the backend returned the wrong number of skeleton sets")?;
        done.reverse();
        let out: Vec<Packed> = geoms
            .iter()
            .map(|g| if g.is_thin(ds) { Packed::empty() } else { done.pop().expect("counted above") })
            .collect();
        self.limits.check_output(out.iter().map(Packed::nbytes))?;
        Ok(out)
    }

    /// The boxes of `slabs` once each is in range, and fits `set`'s slab geometry where `method`'s row of the
    /// accepted-settings table asks for it
    fn slab_geoms(
        &self,
        method: Method,
        block: &Block<'_>,
        slabs: &[SlabSpec],
        set: SettingsSet,
    ) -> Result<Vec<SlabGeom>> {
        let geoms = slabs.iter().map(|s| s.geom(block)).collect::<Result<Vec<_>>>()?;
        if method.accepted().slab_thickness {
            slabs.iter().try_for_each(|s| set.check_slab(block, s))?;
        }
        Ok(geoms)
    }

    fn check_block(&self, block: &Block<'_>, slabs: &[SlabGeom], ds: u32) -> Result<()> {
        let bytes = block.data.len() as u64;
        self.limits.check(Limit::BlockBytes, bytes, || format!("the block has {bytes} bytes"))?;
        let mut pooled = 0;
        for (i, g) in slabs.iter().enumerate() {
            self.check_slab(g, ds, || format!("slab {i} of shape {:?}", g.shape))?;
            if !g.is_thin(ds) {
                pooled += g.pooled_voxels(ds);
            }
        }
        self.check_call_pooled(pooled)
    }

    /// The per-slab input checks; `what` names the slab, as in "slab 3 of shape [2, 1536, 1536]"
    fn check_slab(&self, g: &SlabGeom, ds: u32, what: impl Fn() -> String) -> Result<()> {
        let voxels = g.voxels();
        self.limits.check(Limit::SlabVoxels, voxels, || format!("{} has {voxels} voxels", what()))?;
        self.check_pooled(g.pooled_shape(ds), what)
    }

    /// The per-slab checks on a pooled shape: its longest axis, then its voxels
    fn check_pooled(&self, pooled: [usize; 3], what: impl Fn() -> String) -> Result<()> {
        let longest = pooled.iter().copied().max().unwrap_or(0) as u64;
        self.limits
            .check(Limit::AxisPooled, longest, || format!("{} has a pooled axis of {longest} voxels", what()))?;
        let voxels: u64 = pooled.iter().map(|&s| s as u64).product();
        self.limits.check(Limit::SlabPooled, voxels, || format!("{} has {voxels} pooled voxels", what()))
    }

    fn check_call_pooled(&self, pooled: u64) -> Result<()> {
        self.limits.check(Limit::CallPooled, pooled, || format!("the call's slabs pool to {pooled} voxels"))
    }
}

fn ensure_gpu(cond: bool, msg: &str) -> Result<()> {
    if cond { Ok(()) } else { Err(Error::Gpu(msg.into())) }
}
