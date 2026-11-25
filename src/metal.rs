//! The Metal backend: an objc2-metal host and MSL kernels compiled at load

pub mod gpu;
mod prep;
#[cfg(feature = "testing")]
mod selftest;
mod skel;

use crate::compat::CompatOrders;
use crate::geom::{Block, SlabGeom};
use crate::skel::Choices;
use crate::{
    Backend, CallStats, Error, Labels, LabelsRef, Packed, PrepSettings, Result, SettingsSet, SkelSettings, cpu,
};

use gpu::Gpu;
use prep::{BlockArg, Prep};
use skel::Skel;

/// The device name after the startup checks, without compiling kernels
pub fn probe() -> Result<String> {
    gpu::open_device().map(|d| objc2_metal::MTLDevice::name(&*d).to_string())
}

fn gpu_err(e: color_eyre::eyre::Report) -> Error {
    Error::Gpu(format!("{e:#}"))
}

/// The performance cores (`sysctl hw.perflevel0.physicalcpu`), else every core the OS reports
fn performance_cores() -> usize {
    let mut n: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>();
    // SAFETY: the name is NUL-terminated and `n` and `len` describe a writable int
    let ok = unsafe {
        libc::sysctlbyname(
            c"hw.perflevel0.physicalcpu".as_ptr(),
            (&raw mut n).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    } == 0;
    if ok && n > 0 {
        return n as usize;
    }
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

/// Blocks under this many bytes are copied into a device buffer instead of wrapped in place
const WRAP_MIN_BYTES: usize = 1 << 20;

pub struct MetalBackend {
    gpu: Gpu,
    prep: Prep,
    skel: Skel,
}

impl MetalBackend {
    /// Opens the device and starts the kernel build in the background; host pieces run on `TEASAR_CPU_THREADS`
    /// threads, read once here, by default the performance cores
    pub fn open() -> Result<Self> {
        let host_threads = match std::env::var("TEASAR_CPU_THREADS") {
            Ok(v) => v
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::Value(format!("TEASAR_CPU_THREADS must be a positive integer, got {v:?}")))?,
            Err(_) => performance_cores(),
        };
        let skel = Skel::new(host_threads)
            .map_err(|e| Error::Value(format!("could not start {host_threads} host threads: {e}")))?;
        Ok(Self { gpu: Gpu::open()?, prep: Prep::new(), skel })
    }

    fn run_prep(
        &mut self,
        buf: &gpu::Buf,
        offset: usize,
        block: &Block<'_>,
        slabs: &[SlabGeom],
        prep: &PrepSettings,
    ) -> Result<Vec<Labels>> {
        let arg = BlockArg { buf, offset, rows: block.y, row_bytes: block.row_bytes() };
        self.prep.run(&self.gpu, &arg, slabs, prep).map_err(gpu_err)
    }
}

impl Backend for MetalBackend {
    fn backend(&self) -> &'static str {
        "metal"
    }

    fn host_threads(&self) -> usize {
        self.skel.host_threads()
    }

    fn take_stats(&mut self) -> CallStats {
        CallStats {
            gpu_busy_s: self.gpu.take_busy_secs(),
            host_threads: self.host_threads(),
            ..std::mem::take(&mut self.skel.stats)
        }
    }

    fn name(&self) -> String {
        self.gpu.name()
    }

    fn memory(&self) -> (u64, u64) {
        self.gpu.memory()
    }

    fn prepare_labels(&mut self, block: &Block<'_>, slabs: &[SlabGeom], prep: &PrepSettings) -> Result<Vec<Labels>> {
        // a cell over 64 voxels doesn't fit the kernels' bit masks; the named settings pool by 2 or 4
        if prep.downsample > 4 {
            return Err(Error::Value(format!("the Metal backend pools by at most 4, got {}", prep.downsample)));
        }
        if block.data.len() >= WRAP_MIN_BYTES {
            // SAFETY: the caller keeps `block` alive and unmodified for the call, and `run_prep` waits for the GPU
            // before it returns, so no command reads the wrapped pages after this
            if let Some((buf, offset)) = unsafe { self.gpu.wrap(block.data) } {
                return self.run_prep(&buf, offset, block, slabs, prep);
            }
        }
        let buf = self.gpu.upload(block.data).map_err(gpu_err)?;
        self.run_prep(&buf, 0, block, slabs, prep)
    }

    fn prepare_dense(&mut self, binary: &[u8], shape: [usize; 3], prep: &PrepSettings) -> Result<Labels> {
        let row_bytes = shape[2].div_ceil(8);
        let bytes = shape[0] * shape[1] * row_bytes;
        let buf = self.gpu.alloc(bytes).map_err(gpu_err)?;
        // SAFETY: a fresh buffer no command uses yet
        let dst = unsafe { &mut buf.host_mut::<u8>()[..bytes] };
        crate::geom::pack_bits_into(binary, shape, dst);
        let block = Block::new(dst, [shape[0], shape[1], row_bytes], shape[2])?;
        let slab = SlabGeom::whole(&block);
        if prep.downsample > 4 {
            return Ok(cpu::prep::prepare_slab(&block, &slab, prep));
        }
        let mut out = self.run_prep(&buf, 0, &block, &[slab], prep)?;
        Ok(out.pop().expect("one slab in, one label array out"))
    }

    fn skeletonize(&mut self, labels: &[LabelsRef<'_>], set: SettingsSet, skel: &SkelSettings) -> Result<Vec<Packed>> {
        let choices = Choices::current()?;
        // the size filter keeps pieces under max_count, so by default nothing falls back
        let large = set.prep().max_count;
        let out = self.skel.run(&self.gpu, labels, skel, &choices, large).map_err(gpu_err)?;
        if self.gpu.profiling() {
            self.skel.profile.print(self.skel.host_threads());
            self.skel.profile = skel::SkelProfile::default();
            self.gpu.print_profile("skeletonize").map_err(gpu_err)?;
        }
        Ok(out)
    }

    fn skeletonize_compat(
        &mut self,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        orders: &dyn CompatOrders,
    ) -> Result<Vec<Packed>> {
        // the reference's orders come from Python per piece, so the whole skeleton stage runs on the host pool
        let fix = |shape| crate::effective_fix_borders(skel.fix_borders, shape);
        let (out, c) = self.skel.host_pool().install(|| cpu::km::skeletonize(labels, skel, fix, orders))?;
        self.skel.stats.add_cpu(c.pieces, c.voxels, c.busy_ns as f64 * 1e-9);
        Ok(out)
    }

    #[cfg(feature = "testing")]
    fn set_batch_cells(&mut self, cells: usize) {
        self.prep.batch_cells = cells.max(1);
    }

    #[cfg(feature = "testing")]
    fn inject_gpu_failure(&mut self) {
        self.gpu.inject_failure();
    }

    #[cfg(feature = "testing")]
    fn oob_probe(&mut self) -> Result<()> {
        let run = || -> color_eyre::eyre::Result<()> {
            let buf = self.gpu.alloc(64)?;
            self.gpu.dispatch_1d("oob_probe", 16, 16, &[(&buf).into(), gpu::Arg::U32(16)])?;
            self.gpu.synchronize()
        };
        run().map_err(gpu_err)
    }

    #[cfg(feature = "testing")]
    fn selftest(&mut self, name: &str) -> Result<String> {
        selftest::run(&self.gpu, name).map_err(gpu_err)
    }

    #[cfg(feature = "testing")]
    fn fields(&mut self, labels: &[LabelsRef<'_>], skel: &SkelSettings) -> Result<Vec<crate::skel::FieldDump>> {
        let choices = Choices::current()?;
        self.skel.fields(&self.gpu, labels, skel, &choices).map_err(gpu_err)
    }

    #[cfg(feature = "testing")]
    fn bench_copy(&mut self, bytes: usize, iters: usize) -> Result<f64> {
        let n = bytes / 16;
        let run = || -> color_eyre::eyre::Result<f64> {
            let (src, dst) = (self.gpu.upload(&vec![1u8; n * 16])?, self.gpu.alloc(n * 16)?);
            let copy = |a: &gpu::Buf, b: &gpu::Buf| self.gpu.dispatch_1d("copy_u4", n, 256, &[a.into(), b.into()]);
            copy(&src, &dst)?;
            self.gpu.synchronize()?;
            let t = std::time::Instant::now();
            for i in 0..iters {
                if i % 2 == 0 { copy(&src, &dst)? } else { copy(&dst, &src)? }
            }
            self.gpu.synchronize()?;
            Ok((2 * n * 16 * iters) as f64 / t.elapsed().as_secs_f64() / 1e9)
        };
        run().map_err(gpu_err)
    }
}
