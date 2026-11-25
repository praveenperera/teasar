//! Thin Metal layer: one device and command queue, shared-memory buffers, the MSL kernel library compiled at load,
//! and command batching
//!
//! Every dispatch goes into one serial compute encoder, so kernels run in the order they were encoded and each sees
//! the writes of the ones before it. The open command buffer is committed every [`FLUSH_EVERY`] dispatches so the GPU
//! starts while the host is still encoding, and [`Gpu::synchronize`] commits and waits for everything, failing with
//! the first failed buffer's error.
//!
//! Adapted from inkrs's Metal host (`src/metal/gpu.rs` at `35f35a1`, MIT OR Apache-2.0, Praveen Perera): the
//! reflection-checked dispatch, the batching and the encoder-boundary profiler are inkrs's; teasar's Metal backend drops its
//! precision switch and fences and adds wrapping host memory without a copy and a test hook that fails a submission

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::thread::JoinHandle;

use bytemuck::Pod;
use color_eyre::eyre::{Result, bail, ensure, eyre};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo, NSRange, NSString};
use objc2_metal::{
    MTLBinding, MTLBindingType, MTLBuffer, MTLBufferBinding, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCommonCounterSetTimestamp, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePassDescriptor, MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLGPUFamily,
    MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLPipelineOption, MTLResourceOptions, MTLSize, MTLStorageMode,
};

use crate::Error;

/// The shared header, prepended to every library
const COMMON: &str = include_str!("../../shaders/metal/common.metal");

/// The kernel libraries, each compiled on its own after [`COMMON`], so a compile failure in one (part B's
/// skeletoniser) leaves the others usable and surfaces only when one of its kernels is dispatched
const LIBRARIES: [(&str, &[&str]); 2] = [
    ("part A", &[include_str!("../../shaders/metal/prep.metal")]),
    ("part B", &[include_str!("../../shaders/metal/skel.metal")]),
];

/// Test builds add a kernel that writes past its buffer, so a validation run can show it catches one
const PROBE_SOURCE: &str = "
kernel void oob_probe(device uint* p_data [[buffer(0)]], constant uint& n [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    p_data[n + 4096 + i] = i;
}
";

/// Dispatches encoded before the open command buffer is committed
const FLUSH_EVERY: usize = 48;

/// A device buffer in shared storage, with its length in bytes
pub struct Buf {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
}

// SAFETY: Metal buffers are thread-safe objects; touching their contents while a kernel uses them is not, which the
// backend orders through command-buffer completion
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The buffer from byte `offset` on
    pub fn at_byte(&self, offset: usize) -> Arg<'_> {
        Arg::Buf(self, offset)
    }

    /// The buffer from element `offset` of `T` on
    pub fn at<T>(&self, offset: usize) -> Arg<'_> {
        Arg::Buf(self, offset * std::mem::size_of::<T>())
    }

    /// The contents as a host slice
    ///
    /// # Safety
    /// No command buffer that writes this buffer may be pending, and `T` must be valid for every bit pattern
    pub unsafe fn host<T: Pod>(&self) -> &[T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: shared storage of at least `bytes` bytes, page-aligned, and the caller rules out GPU writes
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast::<T>(), n) }
    }

    /// The contents as a mutable host slice
    ///
    /// # Safety
    /// No queued kernel may read or write this buffer
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn host_mut<T: Pod>(&self) -> &mut [T] {
        let n = self.bytes / std::mem::size_of::<T>();
        // SAFETY: as `host`, and the caller rules out any GPU access
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast::<T>(), n) }
    }
}

/// A kernel argument: a buffer from a byte offset, a small struct copied at encode time, or a 4-byte scalar
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    Buf(&'a Buf, usize),
    Bytes(&'a [u8]),
    U32(u32),
}

impl From<u32> for Arg<'_> {
    fn from(v: u32) -> Self {
        Arg::U32(v)
    }
}

impl<'a> From<&'a Buf> for Arg<'a> {
    fn from(b: &'a Buf) -> Self {
        Arg::Buf(b, 0)
    }
}

/// A kernel parameter as reflection reports it
#[derive(Clone, Copy, Debug)]
enum Param {
    Pointer,
    Scalar(MTLDataType),
}

/// A compiled kernel and its parameter list
struct Kernel {
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    params: Vec<Param>,
}

type Kernels = HashMap<String, Kernel>;

/// The compiled kernels, and for each library that failed to compile, its error
struct Built {
    kernels: Kernels,
    failed: Vec<(&'static str, String)>,
}

/// Compiles `source` as one library with the backend's options
pub fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let options = MTLCompileOptions::new();
    // safe math turns off unsafe float optimisations; part B's float code also guards its own rounding (see
    // skel.metal), so it doesn't depend on what this leaves on
    options.setLanguageVersion(MTLLanguageVersion::Version4_0);
    options.setMathMode(MTLMathMode::Safe);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
        .map_err(|e| eyre!("{}", e.localizedDescription()))
}

/// A 64-bit atomic minimum on device memory, which spec part B §3.2 lists from Apple9 and leaves unverified on this
/// Mac; compiled as its own library so a failure stays local to the probe
pub const ATOMIC64_SOURCE: &str = "
#include <metal_stdlib>
using namespace metal;
kernel void a64_probe(device atomic_ulong* p_out [[buffer(0)]], device const ulong* p_in [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    atomic_min_explicit(&p_out[i % 4], p_in[i], memory_order_relaxed);
}
";

/// Compiles every kernel library without building pipelines, returning each library's name and compile error, for a
/// check that needs no GPU time
pub fn check_libraries() -> std::result::Result<Vec<(&'static str, Option<String>)>, Error> {
    let device = open_device()?;
    Ok(LIBRARIES
        .iter()
        .map(|(name, parts)| {
            let mut source = String::from(COMMON);
            parts.iter().for_each(|p| source.push_str(p));
            (*name, compile_library(&device, &source).err().map(|e| format!("{e:#}")))
        })
        .chain(std::iter::once((
            "atomic64 probe",
            compile_library(&device, ATOMIC64_SOURCE).err().map(|e| format!("{e:#}")),
        )))
        .collect())
}

/// Compiles every library and builds a pipeline for each kernel
fn build_kernels(device: &ProtocolObject<dyn MTLDevice>) -> Result<Built> {
    let mut built = Built { kernels: Kernels::new(), failed: Vec::new() };
    for (i, (name, parts)) in LIBRARIES.iter().enumerate() {
        let mut source = String::from(COMMON);
        parts.iter().for_each(|p| source.push_str(p));
        if i == 0 && cfg!(feature = "testing") {
            source.push_str(PROBE_SOURCE);
        }
        match compile_library(device, &source) {
            Ok(library) => built.kernels.extend(build_pipelines(device, &library)?),
            Err(e) => built.failed.push((name, format!("{e:#}"))),
        }
    }
    if cfg!(feature = "testing") {
        match compile_library(device, ATOMIC64_SOURCE) {
            Ok(library) => built.kernels.extend(build_pipelines(device, &library)?),
            Err(e) => built.failed.push(("atomic64 probe", format!("{e:#}"))),
        }
    }
    // the part A library is required; part B's failure is reported when its kernels are used
    if let Some((name, why)) = built.failed.iter().find(|(n, _)| *n == "part A") {
        bail!("compile the Metal kernels ({name}): {why}");
    }

    Ok(built)
}

/// A pipeline for every kernel of `library`, spread over a few threads
fn build_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Result<Kernels> {
    let names: Vec<String> = library.functionNames().iter().map(|n| n.to_string()).collect();
    let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).min(8);
    std::thread::scope(|s| {
        let workers: Vec<_> = names
            .chunks(names.len().div_ceil(threads).max(1))
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .map(|name| Ok((name.clone(), build_kernel(device, library, name)?)))
                        .collect::<Result<Vec<_>>>()
                })
            })
            .collect();
        let mut kernels = Kernels::new();
        for w in workers {
            kernels.extend(w.join().map_err(|_| eyre!("a Metal pipeline build panicked"))??);
        }

        Ok(kernels)
    })
}

/// The pipeline of kernel `name` and its parameter list, read from reflection
fn build_kernel(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Kernel> {
    let func =
        library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| eyre!("no Metal kernel named {name}"))?;
    let mut reflection = None;
    // SAFETY: the function comes from our library and is only dispatched through `Gpu::dispatch`, which checks its
    // arguments against this reflection
    let pso = unsafe {
        device.newComputePipelineStateWithFunction_options_reflection_error(
            &func,
            MTLPipelineOption::BindingInfo | MTLPipelineOption::BufferTypeInfo,
            Some(&mut reflection),
        )
    }
    .map_err(|e| eyre!("build Metal pipeline {name}: {}", e.localizedDescription()))?;
    let reflection = reflection.ok_or_else(|| eyre!("no reflection for Metal kernel {name}"))?;
    let mut bindings: Vec<(usize, Param)> = Vec::new();
    for b in reflection.bindings().iter() {
        if b.r#type() != MTLBindingType::Buffer {
            continue;
        }
        // SAFETY: a binding of type Buffer is an MTLBufferBinding
        let buffer: &ProtocolObject<dyn MTLBufferBinding> =
            unsafe { &*(&*b as *const ProtocolObject<dyn MTLBinding>).cast() };
        let ty = buffer.bufferDataType();
        // pointers and structs are named p_*; everything else must be a 4-byte scalar
        let param = if b.name().to_string().starts_with("p_") {
            Param::Pointer
        } else {
            ensure!(
                buffer.bufferDataSize() == 4 && matches!(ty, MTLDataType::UInt),
                "Metal kernel {name}: parameter {} is neither a p_* pointer nor a uint",
                b.name()
            );
            Param::Scalar(ty)
        };
        bindings.push((b.index(), param));
    }
    bindings.sort_by_key(|(i, _)| *i);
    for (pos, (i, _)) in bindings.iter().enumerate() {
        ensure!(*i == pos, "Metal kernel {name}: buffer indices must run 0..n without gaps, found {i} at {pos}");
    }

    Ok(Kernel { pso, params: bindings.into_iter().map(|(_, p)| p).collect() })
}

/// The open command buffer and its encoder
struct Batch {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    dispatches: usize,
}

/// Timestamp samples a profiling window holds, the most one counter sample buffer may take
const PROFILE_SAMPLES: usize = 4096;

/// GPU timestamps around every dispatch, for `TEASAR_PROFILE=1`
///
/// Each dispatch gets its own encoder, sampled at its start and end, since Apple GPUs sample only at encoder
/// boundaries. A dispatch is charged the time it adds to the GPU timeline, `end - max(start, previous end)`, so the
/// times sum to the busy time
struct Profiler {
    samples: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    pending: Vec<&'static str>,
    done: Vec<(&'static str, f64)>,
    origin: (u64, u64),
    last_end: u64,
}

impl Profiler {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        ensure!(
            device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            "TEASAR_PROFILE: this GPU can't sample timestamps at encoder boundaries"
        );
        // SAFETY: a static NSString constant of the Metal framework
        let want = unsafe { MTLCommonCounterSetTimestamp };
        let set = device
            .counterSets()
            .and_then(|sets| sets.iter().find(|s| MTLCounterSet::name(&**s).isEqualToString(want)))
            .ok_or_else(|| eyre!("TEASAR_PROFILE: no timestamp counter set"))?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: within the 32 KB a sample buffer may hold
        unsafe { desc.setSampleCount(PROFILE_SAMPLES) };
        let samples = device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| eyre!("TEASAR_PROFILE: {}", e.localizedDescription()))?;
        Ok(Self { samples, pending: Vec::new(), done: Vec::new(), origin: timestamps(device), last_end: 0 })
    }

    fn full(&self) -> bool {
        2 * self.pending.len() >= PROFILE_SAMPLES
    }

    fn sample(&mut self, name: &'static str) -> Retained<MTLComputePassDescriptor> {
        let i = 2 * self.pending.len();
        self.pending.push(name);
        let desc = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: index 0 exists on every pass descriptor, and both sample indices are inside the buffer, since
        // `full` synchronises before it runs out
        unsafe {
            let att = desc.sampleBufferAttachments().objectAtIndexedSubscript(0);
            att.setSampleBuffer(Some(&self.samples));
            att.setStartOfEncoderSampleIndex(i);
            att.setEndOfEncoderSampleIndex(i + 1);
        }
        desc
    }

    fn resolve(&mut self, device: &ProtocolObject<dyn MTLDevice>) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let n = 2 * self.pending.len();
        // SAFETY: the range lies inside the buffer and every command buffer that wrote it has completed
        let data = unsafe { self.samples.resolveCounterRange(NSRange::new(0, n)) }
            .ok_or_else(|| eyre!("TEASAR_PROFILE: could not resolve the timestamps"))?;
        let bytes = data.to_vec();
        ensure!(bytes.len() >= 8 * n, "TEASAR_PROFILE: {} bytes of timestamps for {n}", bytes.len());
        let ticks: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|c| u64::from_ne_bytes(*c)).collect();
        let now = timestamps(device);
        let ns_per_tick = (now.0 - self.origin.0) as f64 / (now.1 - self.origin.1).max(1) as f64;
        for (i, name) in self.pending.drain(..).enumerate() {
            let (a, b) = (ticks[2 * i], ticks[2 * i + 1]);
            // a sample the GPU couldn't take reads as all ones
            if a == u64::MAX || b == u64::MAX || b < a {
                continue;
            }
            let from = a.max(self.last_end);
            self.done.push((name, b.saturating_sub(from) as f64 * ns_per_tick * 1e-9));
            self.last_end = self.last_end.max(b);
        }
        Ok(())
    }
}

fn timestamps(device: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0u64, 0u64);
    // SAFETY: both pointers are to live u64s
    unsafe { device.sampleTimestamps_gpuTimestamp(NonNull::from(&mut cpu), NonNull::from(&mut gpu)) };
    (cpu, gpu)
}

fn failure_reason(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    cb.error().map_or_else(|| "unknown error".into(), |e| e.localizedDescription().to_string())
}

/// The system's Metal device after the startup checks: macOS 26 or newer, then the device, then GPU family Apple7
pub fn open_device() -> std::result::Result<Retained<ProtocolObject<dyn MTLDevice>>, Error> {
    // older systems lack Metal 4 and `setMathMode:`, which raises an Objective-C exception instead of failing
    let info = NSProcessInfo::processInfo();
    let v = info.operatingSystemVersion();
    let new_enough = info.isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
        majorVersion: 26,
        minorVersion: 0,
        patchVersion: 0,
    });
    if !new_enough {
        return Err(Error::BackendUnavailable(format!(
            "the Metal backend needs macOS 26 or newer, and this is macOS {}.{}.{}",
            v.majorVersion, v.minorVersion, v.patchVersion
        )));
    }
    let device = MTLCreateSystemDefaultDevice().ok_or_else(|| Error::BackendUnavailable("no Metal device".into()))?;
    if !device.supportsFamily(MTLGPUFamily::Apple7) {
        return Err(Error::BackendUnavailable(format!(
            "the Metal backend needs an Apple7-family GPU (M1 or newer), and this GPU is {}",
            device.name()
        )));
    }
    Ok(device)
}

/// The Metal device, its queue and the kernel library
pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    build: RefCell<Option<JoinHandle<Result<Built>>>>,
    kernels: OnceCell<Built>,
    batch: RefCell<Option<Batch>>,
    /// Committed command buffers not yet waited on
    inflight: RefCell<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    profile: Option<RefCell<Profiler>>,
    /// GPU start and end times of the command buffers waited on since the last take, in host seconds
    busy: RefCell<Vec<(f64, f64)>>,
    /// Test hook: the next synchronize reports a failed submission
    fail_next: Cell<bool>,
    dispatches: Cell<usize>,
}

// SAFETY: `Gpu` is not `Sync` (its `RefCell`s), so one thread at a time uses it, and Metal's command buffers and
// encoders may move between threads between uses
unsafe impl Send for Gpu {}

impl Gpu {
    /// Opens the device and starts compiling the kernel library on a background thread; the first dispatch waits
    pub fn open() -> std::result::Result<Self, Error> {
        let device = open_device()?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| Error::BackendUnavailable("Metal could not create a command queue".into()))?;
        let build = {
            let device = device.clone();
            std::thread::Builder::new()
                .name("teasar-metal-kernels".into())
                .spawn(move || build_kernels(&device))
                .map_err(|e| Error::BackendUnavailable(format!("could not start the kernel build thread: {e}")))?
        };
        let profile = match std::env::var("TEASAR_PROFILE").as_deref() {
            Ok("1") => Some(Profiler::new(&device).map_err(|e| Error::BackendUnavailable(e.to_string()))?),
            _ => None,
        };
        Ok(Self {
            device,
            queue,
            build: RefCell::new(Some(build)),
            kernels: OnceCell::new(),
            batch: RefCell::default(),
            inflight: RefCell::default(),
            profile: profile.map(RefCell::new),
            busy: RefCell::default(),
            fail_next: Cell::new(false),
            dispatches: Cell::new(0),
        })
    }

    pub fn profiling(&self) -> bool {
        self.profile.is_some()
    }

    /// The GPU's name, such as "Apple M4 Pro"
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// This process's Metal allocations on the device and the device's recommended working set, in bytes
    pub fn memory(&self) -> (u64, u64) {
        (self.device.currentAllocatedSize() as u64, self.device.recommendedMaxWorkingSetSize())
    }

    /// Waits for the kernel build started by [`Gpu::open`], surfacing a compile error
    pub fn ready(&self) -> Result<()> {
        self.kernels().map(|_| ())
    }

    fn kernels(&self) -> Result<&Built> {
        if let Some(k) = self.kernels.get() {
            return Ok(k);
        }
        let build = self.build.borrow_mut().take().ok_or_else(|| eyre!("the Metal kernel build failed earlier"))?;
        let kernels = build.join().map_err(|_| eyre!("the Metal kernel build panicked"))??;
        Ok(self.kernels.get_or_init(|| kernels))
    }

    fn kernel(&self, name: &str) -> Result<&Kernel> {
        let built = self.kernels()?;
        if let Some(k) = built.kernels.get(name) {
            return Ok(k);
        }
        if built.failed.is_empty() {
            bail!("no Metal kernel named {name}");
        }
        let why: Vec<String> =
            built.failed.iter().map(|(lib, e)| format!("the {lib} library failed to compile: {e}")).collect();
        bail!("no Metal kernel named {name}; {}", why.join("; "))
    }

    /// The device, for test probes that compile their own libraries
    #[cfg(feature = "testing")]
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// An uninitialised buffer of `bytes` (at least 4); kernels must write before they read
    pub fn alloc(&self, bytes: usize) -> Result<Buf> {
        let len = bytes.max(4);
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| eyre!("Metal could not allocate {len} bytes"))?;
        Ok(Buf { raw, bytes })
    }

    /// A buffer holding `data`
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buf> {
        let bytes = std::mem::size_of_val(data);
        let buf = self.alloc(bytes)?;
        // SAFETY: the buffer is fresh, at least `bytes` long, and unused by any command
        unsafe { buf.host_mut::<u8>()[..bytes].copy_from_slice(bytemuck::cast_slice(data)) };
        Ok(buf)
    }

    /// Host memory wrapped as a buffer without a copy, with the byte offset of `data` inside it
    ///
    /// The wrapped range is `data` widened to whole pages, which Metal requires; the extra bytes belong to the same
    /// mapped pages and are never read by a kernel. Returns `None` when Metal refuses the range, so the caller can copy
    ///
    /// # Safety
    /// `data` must stay alive and unmodified until every command buffer that reads the buffer has completed
    pub unsafe fn wrap(&self, data: &[u8]) -> Option<(Buf, usize)> {
        // SAFETY: sysconf has no preconditions
        let page = usize::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok().filter(|&p| p > 0)?;
        let start = data.as_ptr() as usize;
        let base = start & !(page - 1);
        let end = (start + data.len()).next_multiple_of(page);
        let ptr = NonNull::new(base as *mut c_void)?;
        // SAFETY: the pages from `base` to `end` are mapped, since `data` lies in them, and the caller keeps them
        // alive for the buffer's use; no deallocator, since the caller owns the memory
        let raw = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                end - base,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }?;
        Some((Buf { raw, bytes: end - base }, start - base))
    }

    /// Encodes kernel `name` over exactly `n` threads, in threadgroups of `per_group` (at most the kernel's limit)
    pub fn dispatch_1d(&self, name: &'static str, n: usize, per_group: usize, args: &[Arg]) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.encode(name, args, |enc, k| {
            let per_group = k.pso.maxTotalThreadsPerThreadgroup().min(per_group);
            enc.dispatchThreads_threadsPerThreadgroup(size([n, 1, 1]), size([per_group, 1, 1]));
            Ok(())
        })
    }

    /// Encodes kernel `name` over `groups` threadgroups of `threads` threads
    pub fn dispatch_groups(&self, name: &'static str, groups: usize, threads: usize, args: &[Arg]) -> Result<()> {
        if groups == 0 {
            return Ok(());
        }
        self.encode(name, args, |enc, k| {
            ensure!(
                threads <= k.pso.maxTotalThreadsPerThreadgroup(),
                "Metal kernel {name}: {threads} threads exceed its limit of {}",
                k.pso.maxTotalThreadsPerThreadgroup()
            );
            enc.dispatchThreadgroups_threadsPerThreadgroup(size([groups, 1, 1]), size([threads, 1, 1]));
            Ok(())
        })
    }

    fn encode(
        &self,
        name: &'static str,
        args: &[Arg],
        run: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>, &Kernel) -> Result<()>,
    ) -> Result<()> {
        let k = self.kernel(name)?;
        ensure!(
            args.len() == k.params.len(),
            "Metal kernel {name} takes {} buffer arguments, got {}",
            k.params.len(),
            args.len()
        );
        for (i, (arg, param)) in args.iter().zip(&k.params).enumerate() {
            let ok = match (arg, param) {
                (Arg::Buf(b, offset), Param::Pointer) => *offset <= b.bytes,
                (Arg::Bytes(b), Param::Pointer) => !b.is_empty() && b.len() <= 4096,
                (Arg::U32(_), Param::Scalar(MTLDataType::UInt)) => true,
                _ => false,
            };
            if !ok {
                bail!("Metal kernel {name}: argument {i} doesn't match parameter {param:?}");
            }
        }

        let mut batch = self.batch.borrow_mut();
        if batch.is_none() {
            let cb = self.queue.commandBuffer().ok_or_else(|| eyre!("Metal could not create a command buffer"))?;
            let enc = self.encoder(&cb, name)?;
            *batch = Some(Batch { cb, enc, dispatches: 0 });
        } else if self.profile.is_some() {
            let b = batch.as_mut().expect("checked above");
            b.enc.endEncoding();
            b.enc = self.encoder(&b.cb, name)?;
        }
        let b = batch.as_mut().expect("opened above");
        b.enc.setComputePipelineState(&k.pso);
        for (i, arg) in args.iter().enumerate() {
            // SAFETY: the buffer outlives the dispatch (Metal retains it until the command buffer completes) and the
            // offset is inside it; bytes and scalars are copied now
            unsafe {
                match *arg {
                    Arg::Buf(buf, offset) => b.enc.setBuffer_offset_atIndex(Some(&buf.raw), offset, i),
                    Arg::Bytes(bytes) => b.enc.setBytes_length_atIndex(
                        NonNull::new(bytes.as_ptr() as *mut c_void).expect("non-empty"),
                        bytes.len(),
                        i,
                    ),
                    Arg::U32(v) => b.enc.setBytes_length_atIndex(NonNull::from(&v).cast(), 4, i),
                }
            }
        }
        run(&b.enc, k)?;
        b.dispatches += 1;
        self.dispatches.set(self.dispatches.get() + 1);
        if self.profile.as_ref().is_some_and(|p| p.borrow().full()) {
            drop(batch);
            return self.synchronize();
        }
        if b.dispatches >= FLUSH_EVERY {
            drop(batch);
            self.commit();
        }
        Ok(())
    }

    fn encoder(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>> {
        let enc = match &self.profile {
            None => cb.computeCommandEncoder(),
            Some(p) => {
                let desc = p.borrow_mut().sample(name);
                cb.computeCommandEncoderWithDescriptor(&desc)
            }
        };
        enc.ok_or_else(|| eyre!("Metal could not create an encoder"))
    }

    /// Commits the open command buffer, if any, without waiting
    pub fn commit(&self) {
        if let Some(b) = self.batch.borrow_mut().take() {
            b.enc.endEncoding();
            b.cb.commit();
            self.inflight.borrow_mut().push(b.cb);
        }
    }

    /// Commits pending work and waits for all of it, failing with the first failed command buffer's error
    pub fn synchronize(&self) -> Result<()> {
        self.commit();
        let inflight = std::mem::take(&mut *self.inflight.borrow_mut());
        let mut failure = None;
        let mut busy = self.busy.borrow_mut();
        for cb in &inflight {
            cb.waitUntilCompleted();
            let (start, end) = (cb.GPUStartTime(), cb.GPUEndTime());
            if start > 0.0 && end >= start {
                busy.push((start, end));
            }
            if cb.status() != MTLCommandBufferStatus::Completed {
                failure.get_or_insert_with(|| failure_reason(cb));
            }
        }
        if !inflight.is_empty() && self.fail_next.replace(false) {
            failure.get_or_insert_with(|| "injected test failure".into());
        }
        if let Some(p) = &self.profile {
            p.borrow_mut().resolve(&self.device)?;
        }
        match failure {
            Some(why) => Err(eyre!("a Metal command buffer failed ({why})")),
            None => Ok(()),
        }
    }

    /// The union of the GPU intervals of the command buffers waited on since the last take, in seconds
    pub fn take_busy_secs(&self) -> f64 {
        crate::stats::union_secs(&mut std::mem::take(&mut *self.busy.borrow_mut()))
    }

    /// Test hook: the next synchronize that waits on work reports a failed submission
    pub fn inject_failure(&self) {
        self.fail_next.set(true);
    }

    /// Dispatches encoded since the last call
    pub fn take_dispatch_count(&self) -> usize {
        self.dispatches.replace(0)
    }

    /// Per-kernel GPU seconds since the last call, sorted by total, with dispatch counts; empty unless profiling
    pub fn take_profile(&self) -> Result<Vec<(&'static str, usize, f64)>> {
        let Some(profile) = &self.profile else { return Ok(Vec::new()) };
        self.synchronize()?;
        let mut by_name: HashMap<&'static str, (usize, f64)> = HashMap::new();
        for (name, t) in profile.borrow_mut().done.drain(..) {
            let e = by_name.entry(name).or_default();
            e.0 += 1;
            e.1 += t;
        }
        let mut rows: Vec<_> = by_name.into_iter().map(|(n, (c, t))| (n, c, t)).collect();
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        Ok(rows)
    }

    /// Prints [`Gpu::take_profile`] to stderr when profiling
    pub fn print_profile(&self, header: &str) -> Result<()> {
        let profile = self.take_profile()?;
        if profile.is_empty() {
            return Ok(());
        }
        let total: f64 = profile.iter().map(|r| r.2).sum();
        eprintln!("TEASAR_PROFILE: {header}: {:.2} ms of GPU time by kernel", total * 1e3);
        for (name, count, t) in profile {
            eprintln!("  {name:>24} {count:>7} {:>10.3} ms {:>5.1}%", t * 1e3, 100.0 * t / total);
        }
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // queued kernels may still use buffers that drop after this
        let _ = self.synchronize();
    }
}

fn size(v: [usize; 3]) -> MTLSize {
    MTLSize { width: v[0], height: v[1], depth: v[2] }
}
