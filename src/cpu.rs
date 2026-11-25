//! The CPU reference backend: every stage in pure Rust, multi-threaded across slabs with rayon

pub mod edt;
pub mod km;
pub mod pieces;
pub mod prep;
pub mod skel;

use rayon::prelude::*;

use crate::compat::CompatOrders;
use crate::geom::{Block, SlabGeom};
use crate::skel::Choices;
use crate::{Backend, CallStats, Error, Labels, LabelsRef, Packed, PrepSettings, Result, SettingsSet, SkelSettings};

/// The CPU reference
pub struct CpuBackend {
    pool: rayon::ThreadPool,
    stats: CallStats,
}

impl CpuBackend {
    /// A backend on `threads` worker threads, 0 meaning all cores
    pub fn new(threads: usize) -> Result<Self> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("teasar-cpu-{i}"))
            .build()
            .map_err(|e| Error::Value(format!("could not start {threads} CPU threads: {e}")))?;
        Ok(Self { pool, stats: CallStats::default() })
    }
}

impl Backend for CpuBackend {
    fn backend(&self) -> &'static str {
        "cpu"
    }

    fn host_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    fn take_stats(&mut self) -> CallStats {
        CallStats { host_threads: self.host_threads(), ..std::mem::take(&mut self.stats) }
    }

    fn name(&self) -> String {
        format!("CPU reference ({} threads)", self.pool.current_num_threads())
    }

    fn memory(&self) -> (u64, u64) {
        (0, 0)
    }

    fn prepare_labels(&mut self, block: &Block<'_>, slabs: &[SlabGeom], prep: &PrepSettings) -> Result<Vec<Labels>> {
        Ok(self.pool.install(|| slabs.par_iter().map(|g| prep::prepare_slab(block, g, prep)).collect()))
    }

    fn skeletonize(&mut self, labels: &[LabelsRef<'_>], _set: SettingsSet, skel: &SkelSettings) -> Result<Vec<Packed>> {
        let choices = Choices::current()?;
        let out: Vec<(Packed, skel::Counts)> = self.pool.install(|| {
            labels
                .par_iter()
                .map(|l| {
                    skel::skeletonize_slab(l, skel, crate::effective_fix_borders(skel.fix_borders, l.shape), &choices)
                })
                .collect()
        });
        let mut c = skel::Counts::default();
        out.iter().for_each(|(_, k)| c += *k);
        self.stats.add_cpu(c.pieces, c.voxels, c.busy_ns as f64 * 1e-9);
        if std::env::var_os("TEASAR_PROFILE").is_some_and(|v| v == "1") {
            eprintln!(
                "teasar cpu skeletonize: {} slabs, {} pieces, {} ordinary and {} mandatory traces",
                labels.len(),
                c.pieces,
                c.ordinary,
                c.mandatory
            );
            let k = skel::prof::take();
            let line: Vec<String> = skel::prof::NAMES
                .iter()
                .zip(k)
                .map(
                    |(n, v)| if *n == "settled" { format!("{n} {v}") } else { format!("{n} {:.2} s", v as f64 * 1e-9) },
                )
                .collect();
            eprintln!("teasar cpu skeletonize thread-time: {}", line.join(", "));
        }
        Ok(out.into_iter().map(|(p, _)| p).collect())
    }

    fn skeletonize_compat(
        &mut self,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        orders: &dyn CompatOrders,
    ) -> Result<Vec<Packed>> {
        let fix = |shape| crate::effective_fix_borders(skel.fix_borders, shape);
        let (out, c) = self.pool.install(|| km::skeletonize(labels, skel, fix, orders))?;
        self.stats.add_cpu(c.pieces, c.voxels, c.busy_ns as f64 * 1e-9);
        Ok(out)
    }

    #[cfg(feature = "testing")]
    fn fields(&mut self, labels: &[LabelsRef<'_>], skel: &SkelSettings) -> Result<Vec<crate::skel::FieldDump>> {
        let choices = Choices::current()?;
        let fix = |l: &LabelsRef<'_>| crate::effective_fix_borders(skel.fix_borders, l.shape);
        Ok(self
            .pool
            .install(|| labels.par_iter().map(|l| skel::fields::slab_fields(l, skel, fix(l), &choices)).collect()))
    }
}
