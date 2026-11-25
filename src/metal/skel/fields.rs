//! B7's field-level dump on the GPU (spec part B §4.7): the path loop runs with `F_FIELDS_STOP`, which ends each
//! piece after its fields with D, DBF and B left in its workspace

use color_eyre::eyre::{Result, ensure};

use super::{F_FIELDS_STOP, PH_DONE, Skel, WS_ARRAYS, WS_VOX};
use crate::metal::gpu::Gpu;
use crate::skel::{Choices, FieldDump};
use crate::{LabelsRef, SkelSettings};

/// Workspace arrays the stopped path loop leaves the fields in (`WS_*` in `skel.metal`)
const WS_COST: usize = 1;
const WS_HCOST: usize = 2;
const WS_HLOC: usize = 3;
const WS_P: usize = 5;

impl Skel {
    /// The fields of every piece of each slab, one slab per batch, with every piece on the GPU
    pub fn fields(
        &mut self,
        gpu: &Gpu,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        choices: &Choices,
    ) -> Result<Vec<FieldDump>> {
        let saved = (self.cpu_piece, self.large_limit);
        (self.cpu_piece, self.large_limit) = (u32::MAX, u32::MAX);
        let out = labels.iter().map(|l| self.slab_fields(gpu, std::slice::from_ref(l), skel, choices)).collect();
        (self.cpu_piece, self.large_limit) = saved;
        out
    }

    fn slab_fields(
        &mut self,
        gpu: &Gpu,
        labels: &[LabelsRef<'_>],
        skel: &SkelSettings,
        choices: &Choices,
    ) -> Result<FieldDump> {
        let mut out = FieldDump::new(labels[0].data.len());
        let b = self.pieces(gpu, labels, skel)?;
        if b.kept.is_empty() {
            return Ok(out);
        }

        let st = self.slab_stages(gpu, labels, &b, skel, choices)?;
        let states = self.path_loop(gpu, &b, &st, skel, choices, F_FIELDS_STOP)?;
        // SAFETY: the path loop has completed
        let ws = unsafe { b.ws.host::<u32>() };
        let base = b.slabs[0].base;
        for (k, pc) in b.pieces.iter().enumerate() {
            ensure!(pc.host == 0 && states[k].phase == PH_DONE, "piece {} did not stop after its fields", pc.pid);
            let m = pc.m as usize;
            let w = &ws[pc.ws as usize..pc.ws as usize + WS_ARRAYS * m];
            let at = |arr: usize, i: usize| w[arr * m + i];
            for i in 0..m {
                let g = (at(WS_VOX, i) - base) as usize;
                out.dbf[g] = at(WS_HCOST, i);
                out.b[g] = at(WS_HLOC, i);
                out.d[g] = at(WS_COST, i);
                out.p[g] = at(WS_P, i);
            }
            out.roots.push(at(WS_VOX, states[k].root as usize) - base);
        }
        out.roots.sort_unstable();
        Ok(out)
    }
}
