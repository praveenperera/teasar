//! The field-level dump on the CPU reference: every traced piece's fields, as the path loop
//! starts from them

use rayon::prelude::*;

use super::{PieceFields, Slab, piece_fields};
use crate::skel::{Choices, FieldDump, radius};
use crate::{LabelsRef, SkelSettings};

/// The fields of every piece of `labels` above the dust threshold
pub fn slab_fields(labels: &LabelsRef<'_>, skel: &SkelSettings, fix_borders: bool, choices: &Choices) -> FieldDump {
    let slab = Slab::new(labels, skel, fix_borders, choices);
    let a = skel.anisotropy[0];
    let kept: Vec<u32> = (0..slab.pieces.pieces.len() as u32)
        .filter(|&p| slab.pieces.pieces[p as usize].voxels > skel.dust_threshold)
        .collect();
    let per: Vec<(u32, PieceFields)> = kept
        .par_iter()
        .map(|&pid| {
            let ctx = slab.ctx(pid);
            let graph = ctx.graph(&|g| slab.local[g]);
            (pid, piece_fields(&ctx, &graph))
        })
        .collect();

    let mut out = FieldDump::new(labels.data.len());
    for (pid, f) in per {
        let vox = slab.vox(pid);
        for (i, &g) in vox.iter().enumerate() {
            let g = g as usize;
            out.dbf[g] = radius(slab.n_dbf[g], a).to_bits();
            out.b[g] = f.pen[i].to_bits();
            out.d[g] = f.d[i];
            out.p[g] = f.p[i].to_bits();
        }
        out.roots.push(vox[f.root]);
    }
    out.roots.sort_unstable();
    out
}
