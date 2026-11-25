//! The pre-skeleton stages on the CPU: 6-connected components over x-runs, max-pool and the size filter
//!
//! This is deliberately a different algorithm from the GPU's (cell-local pieces joined by a lock-free union-find), so
//! the two check each other. Components are found on runs of foreground along x: runs are numbered in C order, so the
//! smallest run index in a component holds the component's first voxel, and ranking the roots in run order gives
//! `scipy.ndimage.label`'s numbering

use crate::geom::{Block, SlabGeom};
use crate::{Labels, PrepSettings};

/// A run of foreground voxels `[start, end)` along x in one (z, y) row
#[derive(Clone, Copy)]
struct Run {
    start: u32,
    end: u32,
}

/// The pooled, filtered labels of one slab, which must not be thin
pub fn prepare_slab(block: &Block<'_>, g: &SlabGeom, prep: &PrepSettings) -> Labels {
    let [zs, ys, _] = g.shape;
    let (runs, row_start) = find_runs(block, g);
    let mut parent: Vec<u32> = (0..runs.len() as u32).collect();

    for z in 0..zs {
        for y in 0..ys {
            let row = z * ys + y;
            if y > 0 {
                join_rows(&runs, &row_start, row, row - 1, &mut parent);
            }
            if z > 0 {
                join_rows(&runs, &row_start, row, row - ys, &mut parent);
            }
        }
    }

    // every root precedes its members, so one pass in run order numbers roots and copies labels to members
    let mut label = vec![0u32; runs.len()];
    let mut count = 0u32;
    for i in 0..runs.len() {
        let r = find(&mut parent, i as u32) as usize;
        if r == i {
            count += 1;
            label[i] = count;
        } else {
            label[i] = label[r];
        }
    }

    let ds = prep.downsample as usize;
    let [pz, py, px] = g.pooled_shape(prep.downsample);
    let mut out = vec![0u32; pz * py * px];
    let x_crop = (px * ds) as u32;
    for z in 0..pz * ds {
        for y in 0..py * ds {
            let row = z * ys + y;
            let cell_row = ((z / ds) * py + y / ds) * px;
            for i in row_start[row]..row_start[row + 1] {
                let run = runs[i];
                let end = run.end.min(x_crop);
                if run.start >= end {
                    continue;
                }
                for cx in (run.start as usize / ds)..=((end - 1) as usize / ds) {
                    let o = &mut out[cell_row + cx];
                    *o = (*o).max(label[i]);
                }
            }
        }
    }

    let mut sizes = vec![0u32; count as usize + 1];
    for &l in &out {
        sizes[l as usize] += 1;
    }
    for l in &mut out {
        let n = sizes[*l as usize];
        if *l != 0 && (n < prep.min_count || n >= prep.max_count) {
            *l = 0;
        }
    }

    Labels { shape: [pz, py, px], data: out }
}

/// The slab's runs in C order, with the index of each row's first run (and one past the last row)
fn find_runs(block: &Block<'_>, g: &SlabGeom) -> (Vec<Run>, Vec<usize>) {
    let [zs, ys, xs] = g.shape;
    let [z0, y0, x0] = g.origin;
    let mut runs = Vec::new();
    let mut row_start = Vec::with_capacity(zs * ys + 1);
    let row_bytes = block.row_bytes();
    for z in 0..zs {
        for y in 0..ys {
            row_start.push(runs.len());
            let row = &block.data[((z0 + z) * block.y + y0 + y) * row_bytes..][..row_bytes];
            let bit = |x: usize| (row[(x0 + x) >> 3] >> (7 - ((x0 + x) & 7))) & 1 != 0;
            let mut x = 0;
            while x < xs {
                // skip whole zero bytes when aligned
                if (x0 + x) & 7 == 0 && x + 8 <= xs && row[(x0 + x) >> 3] == 0 {
                    x += 8;
                    continue;
                }
                if !bit(x) {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < xs && bit(x) {
                    x += 1;
                }
                runs.push(Run { start: start as u32, end: x as u32 });
            }
        }
    }
    row_start.push(runs.len());
    (runs, row_start)
}

/// Joins the runs of row `a` with the overlapping runs of row `b`; face neighbours overlap in x
fn join_rows(runs: &[Run], row_start: &[usize], a: usize, b: usize, parent: &mut [u32]) {
    let (mut i, ie) = (row_start[a], row_start[a + 1]);
    let (mut j, je) = (row_start[b], row_start[b + 1]);
    while i < ie && j < je {
        let (ra, rb) = (runs[i], runs[j]);
        if ra.start < rb.end && rb.start < ra.end {
            union(parent, i as u32, j as u32);
        }
        if ra.end < rb.end {
            i += 1;
        } else {
            j += 1;
        }
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

/// Links the larger root under the smaller, so each component's root is its smallest run index
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::pack_bits;

    fn run(dense: &[u8], shape: [usize; 3], ds: u32, min: u32, max: u32) -> Labels {
        let packed = pack_bits(dense, shape);
        let block = Block::new(&packed, [shape[0], shape[1], shape[2].div_ceil(8)], shape[2]).unwrap();
        let prep = PrepSettings { downsample: ds, min_count: min, max_count: max };
        prepare_slab(&block, &SlabGeom::whole(&block), &prep)
    }

    #[test]
    fn numbers_components_by_first_voxel_in_c_order() {
        // two components: one starts at (0, 0, 2), the other at (0, 1, 0)
        #[rustfmt::skip]
        let dense = [
            0, 0, 1,
            1, 0, 1,
            1, 0, 0,
        ];
        let l = run(&dense, [1, 3, 3], 1, 1, 100);
        assert_eq!(l.data, vec![0, 0, 1, 2, 0, 1, 2, 0, 0]);
    }

    #[test]
    fn a_u_shape_merges_under_its_first_voxel() {
        // the right arm starts first; the left arm joins it through the bottom row
        #[rustfmt::skip]
        let dense = [
            0, 0, 1,
            1, 0, 1,
            1, 1, 1,
        ];
        let l = run(&dense, [1, 3, 3], 1, 1, 100);
        assert_eq!(l.data, vec![0, 0, 1, 1, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn pooling_keeps_the_higher_label() {
        // in one 2x2x2 cell: voxel (0,0,0) is label 1, voxel (1,1,1) is label 2
        let mut dense = vec![0u8; 8];
        dense[0] = 1;
        dense[7] = 1;
        let l = run(&dense, [2, 2, 2], 2, 1, 100);
        assert_eq!(l.data, vec![2]);
    }

    #[test]
    fn components_join_through_the_cropped_tail() {
        // columns x = 0 and x = 2 join only through row y = 2, which the crop to y < 2 removes
        #[rustfmt::skip]
        let plane = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 1, 1, 0,
        ];
        let dense: Vec<u8> = plane.iter().chain(plane.iter()).copied().collect();
        let l = run(&dense, [2, 3, 4], 2, 1, 100);
        assert_eq!(l.data, vec![1, 1]);
        // without the bridge row they are two components
        let mut cut = dense.clone();
        cut[9] = 0;
        cut[12 + 9] = 0;
        assert_eq!(run(&cut, [2, 3, 4], 2, 1, 100).data, vec![1, 2]);
    }

    #[test]
    fn size_filter_keeps_min_up_to_max_exclusive() {
        // three runs of length 2, 3 and 4 along x in rows 0, 2 and 4
        let mut dense = vec![0u8; 5 * 5];
        dense[0..2].fill(1);
        dense[10..13].fill(1);
        dense[20..24].fill(1);
        let l = run(&dense, [1, 5, 5], 1, 3, 4);
        let kept: Vec<u32> = l.data.iter().copied().filter(|&v| v != 0).collect();
        assert_eq!(kept, vec![2, 2, 2]);
    }
}
