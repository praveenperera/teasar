//! The 26-connected pieces of a pooled label array and which labels get a skeleton (spec part A §3.3)
//!
//! A label is made of 26-connected pieces; a label with at least one piece over `dust_threshold` voxels yields one
//! skeleton, and skeletons are ordered by each label's first occurrence in a Fortran-order (first axis fastest) scan

use crate::LabelsRef;

/// One 26-connected piece of one label
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub label: u32,
    pub voxels: u32,
    /// the piece's first voxel in C order
    pub first: u32,
}

/// Every voxel's piece (`u32::MAX` for background) and the pieces, numbered by first voxel in C order
pub struct Pieces {
    pub of_voxel: Vec<u32>,
    pub pieces: Vec<Piece>,
}

/// The 13 neighbours (dz, dy, dx) that precede a voxel in C order; the last is the left one
const BEFORE: [(i64, i64, i64); 13] = [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
];

/// Those not touching the left neighbour (dx = 1); the others precede it, so a matching left neighbour already joined
/// them
const AFTER_LEFT: [(i64, i64, i64); 4] = [(-1, -1, 1), (-1, 0, 1), (-1, 1, 1), (0, -1, 1)];

/// Those neither left nor touching the (0, -1, 0) neighbour while preceding it
const AFTER_UP: [(i64, i64, i64); 4] = [(-1, 1, -1), (-1, 1, 0), (-1, 1, 1), (0, -1, 1)];

/// The 26-connected pieces of each label of `labels`
pub fn pieces26(labels: &LabelsRef<'_>) -> Pieces {
    let [z, y, x] = labels.shape;
    let n = labels.data.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let at = |zz: usize, yy: usize, xx: usize| (zz * y + yy) * x + xx;
    for zz in 0..z {
        for yy in 0..y {
            for xx in 0..x {
                let i = at(zz, yy, xx);
                let l = labels.data[i];
                if l == 0 {
                    continue;
                }
                let same = |(dz, dy, dx): (i64, i64, i64)| {
                    let (nz, ny, nx) = (zz as i64 + dz, yy as i64 + dy, xx as i64 + dx);
                    if nz < 0 || ny < 0 || nx < 0 || ny >= y as i64 || nx >= x as i64 {
                        return None;
                    }
                    let j = at(nz as usize, ny as usize, nx as usize);
                    (labels.data[j] == l).then_some(j as u32)
                };
                // a matching left or up neighbour already joined the preceding voxels it touches, so only the rest
                // are tested
                let rest: &[(i64, i64, i64)] = if let Some(j) = same((0, 0, -1)) {
                    union(&mut parent, i as u32, j);
                    &AFTER_LEFT
                } else if let Some(j) = same((0, -1, 0)) {
                    union(&mut parent, i as u32, j);
                    &AFTER_UP
                } else {
                    &BEFORE[..12]
                };
                for &step in rest {
                    if let Some(j) = same(step) {
                        union(&mut parent, i as u32, j);
                    }
                }
            }
        }
    }

    let mut of_voxel = vec![u32::MAX; n];
    let mut pieces: Vec<Piece> = Vec::new();
    for i in 0..n {
        if labels.data[i] == 0 {
            continue;
        }
        let r = find(&mut parent, i as u32) as usize;
        // roots are each piece's smallest C-order index, so a root is met before its members
        if r == i {
            of_voxel[i] = pieces.len() as u32;
            pieces.push(Piece { label: labels.data[i], voxels: 0, first: i as u32 });
        } else {
            of_voxel[i] = of_voxel[r];
        }
        pieces[of_voxel[i] as usize].voxels += 1;
    }
    Pieces { of_voxel, pieces }
}

/// The labels that get a skeleton, in skeleton order: those with a piece over `dust_threshold` voxels, by the first
/// voxel of such a piece in a Fortran-order scan
pub fn skeleton_ids(labels: &LabelsRef<'_>, pieces: &Pieces, dust_threshold: u32) -> Vec<u32> {
    let [z, y, _] = labels.shape;
    let mut keep: Vec<u32> = pieces.pieces.iter().filter(|p| p.voxels > dust_threshold).map(|p| p.label).collect();
    keep.sort_unstable();
    keep.dedup();
    let mut first_f = vec![u64::MAX; keep.len()];
    // the first voxel of a kept piece: dust pieces do not count toward a label's place (the fixtures' order)
    for (i, &p) in pieces.of_voxel.iter().enumerate() {
        let Some(piece) = pieces.pieces.get(p as usize).filter(|pc| pc.voxels > dust_threshold) else { continue };
        let Ok(k) = keep.binary_search(&piece.label) else { continue };
        let xx = i % labels.shape[2];
        let yy = (i / labels.shape[2]) % y;
        let zz = i / (labels.shape[2] * y);
        let f = (zz + z * (yy + y * xx)) as u64;
        first_f[k] = first_f[k].min(f);
    }
    let mut order: Vec<usize> = (0..keep.len()).collect();
    order.sort_unstable_by_key(|&k| first_f[k]);
    order.into_iter().map(|k| keep[k]).collect()
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize];
        x = p;
    }
    x
}

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

    #[test]
    fn diagonal_neighbours_join_and_dust_is_per_piece() {
        // label 5: a diagonal pair (one 26-piece of 2) and a far single voxel; label 3: one voxel
        let mut data = vec![0u32; 3 * 3 * 4];
        data[0] = 5;
        data[(3 + 1) * 4 + 1] = 5;
        data[(2 * 3 + 2) * 4 + 3] = 5;
        data[3] = 3;
        let l = LabelsRef { shape: [3, 3, 4], data: &data };
        let p = pieces26(&l);
        let sizes: Vec<(u32, u32)> = p.pieces.iter().map(|q| (q.label, q.voxels)).collect();
        assert_eq!(sizes, vec![(5, 2), (3, 1), (5, 1)]);
        assert_eq!(skeleton_ids(&l, &p, 1), vec![5]);
        // Fortran order: label 3 at (0, 0, 3) has f = 27, label 5 at (0, 0, 0) has f = 0
        assert_eq!(skeleton_ids(&l, &p, 0), vec![5, 3]);
    }

    /// Pieces by flood fill over all 26 neighbours, numbered by first voxel in C order
    fn flood(data: &[u32], shape: [usize; 3]) -> Vec<u32> {
        let [z, y, x] = shape.map(|s| s as i64);
        let mut out = vec![u32::MAX; data.len()];
        let mut next = 0;
        for start in 0..data.len() {
            if data[start] == 0 || out[start] != u32::MAX {
                continue;
            }
            out[start] = next;
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                let c = [i as i64 / (y * x), (i as i64 / x) % y, i as i64 % x];
                for d in 0..27 {
                    let n = [c[0] + d / 9 - 1, c[1] + (d / 3) % 3 - 1, c[2] + d % 3 - 1];
                    if n[0] < 0 || n[1] < 0 || n[2] < 0 || n[0] >= z || n[1] >= y || n[2] >= x {
                        continue;
                    }
                    let j = ((n[0] * y + n[1]) * x + n[2]) as usize;
                    if data[j] == data[start] && out[j] == u32::MAX {
                        out[j] = next;
                        stack.push(j);
                    }
                }
            }
            next += 1;
        }
        out
    }

    #[test]
    fn matches_flood_fill_on_random_labels() {
        let mut seed = 99u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        for trial in 0..300 {
            let shape = [1 + next() % 5, 1 + next() % 6, 1 + next() % 7];
            let kinds = 1 + next() % 3;
            let data: Vec<u32> = (0..shape.iter().product::<usize>()).map(|_| (next() % (kinds + 1)) as u32).collect();
            let p = pieces26(&LabelsRef { shape, data: &data });
            assert_eq!(p.of_voxel, flood(&data, shape), "trial {trial} {shape:?}");
        }
    }
}
