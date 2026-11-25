//! Exact squared Euclidean distance transforms of a piece-id array (spec part B §2.2, §2.4 and §2.5)
//!
//! Separable, one axis at a time (Felzenszwalb and Huttenlocher's lower envelope of parabolas). Along each line the
//! envelope is built per run of equal id, from the run's own values plus the first voxel of another id on each side,
//! which is at distance 0 from the run; a run's values never reach past those voxels, since any candidate beyond one
//! is farther than it. Envelope breakpoints are compared as exact rationals, so the result is exact

/// The distance of a voxel that sees no boundary at all
pub const FAR: u32 = u32::MAX;

/// Runs this short take the minimum over their candidates instead of a lower envelope
const SHORT_RUN: usize = 8;

/// For every voxel of `ids` (C order, `shape`), the squared distance in voxels to the nearest voxel of a different
/// id; with `border[k]`, the positions just outside the array along axis `k` count as such voxels too
pub fn edt(ids: &[u32], shape: [usize; 3], border: [bool; 3]) -> Vec<u32> {
    edt_except(ids, shape, border, None)
}

/// `edt`, except that voxels of id `unused` get no distance of their own: their values are unspecified, though they
/// still bound the other ids' runs. A run's values only feed later passes of the same run, so skipping them leaves
/// every other voxel's distance unchanged
pub fn edt_except(ids: &[u32], shape: [usize; 3], border: [bool; 3], unused: Option<u32>) -> Vec<u32> {
    debug_assert_eq!(ids.len(), shape.iter().product::<usize>());
    let mut f = vec![FAR; ids.len()];
    let strides = [shape[1] * shape[2], shape[2], 1];
    let mut line = Line { unused, ..Line::default() };
    for axis in [2, 1, 0] {
        let n = shape[axis];
        if n == 1 && !border[axis] {
            // a one-voxel line with no border sees no boundary along it, so this pass would keep every value
            continue;
        }
        let (a, b) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for i in 0..shape[a] {
            for j in 0..shape[b] {
                let start = i * strides[a] + j * strides[b];
                line.run(ids, &mut f, start, strides[axis], n, border[axis]);
            }
        }
    }
    f
}

/// Scratch for one line, and the id whose runs get no distances
#[derive(Default)]
struct Line {
    unused: Option<u32>,
    ids: Vec<u32>,
    vals: Vec<u32>,
    cands: Vec<(i64, i64)>,
    hull: Vec<usize>,
    breaks: Vec<(i64, i64)>,
}

impl Line {
    fn run(&mut self, ids: &[u32], f: &mut [u32], start: usize, stride: usize, n: usize, border: bool) {
        self.ids.clear();
        self.vals.clear();
        for k in 0..n {
            self.ids.push(ids[start + k * stride]);
            self.vals.push(f[start + k * stride]);
        }

        // positions and values stay small enough for exact i64 breakpoint products (see `le`)
        let wide = n > 1 << 16;
        let mut s = 0;
        while s < n {
            let id = self.ids[s];
            let mut e = s + 1;
            while e < n && self.ids[e] == id {
                e += 1;
            }
            if self.unused == Some(id) {
                s = e;
                continue;
            }
            // the run's boundary voxels, at distance 0: the neighbouring runs, or the border positions
            let left = if s > 0 { Some(s as i64 - 1) } else { border.then_some(-1) };
            let right = if e < n { Some(e as i64) } else { border.then_some(n as i64) };

            if self.vals[s..e].iter().all(|&v| v == FAR) {
                // only the boundary voxels are candidates (every first pass): the distance to the nearer one
                for k in s..e {
                    let x = k as i64;
                    let d = [left.map(|p| x - p), right.map(|p| p - x)].into_iter().flatten().min();
                    f[start + k * stride] = d.map_or(FAR, |d| (d * d) as u32);
                }
                s = e;
                continue;
            }
            if e - s <= SHORT_RUN {
                // few candidates: the minimum over all of them is cheaper than an envelope
                for k in s..e {
                    let x = k as i64;
                    let mut best = [left.map(|p| x - p), right.map(|p| p - x)]
                        .into_iter()
                        .flatten()
                        .map(|d| d * d)
                        .min()
                        .unwrap_or(i64::MAX);
                    for (j, &v) in self.vals[s..e].iter().enumerate() {
                        if v != FAR {
                            let d = x - (s + j) as i64;
                            best = best.min(d * d + i64::from(v));
                        }
                    }
                    f[start + k * stride] = if best == i64::MAX { FAR } else { best as u32 };
                }
                s = e;
                continue;
            }

            self.cands.clear();
            if let Some(p) = left {
                self.cands.push((p, 0));
            }
            for k in s..e {
                if self.vals[k] != FAR {
                    self.cands.push((k as i64, i64::from(self.vals[k])));
                }
            }
            if let Some(p) = right {
                self.cands.push((p, 0));
            }
            self.envelope(wide);

            let mut h = 0;
            for k in s..e {
                f[start + k * stride] = self.eval(k as i64, &mut h);
            }
            s = e;
        }
    }

    /// The lower envelope of the candidate parabolas `(x - pos)^2 + val`; `breaks[k]` (a rational num / den) is
    /// where parabola `hull[k]` takes over from `hull[k - 1]`, and `breaks[0]` is unused
    fn envelope(&mut self, wide: bool) {
        self.hull.clear();
        self.breaks.clear();
        for q in 0..self.cands.len() {
            let mut start = (0, 1);
            while let Some(&top) = self.hull.last() {
                start = self.meet(top, q);
                // two of these parabolas always meet exactly once; `q` hides `top` if it starts before `top` does
                if self.hull.len() > 1 && le(start, self.breaks[self.hull.len() - 1], wide) {
                    self.hull.pop();
                    self.breaks.pop();
                    continue;
                }
                break;
            }
            self.hull.push(q);
            self.breaks.push(start);
        }
    }

    /// Where parabola `q` (further right) starts to lie below parabola `v`
    fn meet(&self, v: usize, q: usize) -> (i64, i64) {
        let (pv, fv) = self.cands[v];
        let (pq, fq) = self.cands[q];
        ((fq + pq * pq) - (fv + pv * pv), 2 * (pq - pv))
    }

    /// The envelope at `x`, with `k` the hull entry for the previous (smaller) `x` of the same run
    fn eval(&self, x: i64, k: &mut usize) -> u32 {
        if self.hull.is_empty() {
            return FAR;
        }
        while *k + 1 < self.hull.len() && le(self.breaks[*k + 1], (x, 1), false) {
            *k += 1;
        }
        let (p, v) = self.cands[self.hull[*k]];
        let d = x - p;
        (d * d + v) as u32
    }
}

/// `a <= b` for rationals with positive denominators. With line lengths up to 2^16 and u32 values, numerators are
/// below 2^34 and denominators below 2^18, so the i64 products are exact; `wide` (longer) lines use i128
#[inline]
fn le(a: (i64, i64), b: (i64, i64), wide: bool) -> bool {
    if wide {
        return i128::from(a.0) * i128::from(b.1) <= i128::from(b.0) * i128::from(a.1);
    }
    a.0 * b.1 <= b.0 * a.1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, by brute force
    fn brute(ids: &[u32], shape: [usize; 3], border: [bool; 3]) -> Vec<u32> {
        let [z, y, x] = shape;
        let mut out = vec![FAR; ids.len()];
        for i in 0..ids.len() {
            let (pz, py, px) = ((i / (y * x)) as i64, ((i / x) % y) as i64, (i % x) as i64);
            let mut best = u64::MAX;
            for j in 0..ids.len() {
                if ids[j] != ids[i] {
                    let (qz, qy, qx) = ((j / (y * x)) as i64, ((j / x) % y) as i64, (j % x) as i64);
                    best = best.min(((pz - qz).pow(2) + (py - qy).pow(2) + (px - qx).pow(2)) as u64);
                }
            }
            for (k, &(p, len)) in [(pz, z), (py, y), (px, x)].iter().enumerate() {
                if border[k] {
                    let d = (p + 1).min(len as i64 - p) as u64;
                    best = best.min(d * d);
                }
            }
            out[i] = if best == u64::MAX { FAR } else { best as u32 };
        }
        out
    }

    #[test]
    fn matches_brute_force_on_random_ids() {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        for trial in 0..300 {
            // thin slabs and long lines too, so short runs, long runs and the envelope all occur
            let shape = match trial % 3 {
                0 => [1 + (next() % 3) as usize, 1 + (next() % 9) as usize, 1 + (next() % 11) as usize],
                1 => [1 + (next() % 2) as usize, 1 + (next() % 4) as usize, 20 + (next() % 60) as usize],
                _ => [1 + (next() % 4) as usize, 15 + (next() % 30) as usize, 1 + (next() % 3) as usize],
            };
            let n: usize = shape.iter().product();
            let kinds = 1 + next() % 4;
            // half the trials clumpy, so runs past SHORT_RUN take the envelope
            let clumpy = trial % 2 == 1;
            let mut id = 0;
            let ids: Vec<u32> = (0..n)
                .map(|_| {
                    if !clumpy || next() % 10 == 0 {
                        id = next() % kinds;
                    }
                    id
                })
                .collect();
            let border = [next() % 2 == 0, next() % 2 == 0, next() % 2 == 0];
            let want = brute(&ids, shape, border);
            assert_eq!(edt(&ids, shape, border), want, "trial {trial} {shape:?} {border:?}");
            // id 0's own distances are skipped; every other voxel's must not change
            let got = edt_except(&ids, shape, border, Some(0));
            for (k, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(ids[k] == 0 || g == w, "trial {trial} voxel {k}: {g} != {w}");
            }
        }
    }

    #[test]
    fn adjacent_rows_see_each_other() {
        // spec part B §2.2: rows of ids 1 and 2 hold no boundary along x, only across y
        let mut ids = vec![1u32; 120];
        ids[60..].fill(2);
        assert!(edt(&ids, [1, 2, 60], [false; 3]).iter().all(|&d| d == 1));
    }
}
