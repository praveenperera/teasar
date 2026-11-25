//! Call statistics (spec part A §1.4): always collected, cheap enough for timed runs

/// What a device's most recent call did: its wall time, the GPU's busy time, and where the pieces ran
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CallStats {
    /// The call's wall time
    pub wall_s: f64,
    /// The union of the call's command-buffer GPU intervals, so overlapping buffers count once
    pub gpu_busy_s: f64,
    /// The host threads per-piece CPU work runs on
    pub host_threads: usize,
    pub cpu_pieces: u64,
    pub cpu_voxels: u64,
    pub gpu_pieces: u64,
    pub gpu_voxels: u64,
    /// The summed busy time of the host threads on per-piece work
    pub cpu_busy_s: f64,
}

impl CallStats {
    /// Adds `pieces` of `voxels` pooled voxels that ran on host threads for `busy_s` thread-seconds
    pub fn add_cpu(&mut self, pieces: u64, voxels: u64, busy_s: f64) {
        self.cpu_pieces += pieces;
        self.cpu_voxels += voxels;
        self.cpu_busy_s += busy_s;
    }

    /// Adds `pieces` of `voxels` pooled voxels that ran on the GPU
    pub fn add_gpu(&mut self, pieces: u64, voxels: u64) {
        self.gpu_pieces += pieces;
        self.gpu_voxels += voxels;
    }
}

/// The total length of the union of `intervals`, each (start, end) in seconds
pub fn union_secs(intervals: &mut [(f64, f64)]) -> f64 {
    intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.0;
    let mut open: Option<(f64, f64)> = None;
    for &(s, e) in intervals.iter() {
        open = match open {
            Some((os, oe)) if s <= oe => Some((os, oe.max(e))),
            Some((os, oe)) => {
                total += oe - os;
                Some((s, e))
            }
            None => Some((s, e)),
        };
    }
    total + open.map_or(0.0, |(s, e)| e - s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_intervals_count_once() {
        let mut v = [(3.0, 4.0), (0.0, 2.0), (1.0, 2.5), (5.0, 5.5)];
        assert_eq!(union_secs(&mut v), 2.5 + 1.0 + 0.5);
        assert_eq!(union_secs(&mut []), 0.0);
    }
}
