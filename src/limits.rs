//! The per-call limits (spec part A §2.5) as one table: each limit's name and default
//!
//! Every limit but `output_bytes` is an input check, which runs before any work and raises
//! [`Error::InputTooLarge`] through [`Limits::check`]. `output_bytes` is the output check, which runs once a call's
//! results are known and raises [`Error::OutputTooLarge`] through [`Limits::check_output`]. The spec revises the
//! values, so a change is one row of [`TABLE`]

use crate::{Error, Result};

/// One limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    BlockBytes,
    AxisPooled,
    SlabVoxels,
    SlabPooled,
    CallPooled,
    OutputBytes,
}

struct Row {
    limit: Limit,
    name: &'static str,
    default: u64,
}

const TABLE: [Row; 6] = [
    // the caller's input block
    Row { limit: Limit::BlockBytes, name: "block_bytes", default: 512 << 20 },
    // each pooled axis of a slab: keeps coordinates exact in float32 and squared distances inside uint32
    Row { limit: Limit::AxisPooled, name: "axis_pooled", default: 32_768 },
    // one slab at full resolution: part A's 5.5 bytes per voxel stay inside the working set
    Row { limit: Limit::SlabVoxels, name: "slab_voxels", default: 3 << 26 },
    // one slab pooled: part B's 48 bytes per pooled voxel stay inside the working set
    Row { limit: Limit::SlabPooled, name: "slab_pooled", default: 1 << 24 },
    // a call's pooled voxels, summed over its slabs: the labels retained between part A and part B
    Row { limit: Limit::CallPooled, name: "call_pooled", default: 1 << 26 },
    // the arrays a call returns
    Row { limit: Limit::OutputBytes, name: "output_bytes", default: 512 << 20 },
];

fn row(limit: Limit) -> &'static Row {
    TABLE.iter().find(|r| r.limit == limit).expect("every limit has a row")
}

/// The limits a device checks, one value per row of the table
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits([u64; TABLE.len()]);

impl Default for Limits {
    fn default() -> Self {
        Self(TABLE.map(|r| r.default))
    }
}

impl Limits {
    fn slot(limit: Limit) -> usize {
        TABLE.iter().position(|r| r.limit == limit).expect("every limit has a row")
    }

    pub fn get(&self, limit: Limit) -> u64 {
        self.0[Self::slot(limit)]
    }

    pub fn set(&mut self, limit: Limit, value: u64) {
        self.0[Self::slot(limit)] = value;
    }

    /// The limit named `name` (the table's names, as the errors print them)
    pub fn by_name(name: &str) -> Option<Limit> {
        TABLE.iter().find(|r| r.name == name).map(|r| r.limit)
    }

    /// The input check: fails with [`Error::InputTooLarge`] when `value` is over `limit`; `what` describes the
    /// value, as in "a slab of shape (2, 8, 8) has 128 voxels"
    pub fn check(&self, limit: Limit, value: u64, what: impl FnOnce() -> String) -> Result<()> {
        debug_assert_ne!(limit, Limit::OutputBytes, "the output check is check_output");
        let max = self.get(limit);
        if value <= max {
            return Ok(());
        }
        Err(Error::InputTooLarge(format!("{}: {}, over the limit of {max}", row(limit).name, what())))
    }

    /// The output check over the bytes of each slab's result, in the call's slab order: fails with
    /// [`Error::OutputTooLarge`] naming the first slab whose result takes the running total over the limit
    pub fn check_output(&self, slab_bytes: impl IntoIterator<Item = u64>) -> Result<()> {
        let max = self.get(Limit::OutputBytes);
        let mut total = 0u64;
        for (slab, bytes) in slab_bytes.into_iter().enumerate() {
            total += bytes;
            if total > max {
                let message = format!(
                    "{}: slab {slab} takes the call's results to {total} bytes, over the limit of {max}",
                    row(Limit::OutputBytes).name
                );
                return Err(Error::OutputTooLarge { slab, message });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_name_the_limit_and_the_slab() {
        let mut l = Limits::default();
        l.set(Limit::CallPooled, 10);
        assert!(l.check(Limit::CallPooled, 10, String::new).is_ok());
        let e = l.check(Limit::CallPooled, 11, || "the call's slabs pool to 11 voxels".into()).unwrap_err();
        assert!(matches!(&e, Error::InputTooLarge(m) if m.starts_with("call_pooled: ")), "{e:?}");

        l.set(Limit::OutputBytes, 10);
        assert!(l.check_output([4, 0, 6]).is_ok());
        let e = l.check_output([4, 0, 6, 0, 1, 9]).unwrap_err();
        assert!(matches!(&e, Error::OutputTooLarge { slab: 4, message } if message.starts_with("output_bytes: ")));
        assert_eq!(Limits::by_name("axis_pooled"), Some(Limit::AxisPooled));
    }
}
