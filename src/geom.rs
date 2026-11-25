//! The bit-packed block, slab geometry, and bit packing

use std::fmt;

use rayon::prelude::*;

use crate::{Error, Result, ensure_value};

/// A binarized z-range, bit-packed along x: byte `[z, y, b]` holds `x = 8b .. 8b + 7`, most significant bit first
#[derive(Clone, Copy)]
pub struct Block<'a> {
    pub data: &'a [u8],
    pub z: usize,
    pub y: usize,
    pub x_size: usize,
}

impl<'a> Block<'a> {
    /// A block over `data`, shaped (z, y, ceil(x_size / 8))
    pub fn new(data: &'a [u8], shape: [usize; 3], x_size: usize) -> Result<Self> {
        ensure_value!(
            shape[2] == x_size.div_ceil(8),
            "block has {} bytes per row, but x_size {x_size} needs {}",
            shape[2],
            x_size.div_ceil(8)
        );
        ensure_value!(data.len() == shape.iter().product::<usize>(), "block data does not match its shape {shape:?}");
        Ok(Self { data, z: shape[0], y: shape[1], x_size })
    }

    pub fn row_bytes(&self) -> usize {
        self.x_size.div_ceil(8)
    }

    /// The full-resolution length of the axis a `kind` slab cuts
    pub fn axis_len(&self, kind: SlabKind) -> usize {
        match kind {
            SlabKind::H => self.z,
            SlabKind::Vy => self.y,
            SlabKind::Vx => self.x_size,
        }
    }

    /// Whether voxel (z, y, x) is foreground
    #[inline]
    pub fn get(&self, z: usize, y: usize, x: usize) -> bool {
        let byte = self.data[(z * self.y + y) * self.row_bytes() + (x >> 3)];
        (byte >> (7 - (x & 7))) & 1 != 0
    }
}

/// The three slab kinds villa cuts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabKind {
    /// z in [lo, hi), all y, all x
    H,
    /// y in [lo, hi), all z, all x
    Vy,
    /// x in [lo, hi), all z, all y
    Vx,
}

impl SlabKind {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "h" => Ok(SlabKind::H),
            "vy" => Ok(SlabKind::Vy),
            "vx" => Ok(SlabKind::Vx),
            _ => Err(Error::Value(format!("slab kind must be 'h', 'vy' or 'vx', got {s:?}"))),
        }
    }

    /// The name `parse` takes
    pub fn name(self) -> &'static str {
        match self {
            SlabKind::H => "h",
            SlabKind::Vy => "vy",
            SlabKind::Vx => "vx",
        }
    }

    fn axis(self) -> usize {
        match self {
            SlabKind::H => 0,
            SlabKind::Vy => 1,
            SlabKind::Vx => 2,
        }
    }
}

/// A slab as the caller names it: `(kind, lo, hi)` in block-local coordinates, before it is checked against a block
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabSpec {
    pub kind: SlabKind,
    pub lo: i64,
    pub hi: i64,
}

impl SlabSpec {
    /// The slab's box in `block`; fails when the range is out of the block
    pub fn geom(&self, block: &Block<'_>) -> Result<SlabGeom> {
        SlabGeom::from_spec(block, self.kind, self.lo, self.hi)
    }
}

/// The Python tuple form, as error messages show it
impl fmt::Display for SlabSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "('{}', {}, {})", self.kind.name(), self.lo, self.hi)
    }
}

/// A slab as a box of the block: origin and extent in full-resolution voxels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabGeom {
    pub origin: [usize; 3],
    pub shape: [usize; 3],
}

impl SlabGeom {
    /// The slab `(kind, lo, hi)` of `block`
    pub fn from_spec(block: &Block<'_>, kind: SlabKind, lo: i64, hi: i64) -> Result<Self> {
        let (axis, len) = (kind.axis(), block.axis_len(kind));
        ensure_value!(
            0 <= lo && lo <= hi && hi as u64 <= len as u64,
            "slab {kind:?} [{lo}, {hi}) is out of range for an axis of length {len}"
        );
        let mut origin = [0; 3];
        let mut shape = [block.z, block.y, block.x_size];
        origin[axis] = lo as usize;
        shape[axis] = (hi - lo) as usize;
        Ok(Self { origin, shape })
    }

    /// The whole block as one slab
    pub fn whole(block: &Block<'_>) -> Self {
        Self { origin: [0; 3], shape: [block.z, block.y, block.x_size] }
    }

    pub fn voxels(&self) -> u64 {
        self.shape.iter().map(|&s| s as u64).product()
    }

    /// Whether any axis is shorter than the pool factor, which gives no pooled voxels
    pub fn is_thin(&self, ds: u32) -> bool {
        self.shape.iter().any(|&s| s < ds as usize)
    }

    /// The pooled shape `(Z // ds, Y // ds, X // ds)`
    pub fn pooled_shape(&self, ds: u32) -> [usize; 3] {
        self.shape.map(|s| s / ds as usize)
    }

    pub fn pooled_voxels(&self, ds: u32) -> u64 {
        self.pooled_shape(ds).iter().map(|&s| s as u64).product()
    }
}

/// Packs a C-order (z, y, x) array of 0/nonzero bytes into a block's layout, writing `dst`
pub fn pack_bits_into(src: &[u8], shape: [usize; 3], dst: &mut [u8]) {
    let x = shape[2];
    let row_bytes = x.div_ceil(8);
    assert_eq!(src.len(), shape.iter().product::<usize>());
    assert_eq!(dst.len(), shape[0] * shape[1] * row_bytes);
    if x == 0 {
        return;
    }
    dst.par_chunks_mut(row_bytes).zip(src.par_chunks(x)).for_each(|(out, row)| {
        for (b, chunk) in out.iter_mut().zip(row.chunks(8)) {
            let mut v = 0u8;
            for (i, &s) in chunk.iter().enumerate() {
                v |= u8::from(s != 0) << (7 - i);
            }
            *b = v;
        }
    });
}

/// [`pack_bits_into`] into a new vector
pub fn pack_bits(src: &[u8], shape: [usize; 3]) -> Vec<u8> {
    let mut dst = vec![0u8; shape[0] * shape[1] * shape[2].div_ceil(8)];
    pack_bits_into(src, shape, &mut dst);
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packing_puts_the_lowest_x_in_the_top_bit() {
        let src = [1u8, 0, 0, 0, 0, 0, 0, 1, 1, 0, 3];
        let packed = pack_bits(&src, [1, 1, 11]);
        assert_eq!(packed, vec![0b1000_0001, 0b1010_0000]);
        let block = Block::new(&packed, [1, 1, 2], 11).unwrap();
        let got: Vec<bool> = (0..11).map(|x| block.get(0, 0, x)).collect();
        let want: Vec<bool> = src.iter().map(|&v| v != 0).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn slabs_cut_one_axis() {
        let data = vec![0u8; 3 * 5 * 2];
        let block = Block::new(&data, [3, 5, 2], 13).unwrap();
        let g = SlabGeom::from_spec(&block, SlabKind::Vx, 6, 10).unwrap();
        assert_eq!(g, SlabGeom { origin: [0, 0, 6], shape: [3, 5, 4] });
        assert!(SlabGeom::from_spec(&block, SlabKind::Vx, 6, 14).is_err());
        assert!(SlabGeom::from_spec(&block, SlabKind::H, -1, 2).is_err());
    }
}
