//! `Packed`: the skeletons of one slab

/// Skeletons in flat arrays: skeleton k has vertices `v_off[k]..v_off[k+1]` and edges `e_off[k]..e_off[k+1]`, and
/// its edges index its own vertices from 0
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Packed {
    pub ids: Vec<u32>,
    pub v_off: Vec<i64>,
    pub e_off: Vec<i64>,
    pub vertices: Vec<[f32; 3]>,
    pub edges: Vec<[u32; 2]>,
    pub radii: Vec<f32>,
}

impl Packed {
    /// No skeletons: `v_off = e_off = [0]`
    pub fn empty() -> Self {
        Self { v_off: vec![0], e_off: vec![0], ..Self::default() }
    }

    /// Appends the skeleton of label `id` in sorted form: vertices in increasing C-order index
    /// of their voxel, every edge written (lower, higher), and the edge rows sorted
    ///
    /// `verts` holds (voxel index in the slab, radius) for every vertex of every piece of the label, in any order and
    /// each voxel once; `edges` joins voxel indices. `shape` is the slab's (z, y, x) and `a` its voxel size per axis.
    /// Both buffers are left in an unspecified state for reuse
    pub fn push_sorted(
        &mut self,
        id: u32,
        verts: &mut [(u32, f32)],
        edges: &mut [[u32; 2]],
        shape: [usize; 3],
        a: [f64; 3],
    ) {
        verts.sort_unstable_by_key(|&(g, _)| g);
        let (y, x) = (shape[1], shape[2]);
        for &(g, r) in verts.iter() {
            let g = g as usize;
            let (zz, yy, xx) = (g / (y * x), (g / x) % y, g % x);
            self.vertices.push([(zz as f64 * a[0]) as f32, (yy as f64 * a[1]) as f32, (xx as f64 * a[2]) as f32]);
            self.radii.push(r);
        }

        let index = |g: u32| verts.binary_search_by_key(&g, |&(v, _)| v).expect("an edge joins two vertices") as u32;
        for e in edges.iter_mut() {
            let (u, v) = (index(e[0]), index(e[1]));
            *e = [u.min(v), u.max(v)];
        }
        edges.sort_unstable();
        self.edges.extend_from_slice(edges);
        self.ids.push(id);
        self.v_off.push(self.vertices.len() as i64);
        self.e_off.push(self.edges.len() as i64);
    }

    /// Bytes of the arrays a caller keeps
    pub fn nbytes(&self) -> u64 {
        (self.ids.len() * 4
            + (self.v_off.len() + self.e_off.len()) * 8
            + self.vertices.len() * 12
            + self.edges.len() * 8
            + self.radii.len() * 4) as u64
    }
}
