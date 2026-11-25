// The pre-skeleton stages on the GPU: 6-connected components at full resolution, max-pool by ds and the pooled size
// filter, for every slab of an internal batch at once
//
// The full-resolution label volume never exists. Each pooled cell (ds^3 voxels, at most 64 bits, read straight
// from the bit-packed block) splits into its 6-connected pieces; pieces are the nodes of a union-find, joined across
// cell faces. A component's number is the rank of its first voxel in C order among all components of the slab, as
// scipy numbers them, and a cell's pooled label is the highest number among its pieces. Cells past the crop are
// still cells, so components that touch only the cropped tail are counted and still connect what they join
//
// Kernel order per batch: prep_count, scan, prep_init, prep_union, prep_compress, prep_keymin, fill, prep_bcount,
// scan, prep_scatter, prep_rank, prep_label, prep_pool, prep_filter

// the cell's foreground bits, read from the packed block; voxels outside the slab read as background
inline ulong cell_mask(device const uchar* blk, constant PrepParams& p, SlabDesc s, uint cz, uint cy, uint cx) {
    uint ds = p.ds;
    uint x_lo = cx * ds;
    uint cnt = min(ds, s.xs - x_lo);
    uint bx = s.x0 + x_lo;
    uint byte = bx >> 3;
    uint sh = bx & 7;
    bool two = sh + cnt > 8;
    uint keep = (1u << cnt) - 1;
    ulong m = 0;
    for (uint dz = 0; dz < ds; dz++) {
        uint z = cz * ds + dz;
        if (z >= s.zs) break;
        for (uint dy = 0; dy < ds; dy++) {
            uint y = cy * ds + dy;
            if (y >= s.ys) break;
            uint addr = ((s.z0 + z) * p.block_rows + (s.y0 + y)) * p.row_bytes + byte;
            uint w = uint(blk[addr]) << 8;
            if (two) w |= uint(blk[addr + 1]);
            // the block stores the lowest x in the top bit; flip so dx = 0 is the lowest bit
            uint bits = (w >> (16 - sh - cnt)) & keep;
            bits = reverse_bits(bits) >> (32 - cnt);
            m |= ulong(bits) << ((dz * ds + dy) * ds);
        }
    }
    return m;
}

// the 6-connected piece of m that holds the seed bits
inline ulong flood(ulong seed, ulong m, constant PrepParams& p) {
    ulong c = seed;
    for (;;) {
        ulong n = c | ((c & p.can_xp) << 1) | ((c & p.can_xm) >> 1) | ((c & p.can_yp) << p.ds)
                  | ((c & p.can_ym) >> p.ds) | (c << p.ds2) | (c >> p.ds2);
        n &= m;
        if (n == c) return c;
        c = n;
    }
}

inline uint count_pieces(ulong m, constant PrepParams& p) {
    uint k = 0;
    while (m != 0) {
        m &= ~flood(m & (~m + 1), m, p);
        k++;
    }
    return k;
}

struct Cell {
    SlabDesc s;
    uint cz, cy, cx;
};

inline Cell locate(device const SlabDesc* slabs, constant PrepParams& p, uint c) {
    Cell r;
    r.s = slabs[slab_by_cell(slabs, p.nslabs, c)];
    uint local = c - r.s.cell_base;
    r.cx = local % r.s.cx;
    uint t = local / r.s.cx;
    r.cy = t % r.s.cy;
    r.cz = t / r.s.cy;
    return r;
}

// nodes past the capacity were never written, so an overflowed batch skips every node stage; the host sees its
// total through prep_total and reruns it with room
inline bool overflowed(device const uint* off, constant PrepParams& p) {
    return off[p.cells] > p.cap;
}

inline uint node_count(device const uint* off, constant PrepParams& p) {
    return overflowed(off, p) ? 0 : off[p.cells];
}

// pieces per cell, scanned into node offsets next
kernel void prep_count(device const uchar* p_block [[buffer(0)]],
                       device const SlabDesc* p_slabs [[buffer(1)]],
                       constant PrepParams& p_prm [[buffer(2)]],
                       device uint* p_off [[buffer(3)]],
                       uint c [[thread_position_in_grid]]) {
    Cell q = locate(p_slabs, p_prm, c);
    ulong m = cell_mask(p_block, p_prm, q.s, q.cz, q.cy, q.cx);
    p_off[c] = m == 0 ? 0 : count_pieces(m, p_prm);
}

// one node per piece: its own parent, keyed by its first voxel's slab-local C-order index, which is its lowest bit
kernel void prep_init(device const uchar* p_block [[buffer(0)]],
                      device const SlabDesc* p_slabs [[buffer(1)]],
                      constant PrepParams& p_prm [[buffer(2)]],
                      device const uint* p_off [[buffer(3)]],
                      device atomic_uint* p_parent [[buffer(4)]],
                      device atomic_uint* p_key [[buffer(5)]],
                      uint c [[thread_position_in_grid]]) {
    uint base = p_off[c];
    uint k = p_off[c + 1] - base;
    if (k == 0 || base + k > p_prm.cap) return;
    Cell q = locate(p_slabs, p_prm, c);
    uint ds = p_prm.ds;
    ulong m = cell_mask(p_block, p_prm, q.s, q.cz, q.cy, q.cx);
    for (uint j = 0; m != 0; j++) {
        ulong seed = m & (~m + 1);
        ulong piece = k == 1 ? m : flood(seed, m, p_prm);
        m &= ~piece;
        uint pos = uint(ctz(seed));
        uint z = q.cz * ds + pos / p_prm.ds2;
        uint y = q.cy * ds + (pos / ds) % ds;
        uint x = q.cx * ds + pos % ds;
        st(p_parent, base + j, base + j);
        st(p_key, base + j, (z * q.s.ys + y) * q.s.xs + x);
    }
}

// joins each piece to the pieces it touches across the cell's +x, +y and +z faces
kernel void prep_union(device const uchar* p_block [[buffer(0)]],
                       device const SlabDesc* p_slabs [[buffer(1)]],
                       constant PrepParams& p_prm [[buffer(2)]],
                       device const uint* p_off [[buffer(3)]],
                       device atomic_uint* p_parent [[buffer(4)]],
                       uint c [[thread_position_in_grid]]) {
    uint base = p_off[c];
    uint k = p_off[c + 1] - base;
    if (k == 0 || overflowed(p_off, p_prm)) return;
    Cell q = locate(p_slabs, p_prm, c);
    uint ds = p_prm.ds;
    ulong m = cell_mask(p_block, p_prm, q.s, q.cz, q.cy, q.cx);
    for (uint dir = 0; dir < 3; dir++) {
        uint nc;
        ulong face;
        uint shift;
        uint ncz = q.cz, ncy = q.cy, ncx = q.cx;
        if (dir == 0) {
            if (q.cx + 1 >= q.s.cx) continue;
            nc = c + 1; face = p_prm.face_xp; shift = ds - 1; ncx++;
        } else if (dir == 1) {
            if (q.cy + 1 >= q.s.cy) continue;
            nc = c + q.s.cx; face = p_prm.face_yp; shift = ds * (ds - 1); ncy++;
        } else {
            if (q.cz + 1 >= q.s.cz) continue;
            nc = c + q.s.cx * q.s.cy; face = p_prm.face_zp; shift = p_prm.ds2 * (ds - 1); ncz++;
        }
        // own face bits, moved onto the neighbour's facing side
        ulong mf = (m & face) >> shift;
        if (mf == 0) continue;
        uint nbase = p_off[nc];
        uint nk = p_off[nc + 1] - nbase;
        if (nk == 0) continue;
        ulong nm = cell_mask(p_block, p_prm, q.s, ncz, ncy, ncx);
        if ((mf & nm) == 0) continue;
        if (k == 1 && nk == 1) {
            uf_union(p_parent, base, nbase);
            continue;
        }
        ulong rem = m;
        for (uint j = 0; rem != 0; j++) {
            ulong a = k == 1 ? rem : flood(rem & (~rem + 1), rem, p_prm);
            rem &= ~a;
            ulong af = (a & face) >> shift;
            if ((af & nm) == 0) continue;
            ulong nrem = nm;
            for (uint jb = 0; nrem != 0; jb++) {
                ulong b = nk == 1 ? nrem : flood(nrem & (~nrem + 1), nrem, p_prm);
                nrem &= ~b;
                if ((af & b) != 0) uf_union(p_parent, base + j, nbase + jb);
            }
        }
    }
}

kernel void prep_compress(constant PrepParams& p_prm [[buffer(0)]],
                          device const uint* p_off [[buffer(1)]],
                          device atomic_uint* p_parent [[buffer(2)]],
                          uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm)) return;
    st(p_parent, n, uf_find(p_parent, n));
}

// each root's key becomes its component's first voxel
kernel void prep_keymin(constant PrepParams& p_prm [[buffer(0)]],
                        device const uint* p_off [[buffer(1)]],
                        device atomic_uint* p_parent [[buffer(2)]],
                        device atomic_uint* p_key [[buffer(3)]],
                        uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm)) return;
    uint r = ld(p_parent, n);
    if (r != n) atomic_fetch_min_explicit(&p_key[r], ld(p_key, n), memory_order_relaxed);
}

inline uint root_bucket(device const SlabDesc* slabs, constant PrepParams& p, device const uint* off, uint n,
                        uint key, thread uint& slab_bucket) {
    SlabDesc s = slabs[slab_by_node(slabs, p.nslabs, off, n)];
    slab_bucket = s.bucket_base;
    return s.bucket_base + (key >> BUCKET_SHIFT);
}

kernel void prep_bcount(device const SlabDesc* p_slabs [[buffer(0)]],
                        constant PrepParams& p_prm [[buffer(1)]],
                        device const uint* p_off [[buffer(2)]],
                        device atomic_uint* p_parent [[buffer(3)]],
                        device atomic_uint* p_key [[buffer(4)]],
                        device atomic_uint* p_bucket [[buffer(5)]],
                        uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm) || ld(p_parent, n) != n) return;
    uint sb;
    uint b = root_bucket(p_slabs, p_prm, p_off, n, ld(p_key, n), sb);
    atomic_fetch_add_explicit(&p_bucket[b], 1, memory_order_relaxed);
}

// after the exclusive scan, each bucket's entry is a bump pointer; once every root has taken a slot it holds the
// bucket's end, which is the next bucket's start
kernel void prep_scatter(device const SlabDesc* p_slabs [[buffer(0)]],
                         constant PrepParams& p_prm [[buffer(1)]],
                         device const uint* p_off [[buffer(2)]],
                         device atomic_uint* p_parent [[buffer(3)]],
                         device atomic_uint* p_key [[buffer(4)]],
                         device atomic_uint* p_bucket [[buffer(5)]],
                         device uint* p_sorted [[buffer(6)]],
                         uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm) || ld(p_parent, n) != n) return;
    uint sb;
    uint key = ld(p_key, n);
    uint b = root_bucket(p_slabs, p_prm, p_off, n, key, sb);
    uint pos = atomic_fetch_add_explicit(&p_bucket[b], 1, memory_order_relaxed);
    p_sorted[pos] = key;
}

// a root's label: roots of earlier buckets of its slab, plus smaller keys in its own bucket, plus one
kernel void prep_rank(device const SlabDesc* p_slabs [[buffer(0)]],
                      constant PrepParams& p_prm [[buffer(1)]],
                      device const uint* p_off [[buffer(2)]],
                      device atomic_uint* p_parent [[buffer(3)]],
                      device atomic_uint* p_key [[buffer(4)]],
                      device const uint* p_bucket [[buffer(5)]],
                      device const uint* p_sorted [[buffer(6)]],
                      uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm) || ld(p_parent, n) != n) return;
    uint sb;
    uint key = ld(p_key, n);
    uint b = root_bucket(p_slabs, p_prm, p_off, n, key, sb);
    uint start = b == 0 ? 0 : p_bucket[b - 1];
    uint end = p_bucket[b];
    uint slab_start = sb == 0 ? 0 : p_bucket[sb - 1];
    uint smaller = 0;
    for (uint i = start; i < end; i++) smaller += p_sorted[i] < key ? 1 : 0;
    st(p_key, n, start - slab_start + smaller + 1);
}

// every node takes its root's label; the parent array is cleared to hold label sizes next
kernel void prep_label(constant PrepParams& p_prm [[buffer(0)]],
                       device const uint* p_off [[buffer(1)]],
                       device atomic_uint* p_parent [[buffer(2)]],
                       device atomic_uint* p_key [[buffer(3)]],
                       uint n [[thread_position_in_grid]]) {
    if (n >= node_count(p_off, p_prm)) return;
    uint r = ld(p_parent, n);
    if (r != n) st(p_key, n, ld(p_key, r));
    st(p_parent, n, 0);
}

inline uint out_cell(device const SlabDesc* slabs, constant PrepParams& p, uint o, thread SlabDesc& s) {
    s = slabs[slab_by_out(slabs, p.nslabs, o)];
    uint local = o - s.out_base;
    uint ox = local % s.px;
    uint t = local / s.px;
    uint oy = t % s.py;
    uint oz = t / s.py;
    return s.cell_base + (oz * s.cy + oy) * s.cx + ox;
}

// the max-pool: a kept cell's label is its pieces' highest label; sizes count pooled voxels per label, stored at
// the slab's first node plus label - 1, since a slab has at least as many nodes as labels
kernel void prep_pool(device const SlabDesc* p_slabs [[buffer(0)]],
                      constant PrepParams& p_prm [[buffer(1)]],
                      device const uint* p_off [[buffer(2)]],
                      device atomic_uint* p_key [[buffer(3)]],
                      device atomic_uint* p_size [[buffer(4)]],
                      device uint* p_out [[buffer(5)]],
                      uint o [[thread_position_in_grid]]) {
    SlabDesc s;
    uint c = out_cell(p_slabs, p_prm, o, s);
    uint lo = p_off[c];
    uint hi = overflowed(p_off, p_prm) ? lo : p_off[c + 1];
    uint best = 0;
    for (uint n = lo; n < hi; n++) best = max(best, ld(p_key, n));
    p_out[o] = best;
    if (best != 0) atomic_fetch_add_explicit(&p_size[p_off[s.cell_base] + best - 1], 1, memory_order_relaxed);
}

// keeps labels with min_count <= size < max_count
kernel void prep_filter(device const SlabDesc* p_slabs [[buffer(0)]],
                        constant PrepParams& p_prm [[buffer(1)]],
                        device const uint* p_off [[buffer(2)]],
                        device atomic_uint* p_size [[buffer(3)]],
                        device uint* p_out [[buffer(4)]],
                        uint o [[thread_position_in_grid]]) {
    uint l = p_out[o];
    if (l == 0) return;
    SlabDesc s;
    out_cell(p_slabs, p_prm, o, s);
    uint size = ld(p_size, p_off[s.cell_base] + l - 1);
    if (size < p_prm.min_count || size >= p_prm.max_count) p_out[o] = 0;
}

// the batch's node count, kept per batch so the host can see an overflowed node capacity after the call's one wait
kernel void prep_total(constant PrepParams& p_prm [[buffer(0)]],
                       device const uint* p_off [[buffer(1)]],
                       device uint* p_totals [[buffer(2)]],
                       constant uint& index [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    p_totals[index] = p_off[p_prm.cells];
}
