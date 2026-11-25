// Shared definitions: the host-mirrored structs, slab lookups, union-find, prefix scan and fill
//
// Buffer parameters are named p_*; every other parameter is a 4-byte uint, which the host checks against reflection

#include <metal_stdlib>
using namespace metal;

// One slab of an internal batch, mirrored by `SlabDesc` in src/metal/prep.rs
struct SlabDesc {
    uint z0, y0, x0;     // origin in the block, full resolution; x0 in bits
    uint zs, ys, xs;     // extent, full resolution
    uint cz, cy, cx;     // cell grid, ceil(extent / ds): partial cells hold the cropped tail
    uint pz, py, px;     // pooled grid, floor(extent / ds)
    uint cell_base;      // first cell of this slab in the batch
    uint out_base;       // first pooled voxel of this slab in the batch
    uint bucket_base;    // first rank bucket of this slab in the batch
    uint pad;
};

// Per-batch constants, mirrored by `PrepParams` in src/metal/prep.rs. Bit p of a cell mask is voxel
// (dz, dy, dx) with p = (dz * ds + dy) * ds + dx, so bit order is C order inside the cell
struct PrepParams {
    ulong can_xp, can_xm;   // bits that may step +x (dx != ds - 1) and -x (dx != 0)
    ulong can_yp, can_ym;   // the same for y
    ulong face_xp, face_yp, face_zp;  // bits on the + face of each axis
    uint ds, ds2;
    uint nslabs, cells;
    uint outs, buckets;
    uint cap, row_bytes;
    uint block_rows, min_count;
    uint max_count, pad;
};

// rank buckets hold 2^9 consecutive slab-local linear indices
constant uint BUCKET_SHIFT = 9;

// the last slab whose cell_base is at most c
inline uint slab_by_cell(device const SlabDesc* s, uint n, uint c) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (s[mid].cell_base <= c) lo = mid; else hi = mid;
    }
    return lo;
}

inline uint slab_by_out(device const SlabDesc* s, uint n, uint o) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (s[mid].out_base <= o) lo = mid; else hi = mid;
    }
    return lo;
}

// the last slab whose first node is at most v; slabs without nodes share the next slab's base, and the last of
// them is the one that holds v
inline uint slab_by_node(device const SlabDesc* s, uint n, device const uint* off, uint v) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (off[s[mid].cell_base] <= v) lo = mid; else hi = mid;
    }
    return lo;
}

inline uint ld(device atomic_uint* a, uint i) {
    return atomic_load_explicit(&a[i], memory_order_relaxed);
}

inline void st(device atomic_uint* a, uint i, uint v) {
    atomic_store_explicit(&a[i], v, memory_order_relaxed);
}

// union-find whose roots are the smallest index of their set: links only ever lower a parent, so the result does
// not depend on the order of unions (Playne and Hawick, 2018)
inline uint uf_find(device atomic_uint* par, uint x) {
    uint p = ld(par, x);
    while (p != x) {
        x = p;
        p = ld(par, x);
    }
    return x;
}

inline void uf_union(device atomic_uint* par, uint a, uint b) {
    for (;;) {
        a = uf_find(par, a);
        b = uf_find(par, b);
        if (a == b) return;
        if (a < b) { uint t = a; a = b; b = t; }
        // a > b: hang a under b; if a gained a parent meanwhile, the min may have replaced it, so join that too
        uint old = atomic_fetch_min_explicit(&par[a], b, memory_order_relaxed);
        if (old == a) return;
        a = old;
    }
}

// ---- prefix scan: 256 threads, 4 items each, 1024 elements per threadgroup ----

#define SCAN_THREADS 256
#define SCAN_ITEMS 4
#define SCAN_BLOCK (SCAN_THREADS * SCAN_ITEMS)

// the exclusive prefix of v over the threadgroup, and the threadgroup's total
inline uint tg_exclusive(uint v, threadgroup uint* sh, uint sid, uint lane, thread uint& total) {
    uint incl = simd_prefix_inclusive_sum(v);
    if (lane == 31) sh[sid] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        uint s = lane < 8 ? sh[lane] : 0;
        uint e = simd_prefix_exclusive_sum(s);
        if (lane < 8) sh[8 + lane] = e;
        if (lane == 7) sh[16] = e + s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = sh[16];
    return sh[8 + sid] + incl - v;
}

// the sum of each 1024-element block of p_data[0, n)
kernel void scan_reduce(device const uint* p_data [[buffer(0)]],
                        device uint* p_sums [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint gid [[threadgroup_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sh[17];
    uint base = gid * SCAN_BLOCK + lid * SCAN_ITEMS;
    uint s = 0;
    for (uint i = 0; i < SCAN_ITEMS; i++) {
        uint k = base + i;
        if (k < n) s += p_data[k];
    }
    uint total;
    tg_exclusive(s, sh, sid, lane, total);
    if (lid == 0) p_sums[gid] = total;
}

// p_data[0, n) becomes its exclusive prefix sum, offset by each block's p_sums entry (already exclusive); with
// with_total, p_data[n] gets the grand total
kernel void scan_apply(device uint* p_data [[buffer(0)]],
                       device const uint* p_sums [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       constant uint& with_total [[buffer(3)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint sid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sh[17];
    uint base = gid * SCAN_BLOCK + lid * SCAN_ITEMS;
    uint v[SCAN_ITEMS];
    uint s = 0;
    for (uint i = 0; i < SCAN_ITEMS; i++) {
        uint k = base + i;
        v[i] = k < n ? p_data[k] : 0;
        s += v[i];
    }
    uint total;
    uint run = tg_exclusive(s, sh, sid, lane, total) + p_sums[gid];
    for (uint i = 0; i < SCAN_ITEMS; i++) {
        uint k = base + i;
        if (k < n) p_data[k] = run;
        run += v[i];
        if (with_total != 0 && k + 1 == n) p_data[n] = run;
    }
}

kernel void fill_u32(device uint* p_data [[buffer(0)]],
                     constant uint& value [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    p_data[i] = value;
}

// a plain copy, to measure the bandwidth floor
kernel void copy_u4(device const uint4* p_src [[buffer(0)]],
                    device uint4* p_dst [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    p_dst[i] = p_src[i];
}
