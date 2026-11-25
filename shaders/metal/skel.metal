// Part B on the GPU: 26-connected pieces, the distance transforms, the border targets and every piece's path loop,
// for all slabs of an internal batch at once
//
// Kernel order per batch (src/metal/skel.rs drives it):
//   sk_ccl_init, sk_ccl_union, sk_ccl_flag, scan, sk_ccl_pid                       -> host wait 1 (piece table)
//   sk_scatter, sk_edt_x, sk_edt_y, sk_edt_z, sk_maxpen, sk_face_*                 -> host wait 2 (border targets)
//   sk_run, repeated until every piece is done                                     -> host wait per dispatch
//
// Voxel indices are batch-global C-order indices; slabs are contiguous, so within a slab they order like the slab's
// own C order. A piece's voxel list is in arbitrary (scatter) order: every tie the spec breaks by "lowest index" is
// broken here by the global index, never by the list position, so the output does not depend on the list order
//
// Float rules (spec part B §2.2, §2.5, §3.2) are kept exactly: radii use a correctly rounded square root, PDRF
// divisions are correctly rounded, and products that feed an addition pass through `opq` so the compiler cannot fuse
// them into a multiply-add

constant uint NONE = 0xffffffffu;
constant uint FAR = 0xffffffffu;
constant uint UNREACHED = 0xffffffffu;

// One slab of a batch, mirrored by `SkSlab` in src/metal/skel.rs
struct SkSlab {
    uint z, y, x;
    uint base;  // first voxel in the batch
};

inline uint sk_slab_of(device const SkSlab* s, uint n, uint v) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (s[mid].base <= v) lo = mid; else hi = mid;
    }
    return lo;
}

// ---- pieces: 26-connected components of equal label ----

kernel void sk_ccl_init(device uint* p_par [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    p_par[i] = i;
}

kernel void sk_ccl_union(device const uint* p_lab [[buffer(0)]],
                         device const SkSlab* p_slabs [[buffer(1)]],
                         device atomic_uint* p_par [[buffer(2)]],
                         constant uint& nslabs [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    uint l = p_lab[i];
    if (l == 0) return;
    SkSlab s = p_slabs[sk_slab_of(p_slabs, nslabs, i)];
    uint r = i - s.base;
    uint xx = r % s.x, yy = (r / s.x) % s.y, zz = r / (s.x * s.y);
    // the 13 neighbours that precede this voxel in C order
    for (int dz = -1; dz <= 0; dz++) {
        if (int(zz) + dz < 0) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = int(yy) + dy;
            if (ny < 0 || ny >= int(s.y)) continue;
            if (dz == 0 && dy > 0) break;
            for (int dx = -1; dx <= 1; dx++) {
                if (dz == 0 && dy == 0 && dx >= 0) break;
                int nx = int(xx) + dx;
                if (nx < 0 || nx >= int(s.x)) continue;
                uint j = s.base + (uint(int(zz) + dz) * s.y + uint(ny)) * s.x + uint(nx);
                if (p_lab[j] == l) uf_union(p_par, i, j);
            }
        }
    }
}

// compresses every voxel to its root and flags roots of labelled voxels, for the scan that numbers pieces
kernel void sk_ccl_flag(device const uint* p_lab [[buffer(0)]],
                        device atomic_uint* p_par [[buffer(1)]],
                        device uint* p_flag [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (p_lab[i] == 0) {
        p_flag[i] = 0;
        return;
    }
    uint r = uf_find(p_par, i);
    st(p_par, i, r);
    p_flag[i] = r == i ? 1 : 0;
}

// piece ids (numbered by first voxel in C order across the batch), piece sizes, labels, first voxels and the
// smallest slab-local Fortran-order index of each piece
kernel void sk_ccl_pid(device const uint* p_lab [[buffer(0)]],
                       device const uint* p_par [[buffer(1)]],
                       device const uint* p_num [[buffer(2)]],
                       device const SkSlab* p_slabs [[buffer(3)]],
                       device uint* p_pid [[buffer(4)]],
                       device atomic_uint* p_size [[buffer(5)]],
                       device atomic_uint* p_ffirst [[buffer(6)]],
                       device uint* p_plabel [[buffer(7)]],
                       device uint* p_pfirst [[buffer(8)]],
                       constant uint& nslabs [[buffer(9)]],
                       uint i [[thread_position_in_grid]]) {
    uint l = p_lab[i];
    if (l == 0) {
        p_pid[i] = NONE;
        return;
    }
    uint r = p_par[i];
    uint p = p_num[r];
    p_pid[i] = p;
    atomic_fetch_add_explicit(&p_size[p], 1, memory_order_relaxed);
    SkSlab s = p_slabs[sk_slab_of(p_slabs, nslabs, i)];
    uint q = i - s.base;
    uint xx = q % s.x, yy = (q / s.x) % s.y, zz = q / (s.x * s.y);
    atomic_fetch_min_explicit(&p_ffirst[p], zz + s.z * (yy + s.y * xx), memory_order_relaxed);
    if (r == i) {
        p_plabel[p] = l;
        p_pfirst[p] = i;
    }
}

// ---- per-piece workspace ----

// One kept piece, mirrored by `SkPiece` in src/metal/skel.rs. Its workspace is WS_ARRAYS arrays of m words from `ws`
struct SkPiece {
    uint pid;
    uint slab;
    uint m;
    uint ws;
    uint valid;     // first word of its valid bits
    uint mand_off;  // its border targets (voxel indices, tracing order) in the target list
    uint mand_n;
    uint first;     // first voxel in C order
    uint ffirst;    // first voxel in Fortran order
    uint host;      // traced on host threads: the workspace holds the voxel list only
    uint pad1, pad2;
};

constant uint WS_VOX = 0;    // voxel index of each list entry
constant uint WS_COST = 1;   // geodesic distance, then path-search cost (float32 bits)
constant uint WS_HCOST = 2;  // heap: cost of each slot
constant uint WS_HLOC = 3;   // heap: list entry of each slot
constant uint WS_POS = 4;    // heap slot of each list entry, or NONE
constant uint WS_P = 5;      // PDRF (float32)
constant uint WS_VID = 6;    // vertex number of each list entry of S, or NONE
constant uint WS_ORDER = 7;  // ordinary target order: list entries by (D desc, voxel index asc)
constant uint WS_TOUCH = 8;  // list entries a path search reached
constant uint WS_VERT = 9;   // vertices (list entries), in vertex order
constant uint WS_NEXT = 10;  // for each vertex but the root, the vertex its edge leads to
constant uint WS_ARRAYS = 11;

// each kept voxel joins its piece's list; its list position is its `local` index
kernel void sk_scatter(device const uint* p_pid [[buffer(0)]],
                       device const uint* p_kidx [[buffer(1)]],
                       device const SkPiece* p_pieces [[buffer(2)]],
                       device atomic_uint* p_fill [[buffer(3)]],
                       device uint* p_ws [[buffer(4)]],
                       device uint* p_local [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    uint p = p_pid[i];
    if (p == NONE) return;
    uint k = p_kidx[p];
    if (k == NONE) {
        p_local[i] = NONE;
        return;
    }
    uint slot = atomic_fetch_add_explicit(&p_fill[k], 1, memory_order_relaxed);
    SkPiece pc = p_pieces[k];
    device uint* w = p_ws + pc.ws;
    w[WS_VOX * pc.m + slot] = i;
    p_local[i] = slot;
    if (pc.host != 0) return;
    w[WS_COST * pc.m + slot] = UNREACHED;
    w[WS_POS * pc.m + slot] = NONE;
    w[WS_VID * pc.m + slot] = NONE;
}

// ---- exact squared distance transforms over views ----
//
// A view is a 3D box of the batch's piece ids (a slab, or one face layer of a slab as a 1 x h x w box) with its
// own dense output. Pass x runs one thread per line with two sweeps; passes y and z run one thread per voxel and
// scan outward until no candidate can beat the best, stopping at the run's ends, which are distance-0 candidates

// Mirrored by `SkView` in src/metal/skel.rs
struct SkView {
    uint id_base;
    uint s0, s1, s2;    // id strides of the view's axes, in voxels
    uint n0, n1, n2;
    uint out;           // first element of the view's dense output
    uint border;        // bit k: the positions outside the view along axis k are boundary
    uint vox_off;       // first voxel of the view in the batch's view voxels
    uint line_off;      // first x line of the view
    uint pad;
};

inline uint sk_view_by(device const SkView* v, uint n, uint x, bool lines) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        uint b = lines ? v[mid].line_off : v[mid].vox_off;
        if (b <= x) lo = mid; else hi = mid;
    }
    return lo;
}

kernel void sk_edt_x(device const SkView* p_views [[buffer(0)]],
                     device const uint* p_ids [[buffer(1)]],
                     device uint* p_f [[buffer(2)]],
                     constant uint& nviews [[buffer(3)]],
                     uint t [[thread_position_in_grid]]) {
    SkView v = p_views[sk_view_by(p_views, nviews, t, true)];
    uint line = t - v.line_off;
    uint a = line / v.n1, b = line % v.n1;
    uint ib = v.id_base + a * v.s0 + b * v.s1;
    uint ob = v.out + line * v.n2;
    uint n = v.n2;
    bool border = (v.border & 4) != 0;
    // forward: distance to the last boundary at or before each position
    uint last = border ? 0 : NONE;  // position + 1 of the last boundary, 0 being the border at -1
    uint prev = NONE;
    for (uint k = 0; k < n; k++) {
        uint id = p_ids[ib + k * v.s2];
        if (k > 0 && id != prev) last = k;  // voxel k - 1 is a boundary for this run
        prev = id;
        p_f[ob + k] = last == NONE ? NONE : k + 1 - last;
    }
    // backward
    uint next = border ? n : NONE;
    prev = NONE;
    for (uint k = n; k-- > 0;) {
        uint id = p_ids[ib + k * v.s2];
        if (k + 1 < n && id != prev) next = k + 1;
        prev = id;
        uint d = p_f[ob + k];
        uint e = next == NONE ? NONE : next - k;
        d = min(d, e);
        p_f[ob + k] = d == NONE ? FAR : d * d;
    }
}

// one pass along axis `axis` (1 or 0 of the view), from `p_in` into `p_out`
inline void sk_edt_scan(SkView v, uint axis, device const uint* ids, device const uint* in, device uint* out,
                        uint local) {
    uint c2 = local % v.n2, c1 = (local / v.n2) % v.n1, c0 = local / (v.n2 * v.n1);
    uint idx = v.id_base + c0 * v.s0 + c1 * v.s1 + c2 * v.s2;
    uint id = ids[idx];
    if (id == NONE) {
        out[v.out + local] = FAR;
        return;
    }
    uint n = axis == 1 ? v.n1 : v.n0;
    uint k = axis == 1 ? c1 : c0;
    uint ids_step = axis == 1 ? v.s1 : v.s0;
    uint out_step = axis == 1 ? v.n2 : v.n2 * v.n1;
    bool border = (v.border & (axis == 1 ? 2u : 1u)) != 0;
    uint best = in[v.out + local];
    bool lo_open = true, hi_open = true;
    for (uint d = 1; lo_open || hi_open; d++) {
        uint dd = d * d;
        if (dd >= best) break;
        if (lo_open) {
            if (d > k) {
                lo_open = false;
                if (border) best = min(best, dd);
            } else if (ids[idx - d * ids_step] != id) {
                lo_open = false;
                best = min(best, dd);
            } else {
                uint f = in[v.out + local - d * out_step];
                if (f != FAR) best = min(best, dd + f);
            }
        }
        if (hi_open) {
            if (k + d >= n) {
                hi_open = false;
                if (border) best = min(best, dd);
            } else if (ids[idx + d * ids_step] != id) {
                hi_open = false;
                best = min(best, dd);
            } else {
                uint f = in[v.out + local + d * out_step];
                if (f != FAR) best = min(best, dd + f);
            }
        }
    }
    out[v.out + local] = best;
}

kernel void sk_edt_y(device const SkView* p_views [[buffer(0)]],
                     device const uint* p_ids [[buffer(1)]],
                     device const uint* p_in [[buffer(2)]],
                     device uint* p_out [[buffer(3)]],
                     constant uint& nviews [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    SkView v = p_views[sk_view_by(p_views, nviews, t, false)];
    sk_edt_scan(v, 1, p_ids, p_in, p_out, t - v.vox_off);
}

kernel void sk_edt_z(device const SkView* p_views [[buffer(0)]],
                     device const uint* p_ids [[buffer(1)]],
                     device const uint* p_in [[buffer(2)]],
                     device uint* p_out [[buffer(3)]],
                     constant uint& nviews [[buffer(4)]],
                     uint t [[thread_position_in_grid]]) {
    SkView v = p_views[sk_view_by(p_views, nviews, t, false)];
    sk_edt_scan(v, 0, p_ids, p_in, p_out, t - v.vox_off);
}

// each kept piece's largest squared penalty distance, for max B
kernel void sk_maxpen(device const uint* p_pid [[buffer(0)]],
                      device const uint* p_kidx [[buffer(1)]],
                      device const uint* p_pen [[buffer(2)]],
                      device atomic_uint* p_max [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    uint p = p_pid[i];
    if (p == NONE) return;
    uint k = p_kidx[p];
    if (k == NONE) return;
    atomic_fetch_max_explicit(&p_max[k], p_pen[i], memory_order_relaxed);
}

// ---- border targets (§2.4) ----
//
// A layer is one face of one slab, laid out densely as h x w over its two in-plane axes in array order, so a
// layer-local index orders like the voxel index. Regions are the 8- (or 4-) connected sets of one kept piece in the
// layer, found by min-root union-find over layer-local indices; their statistics accumulate at the root

// Mirrored by `SkLayer` in src/metal/skel.rs
struct SkLayer {
    uint off;      // first layer voxel in the batch's layer voxels
    uint h, w;
    uint vbase;    // voxel index of layer position (0, 0)
    uint su, sv;   // voxel strides of the two in-plane axes
    uint order;    // the layer's tracing order across the batch
    uint pad;
};

inline uint sk_layer_of(device const SkLayer* l, uint n, uint t) {
    uint lo = 0, hi = n;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (l[mid].off <= t) lo = mid; else hi = mid;
    }
    return lo;
}

// the piece of a layer voxel, if it is kept
inline uint sk_layer_piece(device const uint* pid, device const uint* kidx, SkLayer l, uint u, uint v) {
    uint p = pid[l.vbase + u * l.su + v * l.sv];
    return p == NONE || kidx[p] == NONE ? NONE : p;
}

// the face pick's key (src/cpu/skel.rs `FaceKey`), compared word by word, one word per pass: the centroid's two
// words, the plane centre, corner and edge, then the voxel index (complemented to prefer the highest)
constant uint PICK_WORDS = 6;
constant uint PICK_README = 0, PICK_CENTROID = 1, PICK_CENTROID_HIGH = 2, PICK_LOWEST = 3;

kernel void sk_face_init(device const SkLayer* p_layers [[buffer(0)]],
                         device const uint* p_pid [[buffer(1)]],
                         device const uint* p_kidx [[buffer(2)]],
                         device uint* p_par [[buffer(3)]],
                         device uint* p_stats [[buffer(4)]],
                         constant uint& nlayers [[buffer(5)]],
                         constant uint& total [[buffer(6)]],
                         uint t [[thread_position_in_grid]]) {
    SkLayer l = p_layers[sk_layer_of(p_layers, nlayers, t)];
    uint li = t - l.off;
    uint p = sk_layer_piece(p_pid, p_kidx, l, li / l.w, li % l.w);
    p_par[t] = p == NONE ? NONE : t;
    // n, su, sv, maxd start at 0; the pick key words at NONE
    for (uint k = 0; k < 4; k++) p_stats[k * total + t] = 0;
    for (uint k = 4; k < 4 + PICK_WORDS; k++) p_stats[k * total + t] = NONE;
}

kernel void sk_face_union(device const SkLayer* p_layers [[buffer(0)]],
                          device const uint* p_pid [[buffer(1)]],
                          device const uint* p_kidx [[buffer(2)]],
                          device atomic_uint* p_par [[buffer(3)]],
                          constant uint& nlayers [[buffer(4)]],
                          constant uint& eight [[buffer(5)]],
                          uint t [[thread_position_in_grid]]) {
    if (ld(p_par, t) == NONE) return;
    SkLayer l = p_layers[sk_layer_of(p_layers, nlayers, t)];
    uint li = t - l.off;
    int u = int(li / l.w), v = int(li % l.w);
    uint p = p_pid[l.vbase + uint(u) * l.su + uint(v) * l.sv];
    const int du8[4] = {-1, -1, -1, 0};
    const int dv8[4] = {-1, 0, 1, -1};
    const int du4[2] = {-1, 0};
    const int dv4[2] = {0, -1};
    uint cnt = eight != 0 ? 4 : 2;
    for (uint k = 0; k < cnt; k++) {
        int nu = u + (eight != 0 ? du8[k] : du4[k]);
        int nv = v + (eight != 0 ? dv8[k] : dv4[k]);
        if (nu < 0 || nv < 0 || nu >= int(l.h) || nv >= int(l.w)) continue;
        if (p_pid[l.vbase + uint(nu) * l.su + uint(nv) * l.sv] != p) continue;
        uf_union(p_par, t, l.off + uint(nu) * l.w + uint(nv));
    }
}

kernel void sk_face_stats(device const SkLayer* p_layers [[buffer(0)]],
                          device atomic_uint* p_par [[buffer(1)]],
                          device const uint* p_dist [[buffer(2)]],
                          device atomic_uint* p_stats [[buffer(3)]],
                          constant uint& nlayers [[buffer(4)]],
                          constant uint& total [[buffer(5)]],
                          uint t [[thread_position_in_grid]]) {
    if (ld(p_par, t) == NONE) return;
    SkLayer l = p_layers[sk_layer_of(p_layers, nlayers, t)];
    uint li = t - l.off;
    uint r = uf_find(p_par, t);
    st(p_par, t, r);
    atomic_fetch_add_explicit(&p_stats[r], 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&p_stats[total + r], li / l.w, memory_order_relaxed);
    atomic_fetch_add_explicit(&p_stats[2 * total + r], li % l.w, memory_order_relaxed);
    atomic_fetch_max_explicit(&p_stats[3 * total + r], p_dist[t], memory_order_relaxed);
}

// the squared distance to the region's centroid, times n^2 so it stays an integer; regions are at most one piece,
// and pieces are bounded by the size filter, so it fits 64 bits
inline ulong sk_centroid_key(uint n, uint su, uint sv, uint u, uint v) {
    long du = long(n) * long(u) - long(su);
    long dv = long(n) * long(v) - long(sv);
    return ulong(du * du) + ulong(dv * dv);
}

// word k of a voxel's pick key; a word the mode does not use is 0 for every voxel. Every term fits 32 bits: plane axes
// are at most 32,768 long (part A's per-axis limit)
inline uint sk_pick_word(uint mode, uint k, ulong centroid, SkLayer l, uint u, uint v) {
    uint h = l.h - 1, w = l.w - 1;
    bool readme = mode == PICK_README;
    switch (k) {
        case 0: return mode == PICK_LOWEST ? 0 : uint(centroid >> 32);
        case 1: return mode == PICK_LOWEST ? 0 : uint(centroid);
        case 2: {
            int du = 2 * int(u) - int(h), dv = 2 * int(v) - int(w);
            return readme ? uint(du * du) + uint(dv * dv) : 0;
        }
        case 3: {
            uint cu = min(u, h - u), cv = min(v, w - v);
            return readme ? cu * cu + cv * cv : 0;
        }
        case 4: return readme ? min(min(u, h - u), min(v, w - v)) : 0;
        default: {
            uint idx = l.vbase + u * l.su + v * l.sv;
            return mode == PICK_CENTROID || mode == PICK_LOWEST ? idx : ~idx;
        }
    }
}

// pass k: the smallest word k over the region's distance maxima that match the smallest words 0 to k - 1
kernel void sk_face_pick(device const SkLayer* p_layers [[buffer(0)]],
                         device const uint* p_par [[buffer(1)]],
                         device const uint* p_dist [[buffer(2)]],
                         device atomic_uint* p_stats [[buffer(3)]],
                         constant uint& nlayers [[buffer(4)]],
                         constant uint& total [[buffer(5)]],
                         constant uint& pass [[buffer(6)]],
                         constant uint& mode [[buffer(7)]],
                         uint t [[thread_position_in_grid]]) {
    uint r = p_par[t];
    if (r == NONE) return;
    if (p_dist[t] != atomic_load_explicit(&p_stats[3 * total + r], memory_order_relaxed)) return;
    SkLayer l = p_layers[sk_layer_of(p_layers, nlayers, t)];
    uint li = t - l.off;
    uint u = li / l.w, v = li % l.w;
    uint n = atomic_load_explicit(&p_stats[r], memory_order_relaxed);
    uint su = atomic_load_explicit(&p_stats[total + r], memory_order_relaxed);
    uint sv = atomic_load_explicit(&p_stats[2 * total + r], memory_order_relaxed);
    ulong centroid = sk_centroid_key(n, su, sv, u, v);
    for (uint k = 0; k < pass; k++) {
        uint best = atomic_load_explicit(&p_stats[(4 + k) * total + r], memory_order_relaxed);
        if (sk_pick_word(mode, k, centroid, l, u, v) != best) return;
    }
    atomic_fetch_min_explicit(&p_stats[(4 + pass) * total + r], sk_pick_word(mode, pass, centroid, l, u, v),
                              memory_order_relaxed);
}

// one record per region: (kept piece, layer order, root's layer position, target voxel)
kernel void sk_face_emit(device const SkLayer* p_layers [[buffer(0)]],
                         device const uint* p_par [[buffer(1)]],
                         device const uint* p_pid [[buffer(2)]],
                         device const uint* p_kidx [[buffer(3)]],
                         device const uint* p_stats [[buffer(4)]],
                         device atomic_uint* p_count [[buffer(5)]],
                         device uint4* p_out [[buffer(6)]],
                         constant uint& nlayers [[buffer(7)]],
                         constant uint& total [[buffer(8)]],
                         constant uint& mode [[buffer(9)]],
                         uint t [[thread_position_in_grid]]) {
    if (p_par[t] != t) return;
    SkLayer l = p_layers[sk_layer_of(p_layers, nlayers, t)];
    uint li = t - l.off;
    uint p = p_pid[l.vbase + (li / l.w) * l.su + (li % l.w) * l.sv];
    uint slot = atomic_fetch_add_explicit(p_count, 1, memory_order_relaxed);
    uint idx = p_stats[(3 + PICK_WORDS) * total + t];
    if (mode == PICK_README || mode == PICK_CENTROID_HIGH) idx = ~idx;
    p_out[slot] = uint4(p_kidx[p], l.order, li, idx);
}

// ---- float rules ----

// x, hidden from the optimizer behind a runtime zero, so a product never fuses into the addition that follows
inline float opq(float x, uint zero) {
    return as_type<float>(as_type<uint>(x) ^ zero);
}

// the sign of a^2 * 2^k - n for k < 0
inline int sk_cmp_sq(ulong a, int k, uint n) {
    uint sh = uint(-k);
    ulong a2 = a * a;
    if (sh >= 63 || (ulong(n) >> (63 - sh)) != 0) return -1;
    ulong r = ulong(n) << sh;
    return a2 < r ? -1 : (a2 > r ? 1 : 0);
}

// sqrt(n) correctly rounded to float32: the float s with the midpoints to both neighbours on either side of sqrt(n)
inline float rn_sqrt(uint n) {
    if (n == 0) return 0.0f;
    float s = sqrt(float(n));
    for (uint it = 0; it < 8; it++) {
        uint b = as_type<uint>(s);
        ulong m = ulong((b & 0x7fffffu) | 0x800000u);
        int e = int(b >> 23) - 150;
        // the midpoint above is (2m + 1) 2^(e - 1)
        if (sk_cmp_sq(2 * m + 1, 2 * e - 2, n) < 0) {
            s = as_type<float>(b + 1);
            continue;
        }
        // the midpoint below; a power of two has a finer spacing below
        bool pow2 = m == 0x800000u;
        ulong md = pow2 ? 4 * m - 1 : 2 * m - 1;
        int ed = pow2 ? e - 2 : e - 1;
        if (sk_cmp_sq(md, 2 * ed, n) > 0) {
            s = as_type<float>(b - 1);
            continue;
        }
        break;
    }
    return s;
}

// x / y correctly rounded, for positive finite x and y: the float whose exact residual x - q y is smallest
inline float div_rn(float x, float y) {
    if (x == 0.0f) return 0.0f;
    float r = 1.0f / y;
    float q = x * r;
    q = fma(fma(-q, y, x), r, q);
    for (uint it = 0; it < 4; it++) {
        uint b = as_type<uint>(q);
        float e0 = fabs(fma(-q, y, x));
        float up = as_type<float>(b + 1);
        float dn = as_type<float>(b - 1);
        float eu = fabs(fma(-up, y, x));
        float ed = fabs(fma(-dn, y, x));
        if (eu < e0) { q = up; continue; }
        if (ed < e0) { q = dn; continue; }
        break;
    }
    return q;
}

// self-test kernels for the float rules, compared with the CPU's in a GPU test
kernel void sk_test_sqrt(device const uint* p_n [[buffer(0)]], device uint* p_out [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    p_out[i] = as_type<uint>(rn_sqrt(p_n[i]));
}

kernel void sk_test_div(device const float* p_x [[buffer(0)]], device const float* p_y [[buffer(1)]],
                        device uint* p_out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    p_out[i] = as_type<uint>(div_rn(p_x[i], p_y[i]));
}

inline uint step_rule(uint cost, float w) {
    uint sum = as_type<uint>(as_type<float>(cost) + w);
    return max(sum, cost + 1);
}

// ---- the path loop: one lane per piece ----

// Per-call constants, mirrored by `SkParams` in src/metal/skel.rs
struct SkParams {
    float a, inv_a, scale, constant_;
    float len[4];
    uint flags;
    uint budget;
    uint zero;  // always 0; see opq
    uint nslabs;
    uint inv_field;  // field offsets in the field buffer: invalidation distances and penalty distances
    uint pen_field;
    uint dbf_field;
    uint pad;
};

constant uint F_COUNT_TARGET = 1;
constant uint F_MAND_FIRST = 4;
constant uint F_DAF = 8;
constant uint F_ROOT_SHIFT = 4;  // 3 bits: RootStart in src/skel.rs order
constant uint F_UNSQUARED = 128;
// test builds' B7 field dump: stop after the fields, keeping D in WS_COST and DBF and B radii in the heap arrays
constant uint F_FIELDS_STOP = 256;
constant uint ROOT_C = 0, ROOT_F = 1, ROOT_DBF = 2, ROOT_FACE = 3, ROOT_FACE_LAST = 4, ROOT_FACE_FAR = 5;

// Mirrored by `SkState` in src/metal/skel.rs
struct SkState {
    uint phase;
    uint i;         // cursor of the current phase
    uint hn;        // heap size
    uint nt;        // touched entries
    uint src;       // geodesic source, or the path search's target
    uint best;      // geodesic 0: the farthest entry so far
    uint bestc;
    uint root;
    uint geo;       // 0: the start geodesic, 1: the root's (D)
    uint maxd;      // max D, float32 bits
    uint nverts;
    uint ord;       // next ordinary target in ORDER
    uint mand;      // next border target
    uint len;       // new vertices of the current path
    uint join;
    uint ordinary;  // ordinary traces
    uint steps;     // work units spent, all dispatches
    uint runs;      // dispatches that ran this piece
};

constant uint PH_START = 0;
constant uint PH_DBFMAX = 1;
constant uint PH_GEO = 2;
constant uint PH_RESET = 3;
constant uint PH_ORDER_REV = 4;
constant uint PH_ORDER_TIES = 5;
constant uint PH_FIELD = 6;
constant uint PH_ROOT = 7;
constant uint PH_PICK = 8;
constant uint PH_SEARCH = 9;
constant uint PH_WALK = 10;
constant uint PH_CLEAN = 11;
constant uint PH_INVAL = 12;
constant uint PH_DONE = 13;
// an invariant broke (no predecessor, or a search ran out of voxels); the host reports it
constant uint PH_FAIL = 14;

struct SkCtx {
    device uint* w;
    device uint* valid;
    device const uint* pid;
    device const uint* local;
    device const uint* fields;
    device const uint* mand;
    uint m;
    uint me;  // the piece's id
    uint base, Z, Y, X;
    SkParams prm;
};

inline device uint* arr(thread SkCtx& c, uint k) { return c.w + k * c.m; }

inline bool sk_valid(thread SkCtx& c, uint i) { return (c.valid[i >> 5] >> (i & 31)) & 1; }

// heap order: cost, then voxel index
inline bool sk_less(thread SkCtx& c, uint ca, uint la, uint cb, uint lb) {
    if (ca != cb) return ca < cb;
    return arr(c, WS_VOX)[la] < arr(c, WS_VOX)[lb];
}

inline void sk_sift_up(thread SkCtx& c, uint pos, uint cost, uint li) {
    device uint* hc = arr(c, WS_HCOST);
    device uint* hl = arr(c, WS_HLOC);
    device uint* hp = arr(c, WS_POS);
    while (pos > 0) {
        uint parent = (pos - 1) / 2;
        uint pc = hc[parent], pl = hl[parent];
        if (!sk_less(c, cost, li, pc, pl)) break;
        hc[pos] = pc;
        hl[pos] = pl;
        hp[pl] = pos;
        pos = parent;
    }
    hc[pos] = cost;
    hl[pos] = li;
    hp[li] = pos;
}

// lowers (or inserts) entry li to cost
inline void sk_heap_set(thread SkCtx& c, thread uint& hn, uint li, uint cost) {
    uint pos = arr(c, WS_POS)[li];
    if (pos == NONE) pos = hn++;
    sk_sift_up(c, pos, cost, li);
}

inline uint sk_heap_pop(thread SkCtx& c, thread uint& hn) {
    device uint* hc = arr(c, WS_HCOST);
    device uint* hl = arr(c, WS_HLOC);
    device uint* hp = arr(c, WS_POS);
    uint top = hl[0];
    hp[top] = NONE;
    hn--;
    if (hn == 0) return top;
    uint cost = hc[hn], li = hl[hn];
    uint pos = 0;
    for (;;) {
        uint l = 2 * pos + 1;
        if (l >= hn) break;
        uint r = l + 1;
        uint cc = hc[l], cl = hl[l];
        if (r < hn && sk_less(c, hc[r], hl[r], cc, cl)) {
            cc = hc[r];
            cl = hl[r];
            l = r;
        }
        if (!sk_less(c, cc, cl, cost, li)) break;
        hc[pos] = cc;
        hl[pos] = cl;
        hp[cl] = pos;
        pos = l;
    }
    hc[pos] = cost;
    hl[pos] = li;
    hp[li] = pos;
    return top;
}

inline void sk_coords(thread SkCtx& c, uint g, thread int& z, thread int& y, thread int& x) {
    uint r = g - c.base;
    x = int(r % c.X);
    y = int((r / c.X) % c.Y);
    z = int(r / (c.X * c.Y));
}

inline float sk_radius(thread SkCtx& c, uint n) {
    return c.prm.a * rn_sqrt(n);
}

// clears the valid bit of every piece voxel strictly inside the invalidation ball around entry v (src/skel.rs `Ball`):
// r = fma(scale, dbf, const), and a voxel at index offset d is inside when a^2 * |d|^2 < r * r; returns the voxels of
// the bounding cube visited
inline uint sk_invalidate(thread SkCtx& c, uint v) {
    uint g = arr(c, WS_VOX)[v];
    float r = metal::fma(c.prm.scale, sk_radius(c, c.fields[c.prm.inv_field + g]), c.prm.constant_);
    // inv_a and a * a are powers of two, so both products are exact
    int K = int(floor(r * c.prm.inv_a));
    float rr = r * r;
    float a2 = c.prm.a * c.prm.a;
    int z, y, x;
    sk_coords(c, g, z, y, x);
    int z0 = max(z - K, 0), z1 = min(z + K + 1, int(c.Z));
    int y0 = max(y - K, 0), y1 = min(y + K + 1, int(c.Y));
    int x0 = max(x - K, 0), x1 = min(x + K + 1, int(c.X));
    for (int zz = z0; zz < z1; zz++) {
        uint dz = uint((zz - z) * (zz - z));
        for (int yy = y0; yy < y1; yy++) {
            uint dzy = dz + uint((yy - y) * (yy - y));
            uint row = c.base + (uint(zz) * c.Y + uint(yy)) * c.X;
            for (int xx = x0; xx < x1; xx++) {
                uint h = row + uint(xx);
                uint d2 = dzy + uint((xx - x) * (xx - x));
                if (!(a2 * float(d2) < rr) || c.pid[h] != c.me) continue;
                uint li = c.local[h];
                c.valid[li >> 5] &= ~(1u << (li & 31));
            }
        }
    }
    return uint((z1 - z0) * (y1 - y0) * (x1 - x0));
}

inline float sk_weight(thread SkCtx& c, uint li) {
    return arr(c, WS_VID)[li] != NONE ? 0.0f : as_type<float>(arr(c, WS_P)[li]);
}

// pops one entry of a geodesic and relaxes its neighbours by step length; returns the popped entry
inline uint sk_geo_step(thread SkCtx& c, thread uint& hn) {
    uint i = sk_heap_pop(c, hn);
    device uint* cost = arr(c, WS_COST);
    uint ci = cost[i];
    int z, y, x;
    sk_coords(c, arr(c, WS_VOX)[i], z, y, x);
    for (int dz = -1; dz <= 1; dz++) {
        int nz = z + dz;
        if (nz < 0 || nz >= int(c.Z)) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= int(c.Y)) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                if (nx < 0 || nx >= int(c.X) || (dz == 0 && dy == 0 && dx == 0)) continue;
                uint h = c.base + (uint(nz) * c.Y + uint(ny)) * c.X + uint(nx);
                if (c.pid[h] != c.me) continue;
                uint j = c.local[h];
                uint kind = uint(dz != 0) + uint(dy != 0) + uint(dx != 0);
                uint nc = step_rule(ci, c.prm.len[kind]);
                if (nc < cost[j]) {
                    cost[j] = nc;
                    sk_heap_set(c, hn, j, nc);
                }
            }
        }
    }
    return i;
}

// the lowest-index neighbour of q, outside S and reached, whose step reproduces q's cost (§3.2 phase 2)
inline uint sk_pred(thread SkCtx& c, uint q) {
    device uint* cost = arr(c, WS_COST);
    device uint* vid = arr(c, WS_VID);
    uint cq = cost[q];
    float w = sk_weight(c, q);
    int z, y, x;
    sk_coords(c, arr(c, WS_VOX)[q], z, y, x);
    for (int dz = -1; dz <= 1; dz++) {
        int nz = z + dz;
        if (nz < 0 || nz >= int(c.Z)) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= int(c.Y)) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                if (nx < 0 || nx >= int(c.X) || (dz == 0 && dy == 0 && dx == 0)) continue;
                uint h = c.base + (uint(nz) * c.Y + uint(ny)) * c.X + uint(nx);
                if (c.pid[h] != c.me) continue;
                uint p = c.local[h];
                uint cp = cost[p];
                if (vid[p] == NONE && cp != UNREACHED && step_rule(cp, w) == cq) return p;
            }
        }
    }
    return NONE;
}

// the PDRF of §2.5 in its fixed float order (src/skel.rs `pdrf`); rb and rd are the piece's reciprocals of max B and
// max D, each correctly rounded. Every addition is an explicit fma, and the addend's product is hidden behind opq
inline float pdrf(float b, float rb, float d, float rd, uint flags, uint zero) {
    float t = b * rb;
    float u = (flags & F_UNSQUARED) != 0 ? fma(-b, rb, 1.0f) : fma(-t, t, 1.0f);
    float u2 = u * u;
    float u4 = u2 * u2;
    float u8 = u4 * u4;
    float u16 = u8 * u8;
    if ((flags & F_DAF) == 0) return 5000.0f * u16;
    return fma(5000.0f, u16, opq(d * rd, zero));
}

inline float sk_pdrf(thread SkCtx& c, float b, float rb, float d, float rd) {
    return pdrf(b, rb, d, rd, c.prm.flags, c.prm.zero);
}

// self-test: the PDRF of (b, rb, d, rd) under flags
kernel void sk_test_pdrf(device const float4* p_in [[buffer(0)]], device uint* p_out [[buffer(1)]],
                         constant uint& flags [[buffer(2)]], constant uint& zero [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    float4 x = p_in[i];
    p_out[i] = as_type<uint>(pdrf(x.x, x.y, x.z, x.w, flags, zero));
}

// starts a geodesic from entry src
inline void sk_geo_begin(thread SkCtx& c, thread SkState& s, uint geo, uint src) {
    s.geo = geo;
    s.src = src;
    s.best = src;
    s.bestc = 0;
    s.i = 0;
    arr(c, WS_COST)[src] = 0;
    s.hn = 0;
    sk_heap_set(c, s.hn, src, 0);
    s.phase = PH_GEO;
}

// the entry of a voxel of this piece
inline uint sk_entry(thread SkCtx& c, uint g) { return c.local[g]; }

kernel void sk_run(device const SkPiece* p_pieces [[buffer(0)]],
                   device SkState* p_state [[buffer(1)]],
                   device const uint* p_active [[buffer(2)]],
                   device uint* p_ws [[buffer(3)]],
                   device uint* p_valid [[buffer(4)]],
                   device const uint* p_pid [[buffer(5)]],
                   device const uint* p_local [[buffer(6)]],
                   device const uint* p_fields [[buffer(7)]],
                   device const uint* p_mand [[buffer(8)]],
                   device const SkSlab* p_slabs [[buffer(9)]],
                   device const uint* p_maxpen [[buffer(10)]],
                   constant SkParams& p_prm [[buffer(11)]],
                   constant uint& nactive [[buffer(12)]],
                   uint t [[thread_position_in_grid]]) {
    if (t >= nactive) return;
    uint k = p_active[t];
    SkPiece pc = p_pieces[k];
    SkSlab sl = p_slabs[pc.slab];
    SkCtx c;
    c.w = p_ws + pc.ws;
    c.valid = p_valid + pc.valid;
    c.pid = p_pid;
    c.local = p_local;
    c.fields = p_fields;
    c.mand = p_mand + pc.mand_off;
    c.m = pc.m;
    c.me = pc.pid;
    c.base = sl.base;
    c.Z = sl.z;
    c.Y = sl.y;
    c.X = sl.x;
    c.prm = p_prm;
    SkState s = p_state[k];
    s.runs++;
    uint budget = p_prm.budget;
    uint spent = 0;
    uint mode = (p_prm.flags >> F_ROOT_SHIFT) & 7;
    bool face_root = mode == ROOT_FACE || mode == ROOT_FACE_LAST;
    device uint* cost = arr(c, WS_COST);
    device uint* vid = arr(c, WS_VID);
    device uint* order = arr(c, WS_ORDER);
    device uint* vert = arr(c, WS_VERT);
    device uint* next = arr(c, WS_NEXT);
    device uint* touch = arr(c, WS_TOUCH);

    while (spent < budget && s.phase < PH_DONE) {
        switch (s.phase) {
        case PH_START: {
            if (face_root && pc.mand_n > 0) {
                uint g = mode == ROOT_FACE ? c.mand[0] : c.mand[pc.mand_n - 1];
                s.root = sk_entry(c, g);
                sk_geo_begin(c, s, 1, s.root);
            } else if (mode == ROOT_DBF) {
                s.phase = PH_DBFMAX;
                s.i = 0;
                s.best = NONE;
            } else {
                uint g = mode == ROOT_F ? pc.ffirst : pc.first;
                sk_geo_begin(c, s, 0, sk_entry(c, g));
            }
            spent++;
            break;
        }
        case PH_DBFMAX: {
            // the largest DBF, lowest voxel index among ties
            device uint* vox = arr(c, WS_VOX);
            for (; s.i < c.m && spent < budget; s.i++, spent++) {
                uint g = vox[s.i];
                uint n = c.fields[c.prm.dbf_field + g];
                if (s.best == NONE || n > s.bestc || (n == s.bestc && g < vox[s.best])) {
                    s.best = s.i;
                    s.bestc = n;
                }
            }
            if (s.i == c.m) sk_geo_begin(c, s, 0, s.best);
            break;
        }
        case PH_GEO: {
            while (s.hn > 0 && spent < budget) {
                uint i = sk_geo_step(c, s.hn);
                spent += 4;
                uint ci = cost[i];
                if (s.geo == 0) {
                    // pops come in increasing order, so the first pop of the largest cost is the lowest index
                    if (ci > s.bestc) {
                        s.best = i;
                        s.bestc = ci;
                    }
                } else {
                    order[s.i++] = i;
                    s.maxd = ci;
                }
            }
            if (s.hn > 0) break;
            if (s.geo == 1) {
                s.phase = PH_ORDER_REV;
                s.i = 0;
                break;
            }
            if (mode == ROOT_FACE_FAR && pc.mand_n > 0) {
                // the farthest border target from the C-order first voxel, lowest index among ties
                uint bt = NONE, bc = 0, bg = 0;
                for (uint q = 0; q < pc.mand_n; q++) {
                    uint g = c.mand[q];
                    uint e = sk_entry(c, g);
                    uint d = cost[e];
                    if (bt == NONE || d > bc || (d == bc && g < bg)) {
                        bt = e;
                        bc = d;
                        bg = g;
                    }
                }
                s.root = bt;
            } else {
                s.root = s.best;
            }
            s.phase = PH_RESET;
            s.i = 0;
            break;
        }
        case PH_RESET: {
            for (; s.i < c.m && spent < budget; s.i++, spent++) cost[s.i] = UNREACHED;
            if (s.i == c.m) sk_geo_begin(c, s, 1, s.root);
            break;
        }
        case PH_ORDER_REV: {
            // D's pops ascend by (D, index); reversed they descend by D, but ties come out by index descending
            uint mid = c.m / 2;
            for (; s.i < mid && spent < budget; s.i++, spent++) {
                uint a = order[s.i];
                order[s.i] = order[c.m - 1 - s.i];
                order[c.m - 1 - s.i] = a;
            }
            if (s.i == mid) {
                s.phase = PH_ORDER_TIES;
                s.i = 0;
            }
            break;
        }
        case PH_ORDER_TIES: {
            // reverse each run of equal D back to ascending index
            while (s.i < c.m && spent < budget) {
                uint d = cost[order[s.i]];
                uint e = s.i + 1;
                while (e < c.m && cost[order[e]] == d) e++;
                for (uint a = s.i, b = e - 1; a < b; a++, b--) {
                    uint x = order[a];
                    order[a] = order[b];
                    order[b] = x;
                }
                spent += e - s.i;
                s.i = e;
            }
            if (s.i == c.m) {
                s.phase = PH_FIELD;
                s.i = 0;
            }
            break;
        }
        case PH_FIELD: {
            device uint* vox = arr(c, WS_VOX);
            device uint* pf = arr(c, WS_P);
            float rb = div_rn(1.0f, sk_radius(c, p_maxpen[k]));
            float rd = div_rn(1.0f, as_type<float>(s.maxd));
            bool stop = (p_prm.flags & F_FIELDS_STOP) != 0;
            for (; s.i < c.m && spent < budget; s.i++, spent += 2) {
                float b = sk_radius(c, c.fields[c.prm.pen_field + vox[s.i]]);
                pf[s.i] = as_type<uint>(sk_pdrf(c, b, rb, as_type<float>(cost[s.i]), rd));
                if (stop) {
                    // the heap is empty after the geodesic, so its arrays are free
                    arr(c, WS_HCOST)[s.i] = as_type<uint>(sk_radius(c, c.fields[c.prm.dbf_field + vox[s.i]]));
                    arr(c, WS_HLOC)[s.i] = as_type<uint>(b);
                } else {
                    cost[s.i] = UNREACHED;
                }
            }
            if (s.i == c.m) s.phase = stop ? PH_DONE : PH_ROOT;
            break;
        }
        case PH_ROOT: {
            vid[s.root] = 0;
            vert[0] = s.root;
            s.nverts = 1;
            s.nt = 0;
            s.ord = 0;
            s.mand = 0;
            s.ordinary = 0;
            spent += sk_invalidate(c, s.root);
            s.phase = PH_PICK;
            break;
        }
        case PH_PICK: {
            bool mand_first = (p_prm.flags & F_MAND_FIRST) != 0;
            uint tgt = NONE;
            if (mand_first && s.mand < pc.mand_n) {
                tgt = sk_entry(c, c.mand[s.mand++]);
            } else if (s.ord < c.m) {
                uint e = order[s.ord++];
                spent++;
                if (!sk_valid(c, e)) break;
                tgt = e;
                s.ordinary++;
            } else if (!mand_first && s.mand < pc.mand_n) {
                tgt = sk_entry(c, c.mand[s.mand++]);
            } else {
                s.phase = PH_DONE;
                break;
            }
            if (vid[tgt] != NONE) {
                // a target already in S only invalidates its own cube
                s.len = 0;
                s.join = tgt;
                s.i = 0;
                s.phase = PH_INVAL;
                break;
            }
            uint c0 = (p_prm.flags & F_COUNT_TARGET) != 0 ? arr(c, WS_P)[tgt] : 0;
            s.src = tgt;
            cost[tgt] = c0;
            touch[0] = tgt;
            s.nt = 1;
            s.hn = 0;
            sk_heap_set(c, s.hn, tgt, c0);
            s.phase = PH_SEARCH;
            break;
        }
        case PH_SEARCH: {
            while (spent < budget) {
                // S is never empty and the piece is connected, so the heap never runs dry first
                if (s.hn == 0) {
                    s.phase = PH_FAIL;
                    break;
                }
                uint i = sk_heap_pop(c, s.hn);
                spent += 4;
                if (vid[i] != NONE) {
                    s.join = i;
                    s.len = 0;
                    s.i = i;
                    s.phase = PH_WALK;
                    break;
                }
                uint ci = cost[i];
                int z, y, x;
                sk_coords(c, arr(c, WS_VOX)[i], z, y, x);
                for (int dz = -1; dz <= 1; dz++) {
                    int nz = z + dz;
                    if (nz < 0 || nz >= int(c.Z)) continue;
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny = y + dy;
                        if (ny < 0 || ny >= int(c.Y)) continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = x + dx;
                            if (nx < 0 || nx >= int(c.X) || (dz == 0 && dy == 0 && dx == 0)) continue;
                            uint h = c.base + (uint(nz) * c.Y + uint(ny)) * c.X + uint(nx);
                            if (c.pid[h] != c.me) continue;
                            uint j = c.local[h];
                            uint nc = step_rule(ci, sk_weight(c, j));
                            uint cj = cost[j];
                            if (nc < cj) {
                                if (cj == UNREACHED) touch[s.nt++] = j;
                                cost[j] = nc;
                                sk_heap_set(c, s.hn, j, nc);
                            }
                        }
                    }
                }
            }
            break;
        }
        case PH_WALK: {
            // from the join back to the target; s.i is the current entry, the path lands in vert after the vertices
            while (s.i != s.src && spent < budget) {
                uint p = sk_pred(c, s.i);
                if (p == NONE) {
                    s.phase = PH_FAIL;
                    break;
                }
                vert[s.nverts + s.len] = p;
                s.len++;
                s.i = p;
                spent += 2;
            }
            if (s.i != s.src || s.phase == PH_FAIL) break;
            // the path in order from the target, its vertex numbers and each vertex's edge to the next path voxel
            uint b = s.nverts, L = s.len;
            for (uint a = 0; a < L / 2; a++) {
                uint x = vert[b + a];
                vert[b + a] = vert[b + L - 1 - a];
                vert[b + L - 1 - a] = x;
            }
            for (uint a = 0; a < L; a++) vid[vert[b + a]] = b + a;
            for (uint a = 0; a + 1 < L; a++) next[b + a] = b + a + 1;
            next[b + L - 1] = vid[s.join];
            spent += L;
            s.phase = PH_CLEAN;
            s.i = 0;
            break;
        }
        case PH_CLEAN: {
            device uint* pos = arr(c, WS_POS);
            for (; s.i < s.nt && spent < budget; s.i++, spent++) {
                uint e = touch[s.i];
                cost[e] = UNREACHED;
                pos[e] = NONE;
            }
            if (s.i == s.nt) {
                s.nt = 0;
                s.hn = 0;
                s.phase = PH_INVAL;
                s.i = 0;
            }
            break;
        }
        case PH_INVAL: {
            // the path's voxels: the new vertices, then the join
            for (; s.i <= s.len && spent < budget; s.i++) {
                uint v = s.i < s.len ? vert[s.nverts + s.i] : s.join;
                spent += sk_invalidate(c, v);
            }
            if (s.i > s.len) {
                s.nverts += s.len;
                s.phase = PH_PICK;
            }
            break;
        }
        default:
            s.phase = PH_FAIL;
            break;
        }
    }
    s.steps += spent;
    p_state[k] = s;
}
