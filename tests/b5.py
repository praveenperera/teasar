"""B5 synthetic shapes (spec part B rev 3.1 §4.5): the masks, the reference output for them, and the pass conditions.

Track points are physical units (pooled index × 2 at ``published_0211``), which equal full-resolution voxels; the
reference's ``ends``, ``leaves`` and ``branch_points`` are pooled indices.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from fixtures import data, need

SYNTH = "fixtures/synthetic"
A = 2.0
BARS = {
    "bar_1x5x200_padded": {1: 3},
    "bar_low_border": {1: 2},
    "bar_high_border": {1: 7},
    "two_bars_both_borders": {1: 2, 2: 9},
}
SOLIDS = ("solid_1x5x200", "solid_3x9x60", "solid_one_hole")
# the line-stub masks: a forced line on row 5 with a stub at column 100; the stub tips the strict ball leaves valid
LINES = ("line_stub1", "line_stub2", "line_knight", "line_stub3")
# B5 conditions that compare against the reference's choice of next ordinary target, which the spec leaves to the
# 'kimimaro' addendum; before it lands a miss there is a known gap, reported, not a failure (spec part B §4, §4.5)
KNOWN_GAP = ("L",)


def masks() -> dict[str, np.ndarray]:
    m = np.load(data(SYNTH, "b5_masks.npz"))
    return {k: np.ascontiguousarray(m[k]) for k in m.files}


def reference() -> dict:
    return json.loads(data(SYNTH, "b5_reference.json").read_text())


def tracks(packed) -> list[np.ndarray]:
    """villa's tracks for one Packed at offset (0, 0, 0)."""
    # the villa checkout of spec part A §0 and §6, for its fast_tracks; it is not part of this repo, so there is no
    # default
    villa = need("TEASAR_VILLA", "a villa checkout (the directory holding spiral-fitting/fast_tracks.py)")
    sys.path.insert(0, str(villa / "spiral-fitting"))
    import fast_tracks

    return [np.asarray(t, float) for t in fast_tracks.tracks_from_packed(packed.v_off, packed.e_off, packed.vertices,
                                                                          packed.edges, (0, 0, 0))]


def _nearest(pts: np.ndarray, other: np.ndarray) -> np.ndarray:
    if len(other) == 0:
        return np.full(len(pts), np.inf)
    d = np.linalg.norm(pts[:, None, :] - other[None, :, :], axis=-1)
    return d.min(axis=1)


def _interior(track_list: list[np.ndarray], margin: float = 8.0) -> np.ndarray:
    """Points farther than `margin` voxels from both ends of their own track."""
    keep = []
    for t in track_list:
        far = (np.linalg.norm(t - t[0], axis=1) > margin) & (np.linalg.norm(t - t[-1], axis=1) > margin)
        keep.append(t[far])
    return np.concatenate(keep) if keep else np.zeros((0, 3))


def matches(cand: list[np.ndarray], ref: list[np.ndarray], tol: float = 2.0) -> tuple[bool, float]:
    """Every interior point of each side within `tol` voxels of some point of the other; returns the worst gap."""
    allc = np.concatenate(cand) if cand else np.zeros((0, 3))
    allr = np.concatenate(ref) if ref else np.zeros((0, 3))
    worst = max(_nearest(_interior(cand), allr).max(initial=0), _nearest(_interior(ref), allc).max(initial=0))
    return bool(worst <= tol), float(worst)


def symmetries(mask: np.ndarray) -> list:
    """The mask's symmetries as maps of pooled (z, y, x) point arrays: each axis reflection about the foreground's
    bounding-box centre, and the y/x swap, that take the foreground onto itself, and their products"""
    fg = np.argwhere(mask != 0)
    lo, hi = fg.min(axis=0), fg.max(axis=0)
    pts = set(map(tuple, fg))
    out = []
    for swap in (False, True):
        for flips in range(8):
            def f(p, swap=swap, flips=flips):
                q = np.array(p, float)
                for ax in range(3):
                    if flips >> ax & 1:
                        q[..., ax] = lo[ax] + hi[ax] - q[..., ax]
                if swap:
                    q = q[..., [0, 2, 1]]
                return q

            if set(map(tuple, f(fg).astype(int))) == pts:
                out.append(f)
    return out


def best_image(score, mask: np.ndarray) -> float:
    """The smallest of `score(image)` over the mask's symmetries, `image` mapping pooled points"""
    return min(score(f) for f in symmetries(mask))


def degrees(packed, k: int = 0) -> np.ndarray:
    lo, hi = packed.v_off[k], packed.v_off[k + 1]
    e = packed.edges[packed.e_off[k] : packed.e_off[k + 1]]
    return np.bincount(e.ravel(), minlength=hi - lo)


def branch_points(packed) -> np.ndarray:
    """Pooled indices of vertices of degree 3 or more, over all skeletons."""
    out = []
    for k in range(len(packed.ids)):
        lo = packed.v_off[k]
        deg = degrees(packed, k)
        out.append(packed.vertices[lo + np.nonzero(deg >= 3)[0]] / A)
    return np.concatenate(out) if out else np.zeros((0, 3))


def is_forest(packed) -> bool:
    """Every skeleton is acyclic: union-find finds no edge inside one tree."""
    for k in range(len(packed.ids)):
        n = packed.v_off[k + 1] - packed.v_off[k]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in packed.edges[packed.e_off[k] : packed.e_off[k + 1]]:
            ru, rv = find(int(u)), find(int(v))
            if ru == rv:
                return False
            parent[ru] = rv
    return True


def check(name: str, mask: np.ndarray, packed, ref: dict) -> list[str]:
    """The §4.5 pass conditions for one shape; returns the failures."""
    fails: list[str] = []
    cand = tracks(packed)
    rtracks = [np.asarray(t, float) for t in ref["tracks_full"]]
    if not is_forest(packed):
        fails.append("not a forest")

    if name in BARS:
        for label, row in BARS[name].items():
            mine = [t for t in cand if mask[tuple((t[len(t) // 2] / A).astype(int))] == label]
            if not 1 <= len(mine) <= 2:
                fails.append(f"label {label}: {len(mine)} tracks")
                continue
            pts = np.concatenate(mine)
            share = float(np.mean(pts[:, 1] == row * A))
            if share < 0.85:
                fails.append(f"label {label}: {share:.0%} of track points on row {row}")
            xs = np.nonzero((mask == label).any(axis=(0, 1)))[0]
            ends = np.concatenate([t[[0, -1], 2] for t in mine]) / A
            if min(abs(ends - xs[0]).min(), 99) > 3 or min(abs(ends - xs[-1]).min(), 99) > 3:
                fails.append(f"label {label}: track ends {sorted(ends)} not within 3 of x {xs[0]}..{xs[-1]}")
    elif name == "Y":
        bp = branch_points(packed)
        if len(cand) != 3:
            fails.append(f"{len(cand)} tracks")
        if len(bp) == 0 or np.linalg.norm(bp - np.array([0, 80, 83]), axis=1).min() > 3:
            fails.append(f"branch points {bp.tolist()} not within 3 of (80, 83)")
        worst = track_gap(cand, rtracks, mask, 2.0)
        if worst > 2.0:
            fails.append(f"tracks off the reference by {worst:.2f}")
    elif name == "L":
        if not 2 <= len(cand) <= 3:
            fails.append(f"{len(cand)} tracks")
        worst = track_gap(cand, rtracks, mask, 8.0)
        if worst > 8.0:
            fails.append(f"tracks off the reference by {worst:.2f}")
        bp = branch_points(packed)
        rbp = np.asarray(
            [p for sk in ref["skeletons"].values() for p in sk["branch_points"]], float
        ).reshape(-1, 3)
        near = best_image(lambda f: _nearest(f(rbp), bp).min(initial=np.inf), mask) if len(bp) and len(rbp) else np.inf
        if near > 4:
            fails.append(f"no branch point within 4 of the reference's {rbp.tolist()}: {bp.tolist()}")
    elif name == "ring":
        mine = packed.vertices / A
        rv = np.asarray([v for sk in ref["skeletons"].values() for v in sk["vertices"]], float)

        def gap(f):
            img = f(rv)
            return max(_nearest(mine, img).max(initial=0), _nearest(img, mine).max(initial=0))

        worst = best_image(gap, mask)
        if worst > 4:
            fails.append(f"vertices {worst:.2f} pooled voxels from the reference")
    elif name in LINES:
        fails += _check_line(name, mask, packed, ref)
    elif name == "bar_2thick_padded":
        worst = track_gap(cand, rtracks, mask, 2.0)
        if worst > 2.0:
            fails.append(f"tracks off the reference by {worst:.2f}")
        deg = degrees(packed)
        leaves = np.nonzero(deg == 1)[0]
        if len(leaves) > 3:
            fails.append(f"{len(leaves)} leaves")
        span = np.ptp(packed.vertices[:, 0]) / A
        if span < 190:
            fails.append(f"main path spans only {span:.0f} along z")
        spur = _spur_lengths(packed)
        if len(spur) > 1 or any(s > 2 for s in spur):
            fails.append(f"spurs {spur}")
    return fails


def leaf_points(packed) -> np.ndarray:
    """Pooled indices of vertices of degree 1, over all skeletons."""
    out = [packed.vertices[packed.v_off[k] + np.nonzero(degrees(packed, k) == 1)[0]] / A for k in range(len(packed.ids))]
    return np.concatenate(out) if out else np.zeros((0, 3))


def _check_line(name: str, mask: np.ndarray, packed, ref: dict) -> list[str]:
    """The line stubs: the reference's leaves exactly, and its vertex set (within one pooled voxel on line_knight,
    whose main path detours through a flat-penalty tie), up to the mask's symmetries"""
    fails = []
    sk = ref["skeletons"]["1"]
    rleaves = np.asarray(sk["leaves"], float)
    rverts = np.asarray(sk["vertices"], float)
    mine_leaves = leaf_points(packed)
    mine_verts = packed.vertices / A
    tol = 1.0 if name == "line_knight" else 0.0

    def as_set(pts):
        return {tuple(int(round(c)) for c in p) for p in pts}

    def leaf_miss(f):
        return 0 if as_set(f(rleaves)) == as_set(mine_leaves) else 1

    def vert_gap(f):
        img = f(rverts)
        if tol == 0:
            return 0.0 if as_set(img) == as_set(mine_verts) else np.inf
        return max(_nearest(mine_verts, img).max(initial=0), _nearest(img, mine_verts).max(initial=0))

    if best_image(leaf_miss, mask) != 0:
        fails.append(f"leaves {sorted(as_set(mine_leaves))} differ from the reference's {sorted(as_set(rleaves))}")
    gap = best_image(vert_gap, mask)
    if gap > tol:
        fails.append(f"vertex set {'differs' if tol == 0 else f'off by {gap:.2f}'} from the reference's")
    return fails


def track_gap(cand: list[np.ndarray], rtracks: list[np.ndarray], mask: np.ndarray, tol: float) -> float:
    """The worst interior gap between the candidate's tracks and the reference's nearest symmetric image"""
    # tracks are physical units; images map pooled points
    return best_image(lambda f: matches(cand, [f(t / A) * A for t in rtracks], tol)[1], mask)


def _spur_lengths(packed) -> list[int]:
    """Vertex counts of the branches off a skeleton's longest path, for a tree with one branch point at most."""
    deg = degrees(packed)
    n = len(deg)
    adj = [[] for _ in range(n)]
    for u, v in packed.edges[packed.e_off[0] : packed.e_off[1]]:
        adj[u].append(v)
        adj[v].append(u)
    spurs = []
    for leaf in np.nonzero(deg == 1)[0]:
        length, prev, cur = 0, -1, int(leaf)
        while deg[cur] <= 2 and length <= n:
            length += 1
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            prev, cur = cur, int(nxt[0])
        spurs.append(length)
    spurs.sort()
    # the two longest walks are the main path's two halves
    return spurs[:-2]
